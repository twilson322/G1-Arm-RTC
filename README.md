# G1 Arm RTC: A Real-Time Control Stack for a Humanoid Upper Body

A 1 kHz C++ control stack for the 10-DOF upper body of a Unitree G1 (three waist joints and the seven-joint right arm) running against MuJoCo. The stack sees only what the robot's hardware would report: quantized encoder positions, filtered velocities, a noisy wrist force/torque sensor and the torque the drives say they applied. Commands arrive half a millisecond late, pass a current limit, and stop being applied if they stop arriving.

On top of that sit five torque controllers (joint PD with gravity compensation, computed torque, operational-space control, Cartesian impedance and hybrid force/motion), a momentum observer that estimates external joint torque without torque sensors, collision detection and contact-link isolation built on that residual, online payload identification, a supervisor state machine with four collision reflexes, and a planning pipeline (IK, RRT-Connect, splines and torque-limited TOPP-RA). The same `ControlStack` object runs in a deterministic lockstep harness for experiments, in a threaded runtime with `SCHED_FIFO` threads and lock-free buffers, and behind a ROS 2 Jazzy node.

At the calibrated thresholds the observer detected all 48 injected collisions (5 to 40 N on four links) with a median latency of 11 ms and raised no false alarm on four held-out collision-free motions. Feeding the online payload estimate back into the model cut figure-8 tracking error under an unmodelled 1 kg payload from 20.9 to 0.75 mm RMS. The hybrid controller held a 10 N wiping force within 1.2 N RMS while the table height was wrong by up to 10 mm.

Everything here is simulation, and the controller's dynamics model is built from the same MJCF file as the simulated robot. Read [Limitations](#limitations) before reading any number as hardware performance. [REVIEW.md](REVIEW.md) is a staff-level code review of this repository, including what it found and what is still open.

![Collision detection](results/fig_collision.png)

## The robot and what the controller sees

The scene is the Unitree G1 model from MuJoCo Menagerie (BSD-3-Clause, license copied to `models/LICENSE.unitree_g1`), cut down by `tools/build_scene.py` to the pelvis, the waist and the right arm, with the pelvis welded 0.793 m above the floor. A 3 cm sponge sphere on the hand is the tool. The world holds a table, a shelf and a mocap "intruder" capsule that experiments can move into the arm's path. MuJoCo runs at 2 kHz and the control stack at 1 kHz.

| What hardware would do | How `RobotSim` models it |
|---|---|
| Joint encoders | positions quantized to 2^18 counts per output revolution |
| Joint velocity | differenced from the quantized positions at 2 kHz, then a 150 Hz first-order filter |
| Wrist F/T sensor | 0.3 N and 0.01 N m white noise, reads the tool's own weight, unknown offset removed by taring |
| Command transport | one physics step (0.5 ms) of delay |
| Drive | current limit at each joint's torque limit, optional torque-constant error, reports the integral of the torque it applied |
| Drive watchdog | no new command for 5 ms switches the drive to 3 N m s/rad damping and latches a fault until a command carries a reset flag |
| Joint friction | 0.3 N m Coulomb friction per joint (MuJoCo `frictionloss`), which the controllers do not model |

Ground truth (true state, contact forces, true external joint torques) is exposed separately and used only for scoring.

## Architecture

```
            non-real-time                        |              real-time (1 kHz, no allocation, no locks)
                                                 |
  MotionPlanner (IK, RRT-Connect, spline, TOPP)  |   ControlStack::tick(state)
  ROS 2 node / scenario script                   |     sensing      F/T compensation, dynamics cache, momentum observer
        |  post(task)                            |     monitoring   contact-aware collision detection and isolation
        +---------> TaskMailbox (RCU-style) -----+---> tasks        start-pose check
        |  requestEnable/Start/Stop/Reset        |     supervisor   Disabled, Enabling, Hold, Executing, Reflex, SafeStop
        +---------> Supervisor atomics ----------+---> control      controller for the task's tracking mode, saturation
        |                                        |     adaptation   payload RLS written back into the model
        <---------- telemetry triple buffer -----+---- logging      fixed-size record into an SPSC queue
                                                 |
  robot thread (2 kHz): RobotSim <--- command triple buffer --- control thread
                                  ---- state triple buffer ---->
```

`Lockstep` runs the same tick deterministically: two physics steps, one control tick, no wall clock. Every experiment below reproduces exactly from its seed. `Runtime` runs it on wall-clock threads, where a late control thread shows up as a stale command that the drive watchdog catches, not as a stalled robot.

## How to run

```bash
./tools/fetch_assets.sh                      # Menagerie G1 meshes (sparse clone) and the scene
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build
(cd build && ctest --output-on-failure)      # 41 GoogleTest cases
./build/g1rtc_experiments all results        # or one of: collision observer_floor payload planning reaction timing tracking wipe
./build/g1rtc_rt --tag nominal --out results # threaded scenario; --stress N adds CPU hogs, --no-rt skips SCHED_FIFO
./build/g1rtc_bench                          # Google Benchmark timings
python3 scripts/plot.py                      # figures and results/summary.md
./scripts/run_all.sh                         # all of the above, about 15 minutes on 2 cores
```

Sanitizer builds use `-DG1RTC_SANITIZE=ON` (ASan and UBSan) or `-DG1RTC_TSAN=ON`. `docker build -t g1-arm-rtc .` builds and tests on Ubuntu 24.04. `docker build -f docker/ros2.Dockerfile -t g1-arm-rtc-ros . && docker run --rm g1-arm-rtc-ros` builds the ROS 2 Jazzy node with colcon and runs an end-to-end check of its topics and services.

## Results

All numbers come from `results/experiments.txt` and the CSVs next to it. `results/summary.md` has every table.

### Momentum observer noise floor

The observer residual $r$ obeys $\dot r = K(\tau_{ext} - r)$ only if the model, the sensing and the integration are exact. Anything else lands in $r$, and the collision thresholds have to sit above it. The table shows the largest $|r|$ on the waist joints (the noisiest group) during collision-free motion, changing one thing at a time from the deployed configuration.

| Configuration | Figure 8 [N m] | Fast shelf move [N m] |
|---|---|---|
| Deployed: 18-bit encoders, K = 30 /s, friction model, filter matching | 0.43 | 0.54 |
| Without matching the 150 Hz velocity-filter delay | 0.43 | 1.92 |
| Without the friction model | 0.71 | 0.69 |
| 16-bit encoders | 1.72 | |
| Observer gain 60 /s | 0.90 | |
| Near-ideal sensing (60-bit, no velocity filter), friction modelled | 0.23 | |
| Near-ideal sensing, no plant friction | 0.007 | |

The last row checks the observer's dynamics against MuJoCo's: with ideal sensing and no unmodelled friction the residual is numerical noise. Filter matching (passing $\tau$ and $\beta$ through the same first-order filter as the drive's velocity estimate) does nothing on the slow figure 8 and cuts the floor 3.6x on the 0.3 s shelf move.

![Observer floor](results/fig_observer.png)

### Collision detection and isolation

Thresholds are calibrated per joint as 1.5x the largest unexplained residual over four collision-free motions (two figure-8 speeds, the shelf move and the wiping task), then checked for false alarms on four different motions (another figure-8 speed, a figure 8 with the hand rotating, the shelf move in reverse and wiping a table 5 mm higher). The 48 hits are 150 ms forces of 5, 10, 20 and 40 N in random directions on the shoulder-yaw link, the elbow link, the wrist and the tool, at random times during a figure 8.

| Threshold scale | Friction model | Detected | Clean runs that tripped | Median / p90 latency | Isolation exact | Within one link |
|---|---|---|---|---|---|---|
| 0.75 | yes | 48/48 | 0/4 | 8 / 22 ms | 25/48 | 44/48 |
| 1.0 | yes | 48/48 | 0/4 | 11 / 28 ms | 27/48 | 46/48 |
| 1.5 | yes | 46/48 | 0/4 | 15 / 51 ms | 29/46 | 44/46 |
| 1.0 | no | 44/48 | 0/4 | 11 / 41 ms | 14/44 | 29/44 |
| 0.5 | no | 0/48 | 4/4 | | | |

The friction model buys sensitivity, not just a lower floor: without it the calibrated thresholds rise 25 to 167 % and a third of the 5 N hits go undetected. At half the calibrated thresholds without it, every run tripped before its hit. Isolation fits a point force at each candidate link's center of mass to the residual and picks the most proximal link that explains it. It is exact about half the time and within one link 96 % of the time. Adjacent wrist links have nearly coincident centers of mass and are hard to tell apart.

### Reaction to an unplanned obstacle

A forearm-sized capsule sits in the figure-8's path. Contact starts about half a second into the task and is detected 8 ms later in every case.

| Reaction | Peak force | Impulse over 1 s | Time in contact |
|---|---|---|---|
| Ignore (keep tracking) | 131 N | 39.3 N s | 474 ms |
| Stop (hold the pose at detection) | 12.5 N | 0.49 N s | 63 ms |
| Float (gravity compensation plus damping) | 12.5 N | 0.44 N s | 153 ms |
| Retract (residual-driven, De Luca) | 12.5 N | 0.26 N s | 77 ms |

The 12.5 N peak is reached in the 8 ms before detection, so no reflex can change it. What they change is how long the arm keeps pushing. Retract, which lets the observed external torque drive the arm away, ended with the smallest impulse.

![Reaction](results/fig_reaction.png)

### Figure-8 tracking with and without a payload

The tool traces a 10 x 7 cm figure 8 at 2.5 s per lap, scored on laps 4 to 6 so the payload estimate has converged.

| Controller | No payload | Unmodelled 1 kg | 1 kg, estimated online |
|---|---|---|---|
| Joint PD + gravity | 6.80 mm | 47.1 mm | |
| Cartesian impedance | 7.77 mm | 18.0 mm | |
| Computed torque | 1.89 mm | 14.3 mm | 0.70 mm |
| Operational space | 1.50 mm | 20.9 mm | 0.75 mm |

With the estimator on and no payload, the error matched the nominal row to 0.01 mm and the estimate stayed within 2 g of zero. With the payload estimated, the arm tracked better than with no payload at all. Computed torque and OSC scale their feedback by the model's inertia, so a heavier model is a stiffer servo against the unmodelled joint friction, which is what limits the nominal rows.

Payload identification is recursive least squares on the observer residual with the regressor $Y(q) = [J_v^T g,\ -J_\omega^T [g]_\times R]$ for $\theta = [m,\ m c]$. With the hand at a fixed orientation the COM is not identifiable, so the estimator fits mass alone and read 1.044 kg for the true 1 kg (the COM offset and residual friction bias it). With the hand rotating ±20° it estimated 1.001 kg and a COM of (2.98, 0.05, −2.03) cm against the true (3, 0, −2) cm.

### Force-controlled wiping on an uncertain table

The sponge wipes three 6 cm circles while pushing down with 10 N. The table is moved −10, 0 and +10 mm from where the task thinks it is.

| Controller | Mean force | RMS force error | Tangential error |
|---|---|---|---|
| Hybrid, force PI | 9.95 to 9.96 N | 1.15 to 1.18 N | 5.2 to 5.5 mm |
| Hybrid, force PI + surface friction feed-forward | 9.95 to 9.96 N | 1.22 to 1.23 N | 1.34 to 1.43 mm |
| Hybrid, feed-forward force only | 9.61 to 9.97 N | 2.40 to 2.51 N | 5.4 to 5.5 mm |
| OSC, commanded 1 cm into the nominal surface | 0.50 / 4.55 / 8.92 N | 9.55 / 5.75 / 1.68 N | 1.4 to 7.6 mm |

Position control turns a height error into a force error through whatever stiffness the controller and contact happen to have. The force loop does not care where the table is. Estimating the surface friction coefficient online from the F/T ratio and pushing ahead along the path by $\hat\mu f_n$ cut the tangential error 4x. The detector stayed quiet during every PI run. The feed-forward-only controller's larger force transients tripped it on two of three tables, which is why those baselines are scored with detection off.

![Wiping](results/fig_wipe.png)

### Planning around the shelf

The arm moves from under a shelf to above it. The straight joint-space segment hits the board, so the planner uses RRT-Connect, shortcutting, a natural cubic spline refined until it is valid, and TOPP-RA.

RRT-Connect solved all 20 seeds with a median 32 ms for planning plus shortcutting. The time parameterizations were then executed on the robot with computed torque:

| Time scaling | Duration | Planned peak $\lvert\tau\rvert/\tau_{max}$ | Executed peak | Saturated ticks | RMS joint error |
|---|---|---|---|---|---|
| TOPP-RA, velocity and acceleration limits | 0.306 s | 1.44 | 1.00 | 27 | 10.2 mrad |
| TOPP-RA, plus 80 % torque limit | 0.313 s | 0.80 | 0.93 | 0 | 9.0 mrad |
| Uniform trapezoid | 1.101 s | 0.30 | 0.32 | 0 | 11.5 mrad |

The torque-limited profile costs 2 % in duration and keeps the drives out of saturation, leaving 20 % of each joint's torque to the feedback terms, which used 13 points of it. The discretization study in `results/topp_grid.csv` shows why TOPP-RA's interpolation constraints matter: with 100 grid points, enforcing limits only at the grid points (collocation) let the sampled torque reach 0.95 of the limit against a budget of 0.80, and interpolation held it at 0.85. The planner now checks the sampled torque itself and refines the grid until it is within 5 % of the budget.

![Planning](results/fig_planning.png)

### Timing

Lockstep, wall clock, 8000 ticks each on a 2-core cloud VM (`results/tick_timing.csv`):

| Mode | Mean tick, FTZ off | Mean tick, FTZ on |
|---|---|---|
| Hold | 88.3 µs | 21.9 µs |
| Operational space | 20.5 µs | 19.7 µs |
| Computed torque | 18.9 µs | 18.7 µs |

With the arm at rest, values decaying toward zero reach the denormal range, and Hold ran 4x slower until flush-to-zero was set on the control thread. Google Benchmark (`results/bench.txt`) gives RNEA 1.3 µs, CRBA 1.0 µs, the observer's bias term with dual numbers 10.6 µs, and the full per-tick dynamics update 5.9 µs.

The threaded scenario (enable, move under the shelf, plan around it, inject a 20 ms control-thread stall, reset, return home) ran with `SCHED_FIFO` granted on a shared, non-RT kernel:

| Run | Control ticks | Deadline misses | Wake-up p50 / p99 / max | Compute p50 / p99 | Moves | Unplanned faults |
|---|---|---|---|---|---|---|
| nominal | 3596 | 15 | 42 / 262 / 3904 µs | 27 / 59 µs | 3/3 | 0 |
| two CPU hogs | 3800 | 13 | 20 / 286 / 3277 µs | 20 / 42 µs | 3/3 | 0 |

The injected stall tripped the drive watchdog both times. After the reset the arm had sagged into a joint-limit margin, the planner moved it out with a slow unchecked escape segment, and the move home completed. A kernel without `PREEMPT_RT` cannot bound the 3 to 4 ms worst-case wake-ups, and on this VM nothing else was competing at real-time priority.

![RT latency](results/fig_rt.png)

## What worked

Contact-aware detection kept collision monitoring on during the wiping task. The joint torque explained by the measured tool force, $J^T f$, is subtracted from the residual. It has to be passed through the same first-order lag as the observer first, because the residual sees external torque through $K/(s+K)$. An unmatched 30 Hz filter let the touchdown transient trip the detector.

Integrating the observer with the drives' torque impulse and the measured time between state samples made it insensitive to the control thread's jitter. A regression test drives the stack with 1, 1.5 and 2 ms periods at random. With the period fixed at a nominal 1 ms the residual reached 0.59 N m, above the smallest threshold, from jitter alone.

Exact $\dot M$ from forward-mode dual numbers through the same templated RNEA and CRBA code gave the observer's $\beta = g - C^T\dot q$ without hand-derived Christoffel terms. The test compares it against a central finite difference and checks the passivity identity $\dot q^T(\dot M - 2C)\dot q = 0$.

The supervisor's Hold mode is bumpless: its setpoint is the measured pose on entry, so every transition into it is torque-continuous.

## What did not work

These are the failures found while building the stack and during the review pass, and how each was resolved.

The observer first used a fixed 1 ms period. In the threaded runtime the control thread sometimes sees two robot cycles at once, and the residual tripped the collision detector during free motion. It now integrates the drives' reported torque impulse over the measured period.

After a drive fault the arm sags with only damping. The joint-limit check was absolute, so an arm resting inside the 10 mrad margin tripped SafeStop again on the first Hold tick and could never be re-enabled. Inside the margin it now trips only on motion toward the limit. The planner then refused every plan from that sagged pose until a start-repair step was added that searches for a nearby valid configuration and prepends a slow move to it, as MoveIt's `FixStartStateCollision` adapter does.

Shortcutting left a waypoint 3 mrad from the start of the shelf path. The natural cubic spline bent through it with $\lvert q''\rvert = 690$, which the TOPP grid stepped over: the sampled torque reached 2.7x the actuator limit at 100 grid points and the uniform trapezoid slowed the whole move to 2.9 s. Dropping waypoints closer than 0.05 rad (where the skipping segment is valid) brought the peak curvature to 13, the trapezoid to 1.1 s, and the 100-point grid to 6 % over budget. Nothing had checked the sampled trajectory against its limits before. The planner does now.

The ROS end-to-end check then failed its second move with `spline_invalid`. From the pose the arm actually reached, that RRT-Connect seed produced waypoints whose spline could not be refined into a valid one, On a sweep of 40 seeds over the same query, 1 failed this way with close waypoints dropped and none without that step, since a spline through fewer knots cuts corners harder. The planner now retries path finding with fresh seeds, up to three times, rather than giving up the curvature fix.

A stiff sponge on a rigid table chattered in and out of contact every few milliseconds while sliding (4.3 N RMS force error). A softer contact time constant and elliptic friction cones brought it to 1.15 N. Isolated 1 to 5 ms contact losses remain in the fastest part of each lap, about 0.7 % of samples, and were not investigated further.

The guarded approach first commanded a 1 N force toward the table, which could not break 0.3 N m of joint stiction, so the tool hovered. It is now a velocity servo along the push direction shaped by the task-space inertia.

The payload estimator's first forgetting factor was chosen as if it applied once per tick. It is applied once per joint, ten times per tick, so the memory was a tenth of what was intended and the estimate followed the residual's noise. 0.99998 gives about 5 s.

The first isolation rule picked the most distal joint above its threshold. A hit on one link loads every joint proximal to it, and the joint that crosses its threshold first is often not the struck link's, so it was replaced by residual fitting, which is still exact only about half the time.

## Tools and techniques applied

On the control side the project implements gravity-compensated joint PD, computed torque with bandwidth-set gains, operational-space control with the dynamically consistent null-space projector $\bar J = M^{-1}J^T\Lambda$ and a regularized $\Lambda$, Cartesian impedance with inertia-scaled damping, hybrid force/motion control with a force PI loop, anti-windup, a guarded approach and online surface-friction compensation, and torque saturation. Estimation covers the generalized-momentum observer with friction and filter-delay compensation, threshold calibration with held-out validation, contact-link isolation by least-squares residual fitting, F/T taring and gravity compensation, and RLS with forgetting and covariance capping for payload mass and COM.

Dynamics are hand-written: world-frame recursive Newton-Euler, the composite rigid-body algorithm, point Jacobians and $\dot J\dot q$, all templated on the scalar type so the same code runs on forward-mode dual numbers. They match MuJoCo to 1e-9 or better in the tests. Planning uses damped-least-squares IK with null-space posture control, a sampled allowed-collision matrix, a state validity checker, RRT-Connect, random shortcutting, natural cubic splines with validity-driven refinement, TOPP-RA with velocity, acceleration and torque constraints solved as 2-variable LPs, and post-hoc verification of the sampled result.

The software is C++20 with Eigen fixed-capacity types so the control tick never allocates (a test replaces global `operator new` to prove it), an RCU-style mailbox for handing tasks to the RT thread, triple buffers, an SPSC log queue, `SCHED_FIFO` with `mlockall`, absolute-deadline periodic timers, FTZ/DAZ, a drive watchdog, a supervisor state machine with an allocation-free transition ring, CMake, GoogleTest, AddressSanitizer, UBSan, ThreadSanitizer, Google Benchmark, Docker, GitHub Actions, and a ROS 2 Jazzy node (`rclcpp`, topics, services, a launch file and an `rclpy` checker). Analysis is Python with pandas and Matplotlib.

## Topics from the list used here

- Classical control (1A): PD and PI loops, feedforward, anti-windup, saturation handling, low-pass filtering and filter delay.
- Linear systems and modern control (1B): state observers (the momentum observer is a first-order disturbance observer), multivariable feedback.
- Nonlinear control (1E): feedback linearization (computed torque), operational-space control, impedance control, hybrid force/motion control.
- Robust control (1F): disturbance observers, friction and payload uncertainty, threshold calibration against a measured noise floor.
- Adaptive control (1G): recursive least squares with forgetting, online parameter estimation, indirect adaptation by writing the estimate back into the model.
- Mathematical foundations (2A): rotation matrices, the matrix logarithm for orientation error, skew-symmetric matrices, automatic differentiation with dual numbers.
- Rigid-body kinematics (2B): forward kinematics, geometric Jacobians, $\dot J\dot q$, damped-least-squares inverse kinematics, null-space control, singularities.
- Robot dynamics (2C): recursive Newton-Euler, composite rigid-body algorithm, mass matrix, Coriolis and gravity terms, the $\dot M - 2C$ skew-symmetry property, armature, Coulomb friction, payload modelling.
- Motion and trajectory generation (2D): splines, time-optimal path parameterization, trapezoidal profiles, velocity, acceleration and torque limits.
- Sensor modelling (3A): encoder quantization, velocity estimation by differencing and filtering, force/torque noise, sensor offsets and taring, latency.
- Sampling-based motion planning (4B): RRT-Connect, collision checking, allowed-collision matrices, shortcutting, path smoothing.
- Middleware and simulation (7): ROS 2 nodes, topics, services, QoS (transient local), launch files, MuJoCo, MJCF scene editing with `MjSpec`.
- Software engineering (8): C++, Python, Bash, CMake, Git, Docker, GitHub Actions, GoogleTest, regression and property tests, AddressSanitizer, UBSan, ThreadSanitizer, Google Benchmark, deterministic seeded experiments, ablation studies, interfaces, RAII, lock-free data structures, producer-consumer threads, memory ordering, allocation-free real-time code, denormal handling.
- Real-time control (9B): fixed-rate loops, jitter and deadline measurement, `SCHED_FIFO`, watchdogs, fail-safe states, transport delay, current limits.
- Safety and verification (12): supervisor state machines, collision detection and reaction, safe stops, bumpless transfer, fault injection.
- Manipulation (13B): contact-rich tasks, force control, collision reflexes, payload handling.
- System identification (14): payload identification, threshold calibration on held-out data, Monte Carlo over random hits, baselines and ablations.
- RoboJackets software training slides: Ubuntu, Docker, Git, classes and headers, namespaces, `const` and `constexpr`, templates, RAII, smart pointers, deleted copy operations on types that own threads or hand out `this`, `std::thread` and `join`, atomics and race conditions checked with ThreadSanitizer.

## What I learned

The noise floor sets the detector, and the floor is mostly not noise. Encoder resolution, an unmodelled 0.3 N m of friction and a 150 Hz velocity filter each contributed more residual than the sensor noise did, and two of the three could be modelled away.

Time is an input. An observer that integrates with a nominal period is wrong the moment the thread that runs it is late, and "late" happens every few seconds on an ordinary kernel. Measuring the period and integrating what the drives actually applied made the estimator independent of the scheduler.

A planner's output needs the same verification as its input. Every step of the planning pipeline checked validity, and none checked the final trajectory against the limits it was supposed to respect. A degenerate knot from an earlier step produced a torque spike that only showed up after sampling.

Recovery paths need tests as much as nominal ones. Three of the bugs fixed here only appeared after a fault: the absolute joint-limit check, the sagged start state, and a scenario script that did not handle a start rejected because a fault arrived while it was planning.

## Limitations

The controller's model is built from the same MJCF as the simulated plant. Model error comes only from what was injected deliberately (friction, payload, torque-constant error, sensing), so the tracking and detection numbers are upper bounds on what a real G1 would give. A real arm adds link mass and inertia errors, cable forces, gear backlash, temperature-dependent friction and drive dynamics.

Contact-link isolation assumes the force acts at the link's center of mass, which is exact for the injected test forces and not for real contacts on a link's surface. The start-repair escape move is not validity-checked by design. Wiping uses a soft MuJoCo contact chosen to behave like a compliant sponge. Timing comes from a 2-core cloud VM without `PREEMPT_RT`. There is no hardware result anywhere in this repository.

The next steps that would matter most are a friction feed-forward in the tracking controllers, velocity-dependent collision thresholds, isolation that estimates the contact point rather than assuming one, and running the stack against a model with deliberately perturbed inertial parameters before any hardware.

## Repository layout

```
include/g1rtc/       rbd (dynamics), robot_sim, robot_description, controllers, observer, supervisor,
                     tasks, planning, motion_planner, stack, lockstep, runtime, rt
src/                 implementations
apps/experiments/    one file per experiment, registered by name; support.cpp holds the shared rig
apps/g1rtc_rt.cpp    threaded real-time scenario with fault injection
apps/g1rtc_bench.cpp Google Benchmark timings
tests/               GoogleTest suites (dynamics vs MuJoCo, controllers, observer, planning, RT, supervisor)
ros2/g1_arm_rtc_ros/ ROS 2 package: node, launch file, end-to-end checker
models/, tools/      scene XML, Unitree license, asset fetch and scene builder
scripts/             run_all.sh, plot.py
results/             figures, summary CSVs, logs (high-rate traces are regenerated, not committed)
```
