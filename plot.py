#!/usr/bin/env python3
"""Figures and results/summary.md from the CSVs the experiments write.   python3 scripts/plot.py"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import plotstyle  # noqa: E402
from plotstyle import CONTROLLER_COLOR as CC, MUTED, SERIES  # noqa: E402

plotstyle.apply()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

RES = pathlib.Path(__file__).resolve().parents[1] / "results"
LABEL = {"joint_pd_gravity": "joint PD + gravity", "computed_torque": "computed torque",
         "operational_space": "operational space (OSC)", "cartesian_impedance": "Cartesian impedance",
         "hybrid_pi": "hybrid, force PI", "hybrid_feedforward": "hybrid, feed-forward force",
         "hybrid_pi_friction_ff": "hybrid, force PI + friction feed-forward",
         "osc_position": "OSC, press 1 cm into surface", "topp_kinematic": "TOPP-RA, velocity + acceleration",
         "topp_torque": "TOPP-RA, + 80 % torque", "trapezoid_uniform": "uniform trapezoid"}


def md(df, fmt=".3g"):
    cols = list(df.columns)
    out = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:{fmt}}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def save(fig, name, rect=None):
    fig.tight_layout(rect=rect)
    fig.savefig(RES / name)
    plt.close(fig)


def tracking(parts):
    t = pd.read_csv(RES / "tracking.csv")
    parts.append("## Figure-8 tracking (laps 4-6)\n\n" + md(t))
    ctrls = ["joint_pd_gravity", "cartesian_impedance", "computed_torque", "operational_space"]
    conds = [("nominal", "o", "no payload"), ("payload_1kg", "s", "1 kg unknown payload"),
             ("payload_1kg_adapted", "D", "1 kg, payload estimated online")]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    for i, c in enumerate(ctrls):
        for j, (cond, mk, lab) in enumerate(conds):
            row = t[(t.controller == c) & (t.condition == cond)]
            if row.empty:
                continue
            ax.scatter(row.rms_pos_mm, [i + (j - 1) * 0.18], marker=mk, s=34, color=SERIES[[7, 1, 2][j]],
                       label=lab if i == 2 else None, zorder=3)
    ax.set_yticks(range(len(ctrls)), [LABEL[c] for c in ctrls])
    ax.set_xscale("log")
    ax.set_xlabel("RMS tool position error [mm], log scale")
    ax.set_title("Figure-8 tracking, 10 x 7 cm at 2.5 s per lap")
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "fig_tracking.png")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    ref = pd.read_csv(RES / "trace_operational_space_nominal.csv")
    axes[0].plot(ref.y_ref * 100, ref.z_ref * 100, color=MUTED, lw=2.4, alpha=0.35, label="reference")
    for c, cond in [("operational_space", "nominal"), ("joint_pd_gravity", "payload_1kg"), ("operational_space", "payload_1kg")]:
        d = pd.read_csv(RES / f"trace_{c}_{cond}.csv")
        d = d[d.task_time_s > d.task_time_s.max() - 7.5]
        axes[0].plot(d.y * 100, d.z * 100, lw=1.1, color=CC[c], ls="-" if cond == "nominal" else "--",
                     label=f"{LABEL[c]}{', 1 kg payload' if cond != 'nominal' else ''}")
    axes[0].set_xlabel("y [cm]")
    axes[0].set_ylabel("z [cm]")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=7, loc="upper left")
    axes[0].set_title("Tool path, last three laps")
    for c in ["joint_pd_gravity", "cartesian_impedance", "computed_torque", "operational_space"]:
        d = pd.read_csv(RES / f"trace_{c}_payload_1kg.csv")
        axes[1].plot(d.task_time_s, d.error_mm, lw=1.0, color=CC[c], label=LABEL[c])
    axes[1].set_yscale("log")
    axes[1].set_xlabel("task time [s]")
    axes[1].set_ylabel("position error [mm]")
    axes[1].set_title("With an unmodelled 1 kg payload")
    axes[1].legend(fontsize=7)
    save(fig, "fig_figure8.png")


def payload(parts):
    p = pd.read_csv(RES / "payload.csv")
    parts.append("## Payload estimation (1 kg at (3, 0, -2) cm in the tool frame)\n\n" + md(p))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for name, col in [("no_wobble", SERIES[0]), ("wobble", SERIES[2])]:
        d = pd.read_csv(RES / f"payload_trace_{name}.csv")
        ax.plot(d.task_time_s, d.payload_estimate_kg, color=col, label="mass only, fixed orientation" if name == "no_wobble"
                else "mass + COM, hand rotating +/-20 deg")
    ax.axhline(1.0, color=MUTED, ls=":", lw=1)
    ax.set_xlabel("task time [s]")
    ax.set_ylabel("estimated payload [kg]")
    ax.set_ylim(0, 1.3)
    ax.set_title("RLS on the momentum-observer residual")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig_payload.png")


def wipe(parts):
    w = pd.read_csv(RES / "wipe.csv")
    parts.append("## Force-controlled wiping, 10 N target, table height error +/-10 mm\n\n" + md(w))
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
    for ax, off in zip(axes, [-10, 0, 10]):
        for c in ["osc_position", "hybrid_feedforward", "hybrid_pi"]:  # friction-ff variant is in the table
            f = RES / f"wipe_trace_{c}_{off}mm.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d = d[(d.task_time_s > 2.0) & (d.task_time_s < 6.0)]
            ax.plot(d.task_time_s, d.force_true_n, lw=0.9, color=CC[c], label=LABEL[c])
        ax.axhline(10, color=MUTED, ls=":", lw=1)
        ax.set_title(f"table {'+' if off > 0 else ''}{off} mm from nominal", fontsize=10)
        ax.set_xlabel("task time [s]")
    axes[0].set_ylabel("contact force [N]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    save(fig, "fig_wipe.png", rect=(0, 0, 1, 0.9))


def observer(parts):
    o = pd.read_csv(RES / "observer_floor.csv")
    parts.append("## Momentum-observer residual in collision-free motion [N m]\n\n" + md(o))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6))
    f8 = o[o.motion == "figure8"].set_index("case")
    one_at_a_time = [("deployed", "deployed\n18 bit, K 30,\n150 Hz"), ("encoder_16bit", "16 bit"), ("encoder_20bit", "20 bit"),
                     ("gain_15", "K 15 /s"), ("gain_60", "K 60 /s"), ("velocity_filter_50hz", "50 Hz\nfilter"),
                     ("velocity_filter_300hz", "300 Hz\nfilter"), ("ideal_sensing", "ideal\nsensing")]
    vals = [float(f8.loc[c, "max_r_waist"]) for c, _ in one_at_a_time]
    axes[0].bar(range(len(vals)), vals, color=[SERIES[2]] + [SERIES[0]] * (len(vals) - 1), width=0.6)
    axes[0].set_xticks(range(len(vals)), [lab for _, lab in one_at_a_time], fontsize=7)
    axes[0].set_ylabel("max |r| on the waist joints [N m]")
    axes[0].set_title("One change at a time (figure 8)")
    ablation = [("deployed", "deployed"), ("no_filter_matching", "no filter\nmatching"),
                ("no_friction_model", "no friction\nmodel"), ("no_friction_model_no_matching", "neither")]
    width = 0.38
    for k, (motion, col) in enumerate([("figure8", SERIES[0]), ("shelf", SERIES[1])]):
        d = o[o.motion == motion].set_index("case")
        axes[1].bar(np.arange(len(ablation)) + (k - 0.5) * width, [float(d.loc[c, "max_r_waist"]) for c, _ in ablation],
                    width=width, color=col, label="figure 8" if motion == "figure8" else "fast shelf move")
    axes[1].set_xticks(range(len(ablation)), [lab for _, lab in ablation], fontsize=8)
    axes[1].set_ylabel("max |r| on the waist joints [N m]")
    axes[1].set_title("What the friction model and filter matching buy")
    axes[1].legend(fontsize=8)
    save(fig, "fig_observer.png")


def collision(parts):
    d = pd.read_csv(RES / "collision_detection.csv")
    th = pd.read_csv(RES / "collision_thresholds.csv")
    cal = pd.read_csv(RES / "collision_calibration.csv")
    parts.append("## Collision detection: calibrated thresholds [N m]\n\n" +
                 md(th.pivot(index="joint", columns="friction_model", values="threshold_nm").reset_index()))
    parts.append("## Collision detection: calibration residuals [N m]\n\n" +
                 md(cal.pivot_table(index=["friction_model", "joint"], columns="motion", values="max_unexplained_residual_nm").reset_index()))
    parts.append("## Collision detection sweep (48 injected hits, 4 held-out collision-free motions)\n\n" + md(d))
    hits = pd.read_csv(RES / "collision_hits.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for fm, col, lab in [(1, SERIES[2], "with friction model"), (0, SERIES[7], "without")]:
        x = d[d.friction_model == fm].sort_values("threshold_scale")
        axes[0].plot(x.threshold_scale, 100 * x.detected / x.hits, marker="o", ms=4, color=col, label=lab)
        for _, r in x.iterrows():
            if r.false_positive_runs > 0:
                n_false = int(r.false_positive_runs)
                axes[0].annotate(f"{n_false} of {int(r.validation_runs)} clean runs tripped", (r.threshold_scale, 100 * r.detected / r.hits),
                                 fontsize=7, color=col, xytext=(6, 4), textcoords="offset points")
    axes[0].set_xlabel("threshold scale (1 = 1.5x calibrated max)")
    axes[0].set_ylabel("hits detected [%]")
    axes[0].set_title("Sensitivity vs false alarms")
    axes[0].legend(fontsize=8)
    h = hits[(hits.threshold_scale == 1.0)]
    for fm, col, lab in [(1, SERIES[2], "with friction model"), (0, SERIES[7], "without")]:
        g = h[h.friction_model == fm].groupby("force_n").detected.mean() * 100
        axes[1].plot(g.index, g.values, marker="o", ms=4, color=col, label=lab)
    axes[1].set_xlabel("hit force [N] (150 ms)")
    axes[1].set_ylabel("detected [%] at scale 1")
    axes[1].set_title("Detection rate by hit force")
    axes[1].legend(fontsize=8)
    save(fig, "fig_collision.png")


def reaction(parts):
    r = pd.read_csv(RES / "reaction.csv")
    parts.append("## Reaction to an unplanned obstacle (first second after contact)\n\n" + md(r))
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    for name in ["ignore", "stop", "float", "retract"]:
        d = pd.read_csv(RES / f"reaction_trace_{name}.csv")
        on = d[d.contact_force_n > 0.5]
        if on.empty:
            continue
        t0 = on.time_s.iloc[0]
        d = d[(d.time_s >= t0 - 0.05) & (d.time_s <= t0 + 1.0)]
        ax.plot(1000 * (d.time_s - t0), d.contact_force_n, lw=1.1, color=CC[name], label=name)
    ax.set_xlabel("time since first contact [ms]")
    ax.set_ylabel("contact force [N]")
    ax.set_title("Arm meets a forearm-sized obstacle mid-task")
    ax.legend(fontsize=8)
    save(fig, "fig_reaction.png")


def planning(parts):
    rr = pd.read_csv(RES / "planning_rrt.csv")
    parts.append("## RRT-Connect around the shelf (20 seeds)\n\n" + md(rr.describe().loc[["mean", "50%", "min", "max"]].reset_index()))
    tg = pd.read_csv(RES / "topp_grid.csv")
    parts.append("## TOPP-RA discretization (torque limit 0.80)\n\n" + md(tg))
    tp = pd.read_csv(RES / "planning_topp.csv")
    parts.append("## Time parameterization executed with computed torque\n\n" + md(tp))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    for v in ["topp_kinematic", "topp_torque", "trapezoid_uniform"]:
        d = pd.read_csv(RES / f"topp_profile_{v}.csv")
        axes[0].plot(d.time_s / d.time_s.max(), d.tau_ratio_max, color=CC[v], lw=1.1,
                     label=f"{LABEL[v]} ({d.time_s.max():.2f} s)")
    axes[0].axhline(1.0, color=MUTED, ls=":", lw=1)
    axes[0].axhline(0.8, color=MUTED, ls="--", lw=0.8)
    axes[0].set_xlabel("normalized time")
    axes[0].set_ylabel("max_j |tau_j| / tau_max,j (planned)")
    axes[0].set_title("Torque demand of the shelf move")
    axes[0].legend(fontsize=7)
    g = pd.read_csv(RES / "planning_path.csv")
    axes[1].add_patch(Rectangle((0.22, 0.988), 0.20, 0.024, color=MUTED, alpha=0.5, label="shelf board"))
    axes[1].add_patch(Rectangle((0.24, 0.58), 0.36, 0.04, color=SERIES[3], alpha=0.35, label="table top"))
    axes[1].plot(g.x, g.z, color=SERIES[0], label="tool path (smoothed RRT-Connect)")
    axes[1].scatter([g.x.iloc[0], g.x.iloc[-1]], [g.z.iloc[0], g.z.iloc[-1]], color=SERIES[0], s=16, zorder=3)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_aspect("equal", adjustable="datalim")
    axes[1].set_title("Under the shelf to above it")
    axes[1].legend(fontsize=7, loc="lower left")
    save(fig, "fig_planning.png")


def timing(parts):
    tt = pd.read_csv(RES / "tick_timing.csv")
    parts.append("## Control tick compute time (lockstep, wall clock) [us]\n\n" + md(tt))
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for tag, col in [("nominal", SERIES[0]), ("stress2", SERIES[1])]:
        f = RES / f"rt_summary_{tag}.csv"
        if not f.exists():
            continue
        rows.append(pd.read_csv(f))
        h = pd.read_csv(RES / f"rt_hist_{tag}.csv")
        h = h[h.histogram == "control_wake"].sort_values("bin_start_us")
        if h.empty:
            continue
        c = np.cumsum(h["count"].values)
        ax.step(h.bin_start_us + 2, 1 - c / c[-1] + 1e-7, where="post", color=col, label=tag)
    if rows:
        parts.append("## Threaded real-time runs\n\n" + md(pd.concat(rows)))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("control-thread wake-up lateness [us]")
    ax.set_ylabel("fraction of ticks later than x")
    ax.set_title("1 kHz loop on a shared, non-RT kernel")
    ax.legend(fontsize=8)
    save(fig, "fig_rt.png")
    b = RES / "bench.json"
    if b.exists():
        bj = json.loads(b.read_text())
        bt = pd.DataFrame([{"benchmark": x["name"], "time_us": x["real_time"] / 1000.0 if x["time_unit"] == "ns" else x["real_time"]}
                           for x in bj["benchmarks"]])
        parts.append("## Google Benchmark\n\n" + md(bt))


def main():
    parts = ["# Results (generated by scripts/plot.py)"]
    for fn in [tracking, payload, wipe, observer, collision, reaction, planning, timing]:
        try:
            fn(parts)
        except FileNotFoundError as e:
            print(f"skipping {fn.__name__}: {e}")
    (RES / "summary.md").write_text("\n\n".join(parts) + "\n")
    print((RES / "summary.md").read_text())


if __name__ == "__main__":
    main()
