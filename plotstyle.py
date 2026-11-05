"""Shared matplotlib style: validated colorblind-safe categorical order, thin marks,
recessive grid. Colors are assigned to controllers by name so a controller keeps its color
in every figure."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#1f1f1e"
MUTED = "#6b6a63"

CONTROLLER_COLOR = {
    "joint_pd_gravity": SERIES[7],
    "computed_torque": SERIES[0],
    "operational_space": SERIES[2],
    "cartesian_impedance": SERIES[3],
    "hybrid_pi": SERIES[2],
    "hybrid_pi_friction_ff": SERIES[0],
    "hybrid_feedforward": SERIES[1],
    "osc_position": SERIES[6],
    "ignore": SERIES[7],
    "stop": SERIES[3],
    "float": SERIES[0],
    "retract": SERIES[2],
    "topp_kinematic": SERIES[7],
    "topp_torque": SERIES[2],
    "trapezoid_uniform": SERIES[6],
}


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": "#e4e3dd",
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.6,
        "font.size": 10,
        "axes.titlesize": 11,
        "legend.frameon": False,
        "figure.dpi": 130,
    })
