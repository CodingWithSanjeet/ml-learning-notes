from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    base = Path(__file__).resolve().parent.parent.parent / "images" / "lecture6"
    t = np.linspace(-0.5, 2.5, 400)
    J = (t - 1.0) ** 2
    theta = 1.6  # point on the right of minimum
    J_theta = (theta - 1.0) ** 2
    slope = 2 * (theta - 1.0)
    # Tangent line at (theta, J_theta)
    x_tan = np.linspace(theta - 1.0, theta + 1.0, 100)
    y_tan = J_theta + slope * (x_tan - theta)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(t, J, label=r"$J(\theta_1)=(\theta_1-1)^2$")
    ax.scatter([theta], [J_theta], color="#d62728", zorder=5, label="current point")
    ax.plot(x_tan, y_tan, '--', color="#ff7f0e", label="tangent line")
    ax.annotate(r"slope = dJ/dθ₁", xy=(theta, J_theta), xytext=(theta - 0.9, J_theta + 1.2),
                arrowprops=dict(arrowstyle='->'),
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9))
    ax.scatter([1.0], [0.0], color="green", marker="*", s=120, zorder=5, label="minimum")
    ax.set_xlabel("θ₁")
    ax.set_ylabel("J(θ₁)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    base.mkdir(parents=True, exist_ok=True)
    fig.savefig(base / "derivative_tangent.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

