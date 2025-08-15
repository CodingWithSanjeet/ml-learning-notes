from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_gd_on_quadratic(theta_start: float, alpha: float, steps: int) -> tuple[np.ndarray, np.ndarray]:
    thetas = [theta_start]
    for _ in range(steps):
        grad = 2 * (thetas[-1] - 1.0)  # d/dθ (θ-1)^2 = 2(θ-1)
        thetas.append(thetas[-1] - alpha * grad)
    thetas = np.array(thetas)
    return thetas, (thetas - 1.0) ** 2


def make_single_panel(out_path: Path, alpha: float, title: str, color: str) -> None:
    t = np.linspace(-1.5, 2.5, 400)
    J_curve = (t - 1.0) ** 2
    theta0 = -0.8
    steps = 16
    thetas, Js = run_gd_on_quadratic(theta0, alpha, steps)

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(t, J_curve, color="#1f77b4", label="J(θ₁) = (θ₁−1)²")
    ax.plot(thetas, Js, marker="o", color=color, label=f"α = {alpha}")
    ax.scatter([1.0], [0.0], color="green", marker="*", s=120, zorder=5, label="minimum")
    ax.set_title(title)
    ax.set_xlabel("θ₁")
    ax.set_ylabel("J(θ₁)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parent.parent.parent / "images" / "lecture6"
    make_single_panel(base / "gd_alpha_small.png", alpha=0.05, title="Small α → tiny steps (slow)", color="#2ca02c")
    make_single_panel(base / "gd_alpha_large.png", alpha=0.9, title="Large α → overshoot/oscillate", color="#d62728")
    # Local minimum: derivative = 0, update = 0
    t = np.linspace(-1.5, 2.5, 400)
    J_curve = (t - 1.0) ** 2
    theta_star = 1.0
    J_star = 0.0
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(t, J_curve, color="#1f77b4", label="J(θ₁) = (θ₁−1)²")
    # Horizontal tangent at the minimum
    x_tan = np.linspace(theta_star - 1.0, theta_star + 1.0, 50)
    y_tan = np.zeros_like(x_tan)
    ax.plot(x_tan, y_tan, linestyle="--", color="#ff7f0e", label="tangent (slope = 0)")
    ax.scatter([theta_star], [J_star], color="green", marker="*", s=140, zorder=5, label="local minimum")
    ax.annotate("dJ/dθ₁ = 0\nupdate = −α·0 = 0\nθ stays the same",
                xy=(theta_star, J_star), xytext=(theta_star + 0.3, J_star + 1.0),
                arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    ax.set_title("At local minimum: derivative = 0 (no change)")
    ax.set_xlabel("θ₁")
    ax.set_ylabel("J(θ₁)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(base / "gd_local_minimum_zero_derivative.png", dpi=170)
    plt.close(fig)

    # Automatic step shrinking near minima (fixed α, steps get shorter as |gradient| shrinks)
    t = np.linspace(-1.8, 2.2, 400)
    J_curve = (t - 1.0) ** 2
    theta = -1.2
    alpha = 0.25
    steps = 10
    thetas = [theta]
    for _ in range(steps):
        grad = 2 * (thetas[-1] - 1.0)
        thetas.append(thetas[-1] - alpha * grad)
    thetas = np.array(thetas)
    Js = (thetas - 1.0) ** 2

    fig2, ax2 = plt.subplots(figsize=(7.5, 4.4))
    ax2.plot(t, J_curve, color="#1f77b4", label="J(θ₁) = (θ₁−1)²")
    ax2.scatter([1.0], [0.0], color="green", marker="*", s=120, zorder=5, label="minimum")
    # Draw arrows between successive points; later arrows are shorter
    for i in range(len(thetas) - 1):
        x0, y0 = thetas[i], Js[i]
        x1, y1 = thetas[i + 1], Js[i + 1]
        ax2.annotate("",
                     xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle="->", lw=1.5, alpha=0.8))
    ax2.plot(thetas, Js, "o-", color="#2ca02c", label="GD steps (α fixed)")
    ax2.set_title("Automatic step shrinking near the minimum (|gradient| ↓)")
    ax2.set_xlabel("θ₁")
    ax2.set_ylabel("J(θ₁)")
    ax2.legend(loc="upper left")
    fig2.tight_layout()
    fig2.savefig(base / "gd_step_shrinking.png", dpi=170)
    plt.close(fig2)


if __name__ == "__main__":
    main()

