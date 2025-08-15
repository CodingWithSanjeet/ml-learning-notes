from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def toy_surface(theta0: np.ndarray, theta1: np.ndarray) -> np.ndarray:
    return 0.35 * (theta0 ** 2 + 1.6 * theta1 ** 2) + 0.6 * np.sin(2.6 * theta0) * np.cos(1.8 * theta1)


def numeric_grad(f, t0: float, t1: float, h: float = 1e-4) -> tuple[float, float]:
    d0 = (f(t0 + h, t1) - f(t0 - h, t1)) / (2 * h)
    d1 = (f(t0, t1 + h) - f(t0, t1 - h)) / (2 * h)
    return float(d0), float(d1)


def make_steepest_descent(out_path: Path) -> None:
    # Grid
    t0 = np.linspace(-0.2, 1.2, 160)
    t1 = np.linspace(-0.1, 1.1, 160)
    T0, T1 = np.meshgrid(t0, t1)
    J = toy_surface(T0, T1)

    # A short GD path
    f = lambda a, b: toy_surface(np.array([a]), np.array([b]))[0]
    alpha = 0.25
    steps = 10
    p0, p1 = 0.9, 0.8
    path0, path1 = [p0], [p1]
    for _ in range(steps):
        g0, g1 = numeric_grad(f, p0, p1)
        p0 -= alpha * g0
        p1 -= alpha * g1
        path0.append(p0)
        path1.append(p1)
    pathJ = toy_surface(np.array(path0), np.array(path1))

    fig = plt.figure(figsize=(9.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(T0, T1, J, cmap="YlOrBr", edgecolor="none", alpha=0.95)
    ax.plot(path0, path1, pathJ, color="black", marker="o", markersize=3)
    # Arrow indicating steepest descent near the start
    ax.quiver(path0[0], path1[0], pathJ[0],
              path0[1]-path0[0], path1[1]-path1[0], pathJ[1]-pathJ[0],
              color="black", length=1.0, normalize=True)
    ax.text(path0[1], path1[1], pathJ[1] + 0.4, "Steepest\ndescent", color="black")

    ax.set_xlabel(r"$\theta_0$")
    ax.set_ylabel(r"$\theta_1$")
    ax.set_zlabel(r"$J(\theta_0,\theta_1)$")
    ax.view_init(elev=25, azim=-60)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parent / "images" / "lecture5"
    make_steepest_descent(base / "gd_steepest_descent.png")
    # Also make a simple 1D gradient illustration J(θ1) = (θ1-1)^2
    t = np.linspace(-0.5, 2.5, 400)
    J = (t - 1.0) ** 2
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(t, J, label=r"$J(\theta_1)=(\theta_1-1)^2$")
    # Add extra headroom so annotations don't overlap the frame
    ax.set_ylim(0, J.max() + 1.2)
    ax.margins(x=0.05, y=0.2)
    # Minimum marker and label at θ1=1
    ax.scatter([1.0], [0.0], color="green", marker="*", s=160, zorder=6, label="minimum cost")
    ax.axvline(1.0, color="green", linestyle="--", alpha=0.6)
    ax.annotate("minimum J at θ₁ = 1",
                xy=(1.0, 0.0), xytext=(1.6, 0.9),
                arrowprops=dict(arrowstyle="->", color="green"),
                color="green",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    # Example points and move directions
    example_thetas = [0.5, 1.5]
    colors = ["#d62728", "#1f77b4"]
    for idx, th in enumerate(example_thetas):
        Jth = (th - 1.0) ** 2
        grad = 2 * (th - 1.0)
        ax.scatter([th], [Jth], color=colors[idx], zorder=5)
        if th < 1.0:
            label_xy = (th + 0.12, Jth + 0.35)
            grad_xy = (th - 0.55, Jth + 0.85)
            dx = 0.55
            move_label = "move right →"
        else:
            label_xy = (th - 0.65, Jth + 0.35)
            grad_xy = (th - 0.95, Jth + 0.85)
            dx = -0.55
            move_label = "← move left"
        ax.annotate(f"θ₁={th}", xy=(th, Jth), xytext=label_xy,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        # Direction: opposite the gradient
        # Draw arrow indicating direction to move
        start_x, start_y = th + dx, Jth + 0.08
        end_x, end_y = th, Jth + 0.08
        ax.annotate("",
                    xy=(end_x, end_y),
                    xytext=(start_x, start_y),
                    arrowprops=dict(arrowstyle="->"))
        # Put the label centered slightly above the arrow
        mid_x = (start_x + end_x) / 2.0
        mid_y = (start_y + end_y) / 2.0
        ax.text(mid_x, mid_y + 0.12, move_label,
                ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        # Also annotate gradient sign
        ax.annotate(f"gradient = {grad:+.2f}", xy=(th, Jth), xytext=grad_xy,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$J(\theta_1)$")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    base.mkdir(parents=True, exist_ok=True)
    fig.savefig(base / "gradient_1d_example.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()

