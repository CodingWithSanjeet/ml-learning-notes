import os
import math
import numpy as np
import matplotlib.pyplot as plt


def compute_cost(theta0: float, theta1: float, x: np.ndarray, y: np.ndarray) -> float:
	m = x.shape[0]
	h = theta0 + theta1 * x
	return (1.0 / (2.0 * m)) * np.sum((h - y) ** 2)


def gd_path_normalized_x(x: np.ndarray, y: np.ndarray, alpha: float, iters: int) -> list[tuple[float, float]]:
	"""Run GD on normalized feature; return ORIGINAL (theta0, theta1) each iteration."""
	mu = np.mean(x)
	sigma = np.std(x)
	xn = (x - mu) / sigma
	m = x.shape[0]
	t0p, t1p = 0.0, 0.0
	path: list[tuple[float, float]] = []
	for _ in range(iters):
		# record in original units
		theta1 = t1p / sigma
		theta0 = t0p - (t1p * mu / sigma)
		path.append((theta0, theta1))
		h = t0p + t1p * xn
		err = h - y
		grad0p = (1.0 / m) * np.sum(err)
		grad1p = (1.0 / m) * np.sum(err * xn)
		t0p -= alpha * grad0p
		t1p -= alpha * grad1p
	# final
	theta1 = t1p / sigma
	theta0 = t0p - (t1p * mu / sigma)
	path.append((theta0, theta1))
	return path


def build_contour(x: np.ndarray, y: np.ndarray, t0_range=(-1000, 2000), t1_range=(-0.5, 0.5), n=220):
	t0_vals = np.linspace(t0_range[0], t0_range[1], n)
	t1_vals = np.linspace(t1_range[0], t1_range[1], n)
	T0, T1 = np.meshgrid(t0_vals, t1_vals)
	J = np.zeros_like(T0)
	for i in range(T0.shape[0]):
		for j in range(T0.shape[1]):
			J[i, j] = compute_cost(T0[i, j], T1[i, j], x, y)
	imin, jmin = np.unravel_index(np.argmin(J), J.shape)
	levels = np.linspace(np.min(J), np.max(J) * 0.9, 25)
	min_level = J[imin, jmin] + (levels[1] - levels[0]) * 0.35
	return T0, T1, J, levels, (imin, jmin), min_level


def save_pair(left_title_line: str, right_title_line: str, x, y, theta0, theta1,
			T0, T1, J, levels, min_idx, min_level, out_path: str):
	fig, axes = plt.subplots(1, 2, figsize=(12, 5))
	# Left: h(x)
	axes[0].scatter(x, y, c="#d16b6b", marker="x", alpha=0.7, label="Training data")
	xline = np.linspace(x.min() * 0.9, x.max() * 1.05, 100)
	yline = theta0 + theta1 * xline
	axes[0].plot(xline, yline, color="#1f77b4", linewidth=2.5, label="Current hypothesis")
	axes[0].set_title(left_title_line)
	axes[0].set_xlabel(r"Size (feet$^2$)")
	axes[0].set_ylabel(r"Price $ (in \; 1000\text{s})$")
	axes[0].set_ylim(0, 700)
	axes[0].legend(loc="lower right")
	# Right: contour with min ring
	CS = axes[1].contour(T0, T1, J, levels=levels, colors=["#1f77b4"], linewidths=1.1)
	axes[1].clabel(CS, inline=True, fmt="%.1f", fontsize=7)
	axes[1].contour(T0, T1, J, levels=[min_level], colors=["#ff7f0e"], linewidths=2.0)
	imin, jmin = min_idx
	axes[1].scatter([T0[imin, jmin]], [T1[imin, jmin]], c="orange", marker="*", s=120)
	# Current point and J ring
	J_val = compute_cost(theta0, theta1, x, y)
	axes[1].contour(T0, T1, J, levels=[J_val], colors=["#2ca02c"], linestyles="--", linewidths=2.0)
	axes[1].scatter([theta0], [theta1], c="red", marker="x", s=90)
	# Place cost label slightly offset so it's on the ring near the point
	label_text = f"({theta0:.2f}, {theta1:.4f})  J={J_val:.1f}"
	axes[1].annotate(label_text, (theta0, theta1), textcoords="offset points", xytext=(10, -12), fontsize=9, color="#2ca02c",
				  bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#2ca02c", alpha=0.8))
	axes[1].set_title(right_title_line)
	axes[1].set_xlabel(r"$\theta_0$")
	axes[1].set_ylabel(r"$\theta_1$")
	fig.tight_layout()
	fig.savefig(out_path, dpi=160)
	plt.close(fig)


def pick_unique_milestones(path: list[tuple[float, float]], target: int = 8,
						  eps0_start: float = 1e-2, eps1_start: float = 1e-4) -> list[int]:
	"""Scan the path and collect indices where theta changes at least eps; adapt eps to get target points."""
	indices = [0]
	last_t0, last_t1 = path[0]
	eps0, eps1 = eps0_start, eps1_start
	for i in range(1, len(path)):
		t0, t1 = path[i]
		if abs(t0 - last_t0) > eps0 or abs(t1 - last_t1) > eps1:
			indices.append(i)
			last_t0, last_t1 = t0, t1
			if len(indices) >= target:
				break
	# If not enough, relax thresholds and rescan
	while len(indices) < target:
		indices = [0]
		last_t0, last_t1 = path[0]
		eps0 *= 0.5
		eps1 *= 0.5
		for i in range(1, len(path)):
			t0, t1 = path[i]
			if abs(t0 - last_t0) > eps0 or abs(t1 - last_t1) > eps1:
				indices.append(i)
				last_t0, last_t1 = t0, t1
				if len(indices) >= target:
					break
		# Safety: if thresholds are extremely small, stop
		if eps0 < 1e-6 and eps1 < 1e-8:
			break
	# Ensure last index (best) included and unique
	if indices[-1] != len(path) - 1:
		indices[-1] = len(path) - 1
	# If any accidental duplicates, make them strictly increasing
	uniq = []
	for idx in indices:
		if not uniq or idx > uniq[-1]:
			uniq.append(idx)
	return uniq[:target]


def main():
	# Output dir
	root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
	img_dir = os.path.join(root, "images", "lecture7")
	os.makedirs(img_dir, exist_ok=True)

	# Mock dataset
	rng = np.random.default_rng(7)
	x = rng.uniform(600, 4500, size=47)
	y = 50.0 + 0.12 * x + rng.normal(0.0, 60.0, size=x.shape[0])

	# Gradient descent path
	path = gd_path_normalized_x(x, y, alpha=0.2, iters=300)

	# Contour grid
	T0, T1, J, levels, min_idx, min_level = build_contour(x, y)

	# Unique milestones from worst to best
	milestones = pick_unique_milestones(path, target=8)

	for frame_idx, path_idx in enumerate(milestones):
		t0, t1 = path[path_idx]
		left_title = r"$h_\theta(x)$\n(for fixed $\theta_0,\theta_1$, this is a function of $x$)"
		right_title = r"$J(\theta_0,\theta_1)$\n(function of the parameters $\theta_0,\theta_1$)"
		out_path = os.path.join(img_dir, f"algorithm_step_{frame_idx:02d}.png")
		save_pair(left_title, right_title, x, y, t0, t1, T0, T1, J, levels, min_idx, min_level, out_path)
		if frame_idx == 1:
			dup = os.path.join(img_dir, "algorithm_step_01_latest.png")
			import shutil
			shutil.copyfile(out_path, dup)

	# Final summary
	final_t0, final_t1 = path[-1]
	save_pair("Best fit: h(x) after GD", "Final position on J(θ0,θ1)", x, y, final_t0, final_t1,
			  T0, T1, J, levels, min_idx, min_level, os.path.join(img_dir, "best_fit_pair.png"))


if __name__ == "__main__":
	main()
