import numpy as np
import matplotlib.pyplot as plt
import os

def plot_contours(ax, center, width, height, angle, levels=30, label=None):
    from matplotlib.patches import Ellipse
    # Create an ellipse to represent contour
    ellipse = Ellipse(xy=center, width=width*2, height=height*2,
                      angle=angle, edgecolor='black', fc='None', lw=2, label=label)
    ax.add_patch(ellipse)
    # For contour lines, plot smaller nested ellipses
    for scale in np.linspace(0.2, 1, levels):
        ell = Ellipse(xy=center, width=width*2*scale, height=height*2*scale,
                      angle=angle, edgecolor='gray', fc='None', lw=0.7, alpha=0.3)
        ax.add_patch(ell)

def plot_gradient_path(ax, path_x, path_y, style, label):
    ax.plot(path_x, path_y, style, marker='o', label=label)

# Setup figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Unscaled feature cost contours (ellipse, theta1 large scale)
axs[0].set_title('Unscaled Features')
axs[0].set_xlabel(r'$\theta_1$ (House Size)')
axs[0].set_ylabel(r'$\theta_2$ (Bedrooms)')
axs[0].set_xlim(-1, 3)
axs[0].set_ylim(-1, 3)

# Plot unscaled ellipse contours
plot_contours(axs[0], center=(1, 1), width=1.5, height=0.3, angle=0)

# Plot zigzag gradient descent path for unscaled
path_x_unscaled = [2.9, 0.4, 2.4, 0.8, 1.5, 1.0]
path_y_unscaled = [0.2, 0.8, 1.3, 1.7, 1.5, 1.0]
plot_gradient_path(axs[0], path_x_unscaled, path_y_unscaled, 'r--', 'Gradient Descent Path')

# Scaled feature cost contours (circle, both features scaled 0 to 1)
axs[1].set_title('Scaled Features')
axs[1].set_xlabel(r'$\theta_1$ (Scaled House Size)')
axs[1].set_ylabel(r'$\theta_2$ (Scaled Bedrooms)')
axs[1].set_xlim(-0.5, 1.5)
axs[1].set_ylim(-0.5, 1.5)

# Plot circular contours for scaled features
plot_contours(axs[1], center=(0.5, 0.5), width=0.6, height=0.6, angle=0)

# Plot straight gradient descent path for scaled
path_x_scaled = [1.3, 0.9, 0.7, 0.6, 0.5]
path_y_scaled = [1.3, 0.9, 0.7, 0.6, 0.5]
plot_gradient_path(axs[1], path_x_scaled, path_y_scaled, 'g-', 'Gradient Descent Path')

# Common settings
for ax in axs:
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

plt.suptitle('Comparison of Cost Function Contours and Gradient Descent Paths\nUnscaled vs. Scaled Features', fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Ensure output directory
out_dir = os.path.join('Module 4', 'images')
os.makedirs(out_dir, exist_ok=True)

# Save combined figure
plt.savefig(os.path.join(out_dir, 'feature_scaling_comparison.png'), dpi=300)

# Also save individual panels
for idx, name in enumerate(['unscaled_contours.png', 'scaled_contours.png']):
    fig_single, ax_single = plt.subplots(1, 1, figsize=(6, 6))
    # Copy artists from corresponding axis
    for artist in axs[idx].get_children():
        try:
            artist.remove()
            ax_single.add_artist(artist)
        except Exception:
            pass
    ax_single.set_xlim(axs[idx].get_xlim())
    ax_single.set_ylim(axs[idx].get_ylim())
    ax_single.set_aspect('equal')
    ax_single.set_title(axs[idx].get_title())
    ax_single.set_xlabel(axs[idx].get_xlabel())
    ax_single.set_ylabel(axs[idx].get_ylabel())
    fig_single.tight_layout()
    fig_single.savefig(os.path.join(out_dir, name), dpi=300)
    plt.close(fig_single)

plt.show()
