import numpy as np
import matplotlib.pyplot as plt

# Iterations for gradient descent
iterations = np.arange(1, 101)

# Simulated cost function J(theta) curves for different learning rates alpha
def simulate_J(alpha, iterations):
    """Simulate J(theta) curve for given learning rate alpha."""
    noise = 0.02 * np.random.randn(len(iterations))
    decay_rate = alpha * 5  # scale alpha for decay speed
    base_curve = 10 * np.exp(-decay_rate * iterations)
    if alpha > 0.05:  # Large alpha causes oscillations and divergence
        oscillation = 3.0 * np.sin(5 * iterations)
        return base_curve + oscillation + noise + iterations * 0.05
    else:  # Small or moderate alpha: exponential decay plus noise
        return base_curve + noise

alphas = [0.001, 0.003, 0.01, 0.03, 0.1]
colors = ['gray', 'blue', 'green', 'orange', 'red']  # green for best alpha

plt.figure(figsize=(12, 7))

for alpha, color in zip(alphas, colors):
    J_curve = simulate_J(alpha, iterations)
    label = f'α = {alpha}'
    plt.plot(iterations, J_curve, label=label, color=color, linewidth=2 if color=="green" else 1)

# Highlight best alpha (assuming 0.01 is best here)
best_alpha = 0.01
best_index = alphas.index(best_alpha)

# Annotate best alpha on plot
plt.annotate('Best α',
             xy=(iterations[-1], simulate_J(best_alpha, iterations)[-1]),
             xytext=(70, 8),
             arrowprops=dict(facecolor='green', shrink=0.05),
             fontsize=14,
             color='green')

plt.title('Choosing Learning Rate α: Sweep of J(θ) over Iterations')
plt.xlabel('Iteration Number')
plt.ylabel(r'Cost Function $J(\theta)$')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('learning_rate_sweep_best_alpha.png', dpi=300)
plt.show()
