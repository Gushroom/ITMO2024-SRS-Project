import numpy as np
import matplotlib.pyplot as plt

# Parameters
k = 10  # Steepness
theta_threshold = np.pi / 2  # Threshold (45 deg)
K_p_pos_base = 1.0  # Base value for K_p_pos, you can adjust this as needed

# Generate the range of error_theta
error_theta = np.linspace(-np.pi, np.pi, 1000)

# Compute K_p_pos
# K_p_pos = K_p_pos_base / (1 + np.exp(k * (np.abs(error_theta) - theta_threshold)))
K_p_pos = K_p_pos_base * np.maximum(0, np.cos(error_theta))

# Plot the function
plt.figure(figsize=(10, 6))
plt.plot(error_theta, K_p_pos)
plt.title(r"Logistic Function to Adjust K_p_pos Based on $\text{error}_\theta$")
plt.xlabel(r'$\text{error}_\theta$ (radians)')
plt.ylabel("K_pos adjustment factor")
plt.axvline(x=-theta_threshold, color='r', linestyle='--', label=r'$-\theta_{\text{threshold}}$')
plt.axvline(x=theta_threshold, color='r', linestyle='--', label=r'$\theta_{\text{threshold}}$')
plt.legend()
plt.grid(True)
plt.show()