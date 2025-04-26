import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Schwarzschild radius and constants
M = 1.0  # Mass of the black hole in natural units
E = 1.0  # Energy per unit mass of the photon
L = 5.0  # Angular momentum per unit mass of the photon

# Differential equations for the geodesics
def geodesic_equations(lambda_, y):
    t, r, theta, phi, dt_dl, dr_dl, dtheta_dl, dphi_dl = y
    
    # Metric functions
    A = 1 - 2 * M / r
    B = r**2
    
    # Geodesic equations
    d2r_dl2 = r**2 * (dphi_dl**2 * np.sin(theta)**2 - dtheta_dl**2) - A * (1 - 2 * M / r) * (dr_dl**2 - dt_dl**2)
    d2theta_dl2 = (dphi_dl**2 - np.sin(theta) * np.cos(theta) * (dphi_dl**2))
    d2phi_dl2 = 0  # Assumes no explicit dependence on lambda for simplicity

    return [dt_dl, dr_dl, dtheta_dl, dphi_dl, d2r_dl2, d2theta_dl2, d2phi_dl2, 0]

# New initial conditions
r0 = 15.0  # Increased initial radial distance
theta0 = np.pi / 6  # Starting with a different polar angle
phi0 = 0.0  # Initial azimuthal angle

# Adjusted derivatives to modify direction
dt_dl0 = E / (1 - 2 * M / r0)
dr_dl0 = -np.sqrt(E**2 - (1 - 2 * M / r0) * (L**2 / r0**2))  # Negative sign to ensure inward motion
dtheta_dl0 = 0.1  # Small change in polar angle
dphi_dl0 = L / r0**2

# Initial state vector
y0 = [0, r0, theta0, phi0, dt_dl0, dr_dl0, dtheta_dl0, dphi_dl0]

# Integration range
lambda_span = [0, 50]  # Adjust the range as needed

# Solve the system
sol = solve_ivp(geodesic_equations, lambda_span, y0, t_eval=np.linspace(lambda_span[0], lambda_span[1], 1000))

# Extract the solution
r_sol = sol.y[1]
theta_sol = sol.y[2]
phi_sol = sol.y[3]

# Convert to Cartesian coordinates for 3D plotting
x_sol = r_sol * np.sin(theta_sol) * np.cos(phi_sol)
y_sol = r_sol * np.sin(theta_sol) * np.sin(phi_sol)
z_sol = r_sol * np.cos(theta_sol)

# 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_sol, y_sol, z_sol, label="Light Ray")
ax.scatter(0, 0, 0, color='k', s=100, label="Black Hole")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title("Geodesics of a Light Ray in 3D Around a Black Hole")
plt.show()