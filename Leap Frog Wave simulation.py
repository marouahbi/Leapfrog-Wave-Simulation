import numpy as np
import matplotlib.pyplot as plt

# 5. Wave equation
def leapfrog_wave_sim(length=1.0, nx=100, c=1.0, dt=0.01, steps=1000, pluck_pos=0.8, plot_times=[0, 100, 200, 500]):

    """
    It's a simulates of a wave on a string using the leapfrog algorithm.
    While also plots the results at different times, with different grids.
    To determine which graph is gives clearer representation.
    """

    h = length / (nx - 1)  # the spacing
    c_prime = c * dt / h  # this is the CFL ratio

    # CFL check, to make sure everything goes well
    if c_prime > 1:
        raise ValueError("CFL condition violated. Fix dt or nx.")

    x = np.linspace(0, length, nx)  # positions on the string

    # Initial conditions
    u = np.zeros(nx)
    u_new = np.zeros(nx)
    u_old = np.zeros(nx)
    pluck_idx = int(pluck_pos * (nx - 1))
    u[pluck_idx] = 1.0

    # First time step (simple start)
    for i in range(1, nx - 1):
        u_new[i] = u[i] + 0.5 * c_prime**2 * (u[i + 1] + u[i - 1] - 2 * u[i])

    # Leapfrog time-steps
    results = []
    for n in range(steps):
        for i in range(1, nx - 1):
            u_old[i] = 2 * u[i] - u_old[i] + c_prime**2 * (u[i + 1] + u[i - 1] - 2 * u[i])

        # Array is rotated
        u_old, u, u_new = u, u_old, u_new

        # Save specific time steps
        if n in plot_times:
            results.append((n, u.copy()))

    # Plotting
    plt.figure(figsize=(10, 6))
    for t, u_t in results:
        plt.plot(x, u_t, label=f"t = {t * dt:.2f} s")
    plt.title("Wave Simulation")
    plt.xlabel("Position")
    plt.ylabel("Displacement")
    plt.legend()
    plt.grid()
    plt.show()


# Simulation with two different grids to be compared
leapfrog_wave_sim(nx=100, dt=0.01, steps=500, plot_times=[0, 50, 100, 200])
leapfrog_wave_sim(nx=200, dt=0.005, steps=500, plot_times=[0, 50, 100, 200])
