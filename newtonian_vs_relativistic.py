import numpy as np
import matplotlib.pyplot as plt

from integrator import (
    vel_verlet,
    newtonian_grav_acc,
    relativistic_grav_acc
)
from orbital_elements import get_orbital_elements

def run_orbit(r0, v0, dt, t_array, acc_func, acc_kwargs):
    r = r0.copy()
    v = v0.copy()

    r_traj = []
    peri_list = []

    for i in t_array:
        r_traj.append(r.copy())

        elems = get_orbital_elements(r, v, acc_kwargs["mu"])
        e_vec = elems["eccentricity vector"]
        peri_angle = np.arctan2(e_vec[1], e_vec[0])
        peri_list.append(peri_angle)

        r, v = vel_verlet(r, v, dt, acc_func=acc_func, acc_kwargs=acc_kwargs)

    return np.array(r_traj), np.unwrap(np.array(peri_list))

mu = 1.0
alpha = 1e-3
dt = 1e-3
t_max = 40.0

r0 = np.array([1.0, 0.0, 0.0])
v0 = np.array([0.0, 0.8, 0.0])

t = np.arange(0.0, t_max, dt)

# newtonian simulation
r_newt, omega_newt = run_orbit(
    r0, v0, dt, t,
    acc_func = newtonian_grav_acc,
    acc_kwargs={"mu": mu}
)

# relativistic simulation
r_gr, omega_gr = run_orbit(
    r0, v0, dt, t,
    acc_func = relativistic_grav_acc,
    acc_kwargs={"mu": mu, "alpha": alpha}
)

# precession analysis
delta_omega = omega_gr - omega_newt
total_precession = delta_omega[-1] - delta_omega[0]
precession_rate = total_precession / t_max

delta_omega = omega_gr - omega_newt

# total precession with numerical noise
total_precession = delta_omega[-1] - delta_omega[0]
precession_rate = total_precession / t_max

# linear fit to determine secular drift
coeffs = np.polyfit(t, delta_omega, 1)
omega_dot_fit = coeffs[0]              
omega_0_fit   = coeffs[1]

# orbital period estimate
a = get_orbital_elements(r0, v0, mu)["semi major axis"]
T = 2 * np.pi * np.sqrt(a**3 / mu)

precession_per_orbit = omega_dot_fit * T

print(f"Relativistic strength = {alpha}")
print(f"Total simulation time = {t_max}")
print(f"(Linear Fit) Secular precession rate (rad / time) = {omega_dot_fit:.6f}")
print(f"(Linear Fit) Precession per orbit (rad / orbit) = {precession_per_orbit:.6f}")

# step to only plot every 10 points to reduce run-time
step = 10

# orbit comparison
plt.figure()
plt.plot(r_newt[::step, 0], r_newt[::step, 1], label="Newtonian")
plt.plot(r_gr[::step, 0], r_gr[::step, 1], "--", label="Relativistic")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Newtonian vs Relativistic Orbit")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("orbit_precession.png")

# argument of periapsis
plt.figure()
plt.plot(t, omega_newt, label="Newtonian")
plt.plot(t, omega_gr, "--", label="Relativistic")
plt.xlabel("Time")
plt.ylabel("Argument of Periapsis (rad)")
plt.title("Periapsis Evolution")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("periapsis_vs_time.png")

# differential precession
plt.figure()
plt.plot(t, delta_omega)

delta_omega_fit = omega_dot_fit * t + omega_0_fit
plt.plot(t, delta_omega_fit, label="Precession Rate = 1.24*10^-3")
plt.xlabel("Time")
plt.ylabel("Precession")
plt.title("Relativistic Periapsis Precession")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("precession_difference.png")

plt.show()