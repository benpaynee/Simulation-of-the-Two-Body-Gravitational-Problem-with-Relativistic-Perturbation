import numpy as np
import matplotlib.pyplot as plt

from integrator import vel_verlet, newtonian_grav_acc
from analytic_kepler_solver import propagate_kepler

# masses
m1 = 2.0
m2 = 1.0
M = m1 + m2

mu = 1.0            
dt = 1e-3
t_max = 40.0

acc_func = newtonian_grav_acc
acc_kwargs = {"mu": mu}

# initial relative conditions (COM frame)
r0 = np.array([1.0, 0.0, 0.0])
v0 = np.array([0.0, 0.8, 0.0])

t = np.arange(0.0, t_max, dt)
N = len(t)

# numerical integration
r_rel_num = np.zeros((N, 3))
v_rel_num = np.zeros((N, 3))

r = r0.copy()
v = v0.copy()

for i in range(N):
    r_rel_num[i] = r
    v_rel_num[i] = v
    r, v = vel_verlet(r, v, dt, acc_func=acc_func, acc_kwargs=acc_kwargs)

# analytic kepler solution
r_rel_ana, elems = propagate_kepler(t, r0, v0, mu)

# convert relative motion to individual motion
def two_body_positions(r_rel, m1, m2):
    M = m1 + m2
    r1 = (m2 / M) * r_rel
    r2 = -(m1 / M) * r_rel
    return r1, r2

r1_num, r2_num = two_body_positions(r_rel_num, m1, m2)
r1_ana, r2_ana = two_body_positions(r_rel_ana, m1, m2)

# error analysis (body 1)
pos_err = np.linalg.norm(r1_num - r1_ana, axis=1)
r_mag = np.linalg.norm(r1_ana, axis=1)
rel_pos_err = pos_err / r_mag

# step so it only plots every 10 points to save run-time
step = 10

# trajectory comparison (body 1)
plt.figure()
plt.plot(r1_num[::step, 0], r1_num[::step, 1], label="Body 1 Numeric")
plt.plot(r1_ana[::step, 0], r1_ana[::step, 1], "--", label="Body 1 Analytic")
plt.plot([0], [0], "k+", label="Center of Mass")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Body 1: Numerical vs Analytic Orbit (COM Frame)")
plt.axis("equal")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("body1_orbit_comparison.png")


# absolute position error vs time
plt.figure()
plt.plot(t, pos_err)
plt.xlabel("Time")
plt.ylabel(r"Absolute Position Error")
plt.title("Body 1 Absolute Position Error vs Time")
plt.yscale("log")
plt.grid()
plt.tight_layout()
plt.savefig("body1_position_error.png")

plt.show()