import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from integrator import (
    vel_verlet,
    newtonian_grav_acc,
    relativistic_grav_acc
)
from orbital_elements import get_orbital_elements

# load JSON with error handling
if len(sys.argv) < 2:
    print("Usage: python main.py input.json")
    sys.exit(1)

try:
    with open(sys.argv[1], "r") as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: JSON file not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print("Error: Invalid JSON format.")
    sys.exit(1)

# validate JSON structure
required_structure = {
    "masses": ["m1", "m2"],
    "radii": ["R1", "R2"],
    "initial_conditions": ["r_rel", "v_rel"],
    "gravity": ["use_relativistic", "alpha"],
    "time": ["dt", "t_max"]
}

for section, keys in required_structure.items():
    if section not in config:
        print(f"Error: Missing '{section}' section in JSON.")
        sys.exit(1)
    for key in keys:
        if key not in config[section]:
            print(f"Error: Missing '{key}' in '{section}' section.")
            sys.exit(1)

try:
    m1 = float(config["masses"]["m1"])
    m2 = float(config["masses"]["m2"])
    R1 = float(config["radii"]["R1"])
    R2 = float(config["radii"]["R2"])

    r_rel = np.array(config["initial_conditions"]["r_rel"], dtype=float)
    v_rel = np.array(config["initial_conditions"]["v_rel"], dtype=float)

    use_rel = bool(config["gravity"]["use_relativistic"])
    alpha = float(config["gravity"]["alpha"])

    dt = float(config["time"]["dt"])
    t_max = float(config["time"]["t_max"])
except (TypeError, ValueError):
    print("Error: One or more parameters have invalid types.")
    sys.exit(1)

# check physical validity of JSON inputs
if m1 <= 0 or m2 <= 0:
    print("Error: Masses must be positive.")
    sys.exit(1)

if r_rel.shape != (3,) or v_rel.shape != (3,):
    print("Error: r_rel and v_rel must be 3-element vectors.")
    sys.exit(1)

if dt <= 0 or t_max <= 0:
    print("Error: dt and t_max must be positive.")
    sys.exit(1)
    
if dt > t_max:
    print("Error: dt cannot be larger than t_max (dt > t_max).")
    sys.exit(1)

# user-facing confirmation
print("Valid Input. Starting Simulation Now.")

# derive necessary quantities
M = m1 + m2
collision_rad = R1 + R2
mu = 1.0           # using G = 1
steps = int(t_max / dt)

# choose gravity model based on JSON
if use_rel:
    acc_func = relativistic_grav_acc
    acc_kwargs = {"mu": mu, "alpha": alpha}
    title = "Two-Body Orbit (Relativistic)"
else:
    acc_func = newtonian_grav_acc
    acc_kwargs = {"mu": mu}
    title = "Two-Body Orbit (Newtonian)"

r1_traj, r2_traj = [], []
energy_list, h_list, ecc_list = [], [], []
collision_index = None

# relative coordinates
r = r_rel.copy()
v = v_rel.copy()

# initial orbital elements for reference
init_elems = get_orbital_elements(r, v, mu)
eps0 = init_elems["specific energy"]
h0   = init_elems["specific angular momentum norm"]
e0   = init_elems["eccentricity"]

for i in range(steps):
    # positions of bodies in COM frame
    r1 =  (m2 / M) * r
    r2 = -(m1 / M) * r
    r1_traj.append(r1)
    r2_traj.append(r2)

    elems = get_orbital_elements(r, v, mu)
    energy_list.append(elems["specific energy"])
    h_list.append(elems["specific angular momentum norm"])
    ecc_list.append(elems["eccentricity"])

    # collision detection in relative coordinates
    if np.linalg.norm(r) <= collision_rad:
        collision_index = i
        print(f"\nCollision detected at t = {i*dt:.3f}\n")
        break

    # advance one step
    r, v = vel_verlet(r, v, dt, acc_func=acc_func, acc_kwargs=acc_kwargs)

r1_traj = np.array(r1_traj)
r2_traj = np.array(r2_traj)

energy = np.array(energy_list)
h_mag  = np.array(h_list)
ecc    = np.array(ecc_list)

t = np.arange(len(energy)) * dt

# orbital classification
if abs(e0 - 1.0) < 1e-6:
    orbit_type = "Marginally bound (parabolic)"
elif e0 < 1.0:
    orbit_type = "Bound (elliptic)"
else:
    orbit_type = "Unbounded (hyperbolic)"

print("Orbit Type:", orbit_type)

# threshold for "zero" energy
ENERGY_TOL = 1e-6

if abs(eps0) > ENERGY_TOL:
    # normal bound / unbound cases
    energy_err = (energy - eps0) / abs(eps0)
else:
    # parabolic (marginally bound) case
    energy_err = energy - eps0

h_err = (h_mag - h0) / h0

plt.figure()
plt.plot(t, energy_err)
plt.xlabel("Time")
plt.ylabel("Relative Energy Error")
plt.title("Relative Specific Energy Error")
plt.grid()
plt.savefig("energy_error.png")

plt.figure()
plt.plot(t, h_err)
plt.xlabel("Time")
plt.ylabel("Relative Angular Momentum Error")
plt.title("Relative Specific Angular Momentum Error")
plt.grid()
plt.savefig("angular_momentum_error.png")

plt.figure()
plt.plot(t, ecc)
plt.xlabel("Time")
plt.ylabel("Eccentricity")
plt.title("Eccentricity vs Time")
plt.grid()
plt.savefig("eccentricity.png")

# 3D plot animation
all_x = np.concatenate((r1_traj[:, 0], r2_traj[:, 0]))
all_y = np.concatenate((r1_traj[:, 1], r2_traj[:, 1]))
all_z = np.concatenate((r1_traj[:, 2], r2_traj[:, 2]))

span = max(
    all_x.max() - all_x.min(),
    all_y.max() - all_y.min(),
    all_z.max() - all_z.min()
)
margin = 0.2 * span if span > 0 else 1.0

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_title(title)

ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
ax.set_zlim(all_z.min() - margin, all_z.max() + margin)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# viewing angle
ax.view_init(elev=20, azim=45)

body1, = ax.plot([], [], [], "o", ms=8, label="Body 1")  
body2, = ax.plot([], [], [], "o", ms=6, label="Body 2")  
trail1, = ax.plot([], [], [], "-", lw=1)                 
trail2, = ax.plot([], [], [], "-", lw=1)                 
ax.legend()

def update(frame):
    # points
    body1.set_data([r1_traj[frame, 0]], [r1_traj[frame, 1]])
    body1.set_3d_properties([r1_traj[frame, 2]])

    body2.set_data([r2_traj[frame, 0]], [r2_traj[frame, 1]])
    body2.set_3d_properties([r2_traj[frame, 2]])

    # trails
    trail1.set_data(r1_traj[:frame, 0], r1_traj[:frame, 1])
    trail1.set_3d_properties(r1_traj[:frame, 2])

    trail2.set_data(r2_traj[:frame, 0], r2_traj[:frame, 1])
    trail2.set_3d_properties(r2_traj[:frame, 2])

    return body1, body2, trail1, trail2

# reduce points plotted in animation to reduce runtime
ani_stride = 10
frames = range(0, len(r1_traj), ani_stride)

ani = FuncAnimation(fig, update, frames=frames, interval=20)
plt.show()