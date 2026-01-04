import numpy as np

# magnitude of position vector
def r_norm(r):
    return np.linalg.norm(r)

# magnitude of velocity vector
def v_norm(v):
    return np.linalg.norm(v)

###############################################################################
"""
define functions to compute necessary orbital elements of the system
"""

# specific energy for newtonian gravity
def specific_energy(r, v, mu):
    return 0.5 * v_norm(v) ** 2 - mu / r_norm(r)

# specific energy for relativistic gravity
def relativistic_specific_energy(r, v, mu, alpha):
    return (0.5 * v_norm(v) ** 2) - (mu / r_norm(r)) - (0.5 * alpha * mu / r_norm(r) ** 2)

def spec_ang_momentum(r, v):
    return np.cross(r, v)

def h_norm(r, v):
    return np.linalg.norm(spec_ang_momentum(r, v))

def semi_major_axis(r, v, mu):
    epsilon = specific_energy(r, v, mu)
    if epsilon == 0:
        return np.inf
    return -mu / (2 * epsilon)

def eccentricity_vector(r, v, mu):
    h = spec_ang_momentum(r, v)
    return (np.cross(v, h) / mu) - (r / r_norm(r))

def eccentricity(r, v, mu):
    return np.linalg.norm(eccentricity_vector(r, v, mu))

def inclination(r, v):
    h = spec_ang_momentum(r, v)
    h_z = h[2]
    return np.arccos(h_z / h_norm(r, v))

def node_vector(r, v):
    h = spec_ang_momentum(r, v)
    k = np.array([0.0, 0.0, 1.0])
    return np.cross(k, h)

def ascending_node_long(r, v):
    n = node_vector(r, v)
    n_norm = np.linalg.norm(n)
    if n_norm == 0:
        return 0.0
    return np.arctan2(n[1], n[0])

def arg_of_periapsis(r, v, mu):
    n = node_vector(r, v)
    e = eccentricity_vector(r, v, mu)
    n_norm = np.linalg.norm(n)
    e_norm = eccentricity(r, v, mu)
    if n_norm == 0 or e_norm == 0:
        return 0.0
    return np.arccos(np.dot(n, e) / (n_norm * e_norm))

def true_anomaly(r, v, mu):
    e = eccentricity_vector(r, v, mu)
    e_norm = eccentricity(r, v, mu)
    if e_norm == 0:
        return 0.0
    cos_nu = np.dot(e, r) / (e_norm * r_norm(r))
    nu = np.arccos(cos_nu)
    if np.dot(r, v) < 0:
        nu = 2*np.pi - nu
    return nu

###############################################################################
"""
classify orbit type and boundedness of orbit
"""

def classify_orbit(r, v, mu, tol=1e-6):
    e = eccentricity(r, v, mu)   
    if e < tol:
        orbit_type = "circular"
    elif abs(e - 1.0) <= tol:
        orbit_type = "parabolic"
    elif e < 1.0:
        orbit_type = "elliptical"
    else:
        orbit_type = "hyperbolic"
        
    if e < 1.0 - tol:
        boundedness = "bounded"
    elif e > 1.0 + tol:
        boundedness = "unbounded"
    else:
        boundedness = "boundary case (parabolic)"
    
    return orbit_type, boundedness

###############################################################################
"""
define function which contains orbital elements
updates as position and velocity update
"""

def get_orbital_elements(r, v, mu, alpha=None):
    orbit_type, boundedness = classify_orbit(r, v, mu, tol=1e-6)
    elems= {
        "position norm": r_norm(r),
        "velocity norm": v_norm(v),
        "specific energy": specific_energy(r, v, mu),
        "specific angular momentum vector": spec_ang_momentum(r, v),
        "specific angular momentum norm": h_norm(r, v),
        "semi major axis": semi_major_axis(r, v, mu),
        "eccentricity vector": eccentricity_vector(r, v, mu),
        "eccentricity": eccentricity(r, v, mu),
        "inclination": inclination(r, v),
        "node vector": node_vector(r, v),
        "ascending node longitude": ascending_node_long(r, v),
        "argument of periapsis": arg_of_periapsis(r, v, mu),
        "true anomaly": true_anomaly(r, v, mu),
        "orbit type": orbit_type,
        "boundedness": boundedness
        }
    # define energy differently for relativistic gravity
    if alpha is not None:
        elems["specific energy"] = relativistic_specific_energy(r, v, mu, alpha)
        
    return elems
        
"""
if __name__ == "__main__":
    elements = get_orbital_elements(r_vec, v_vec, mu)
    for k, v in elements.items():
        print(f"{k}: {v}")
"""