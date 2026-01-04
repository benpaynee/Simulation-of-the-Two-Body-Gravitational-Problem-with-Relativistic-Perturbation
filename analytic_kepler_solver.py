import numpy as np
from orbital_elements import get_orbital_elements


def propagate_kepler(t_array, r0, v0, mu):
    """
    Analytic Kepler propagation of the relative coordinate r(t)
    for an elliptic orbit.

    Returns:
        r_array : (N, 3) ndarray of analytic positions
        elems   : orbital elements dictionary
    """

    elems = get_orbital_elements(r0, v0, mu)

    a = elems["semi major axis"]
    e = elems["eccentricity"]

    # construct orbital plain
    h_vec = np.cross(r0, v0)
    h_hat = h_vec / np.linalg.norm(h_vec)

    e_vec = elems["eccentricity vector"]
    p_hat = e_vec / np.linalg.norm(e_vec)          
    q_hat = np.cross(h_hat, p_hat)                  

    # mean motion
    n = np.sqrt(mu / a**3)

    # initial true anomaly
    r0_norm = np.linalg.norm(r0)
    cos_nu0 = np.dot(p_hat, r0) / r0_norm
    cos_nu0 = np.clip(cos_nu0, -1.0, 1.0)

    nu0 = np.arccos(cos_nu0)
    if np.dot(r0, v0) < 0:
        nu0 = 2*np.pi - nu0

    # initial eccentric and mean anomaly
    E0 = 2*np.arctan(
        np.tan(nu0 / 2) * np.sqrt((1 - e) / (1 + e))
    )
    M0 = E0 - e * np.sin(E0)

    # Newton solver for Kepler's equation
    def solve_kepler(M):
        E = M
        for i in range(50):
            f = E - e*np.sin(E) - M
            fp = 1 - e*np.cos(E)
            dE = f / fp
            E -= dE
            if abs(dE) < 1e-12:
                break
        return E

    # analytic propagation
    r_list = []

    for t in t_array:
        M = M0 + n*t
        E = solve_kepler(M)

        x = a * (np.cos(E) - e)
        y = a * np.sqrt(1 - e**2) * np.sin(E)

        r = x * p_hat + y * q_hat
        r_list.append(r)

    return np.array(r_list), elems

