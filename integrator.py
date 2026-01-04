import numpy as np

# mu = G(m1 + m2)

# define newtonian gravitational acceleration
def newtonian_grav_acc(r, mu):
    r_norm = np.linalg.norm(r)
    return (-mu * r) / (r_norm**3)

# define gravitational acceleration with relativistic perturbation
def relativistic_grav_acc(r, mu, alpha):
    r_norm = np.linalg.norm(r)
    a_newton = (-mu * r) / (r_norm**3)
    a_perturbation = alpha * (-mu * r) / (r_norm**4)
    return a_newton + a_perturbation

# define step-integration method to compute r(t), v(t)
def vel_verlet(r, v, dt, acc_func, acc_kwargs):
    a_0 = acc_func(r, **acc_kwargs)
    v_half = v + (0.5 * a_0 * dt)
    r_new = r + (v_half * dt)
    a_1 = acc_func(r_new, **acc_kwargs)
    v_new = v_half + (0.5 * a_1 * dt)
    return r_new, v_new