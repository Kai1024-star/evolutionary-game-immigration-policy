"""English implementations of the two distinct specifications in the source folder.

The vector fields and Jacobians describe the continuous mathematical models.
``simulate`` reproduces the historical clipped RK4/Euler schemes. The adaptive
SciPy solver is an independent check added for this English release.
"""

from itertools import product
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit, logit


GOVERNMENT = dict(E1=2.0, E2=1.5, C1=0.6, C2=0.2,
                  E3=2.0, E4=1.8, C3=0.9, C4=0.5,
                  E5=2.0, E6=1.9, C5=0.6, S1=0.8, S2=0.6,
                  R1=1.2, R2=0.8)
FIRM = dict(E1=2.60, E2=1.70, C1=0.35, C2=0.20,
            E3=1.20, E4=1.15, C3=1.05, C4=0.60,
            E5=1.85, E6=1.35, C5=0.80, S1=0.10, S2=0.22,
            eta=0.25, rx=1.0, ry=1.0, rz=1.0,
            mu_x=0.0005, mu_y=0.0010, mu_z=0.0005)


def parameters(model="government", **overrides):
    """Return a copy of the original baseline; reject misspelled parameter names."""
    if model not in ("government", "firm"):
        raise ValueError("model must be government or firm")
    p = dict(GOVERNMENT if model == "government" else FIRM)
    if set(overrides) - set(p):
        raise ValueError(f"Unknown parameters: {set(overrides) - set(p)}")
    p.update(overrides)
    return p


def payoffs(state, p, model="government"):
    """Return [[K1,K2], [M1,M2], [N1,N2]] expected strategy payoffs.

    x = restrictive-policy share, y = foreign-hiring share,
    z = share of talent choosing Japan. x=0 denotes expansion, not contraction.
    """
    x, y, z = np.asarray(state, dtype=float)
    g1 = p["E2"] - p["C2"]
    f2 = p["E4"] - p["C4"]
    t1 = p["E5"] - p["C5"] + (1 - x) * p["S2"]
    if model == "government":
        g2 = (p["E1"] - p["C1"] + (p["R1"] - p["S1"]) * y * z
              + (p["R2"] - p["S2"]) * z)
        f1 = -p["C3"] + z * p["E3"] + (1 - x) * z * p["S1"]
    elif model == "firm":
        g2 = p["E1"] - p["C1"] - y * p["S1"] - z * p["S2"]
        cost = max(p["C3"] - p["eta"] * z, 0.0)
        f1 = 0.15 * p["E4"] + z * p["E3"] + (1 - x) * p["S1"] - cost
    else:
        raise ValueError("Unknown model")
    return np.array([[g1, g2], [f1, f2], [t1, p["E6"]]])


def rhs(state, p, model="government"):
    q = np.asarray(state, dtype=float)
    values = payoffs(q, p, model)
    gaps = values[:, 0] - values[:, 1]
    if model == "government":
        return q * (1 - q) * gaps
    rates = np.array([p["rx"], p["ry"], p["rz"]])
    mutation = np.array([p["mu_x"], p["mu_y"], p["mu_z"]])
    return rates * q * (1 - q) * gaps + mutation * (0.5 - q)


def jacobian(state, p, model="government"):
    """Analytic continuous-time Jacobian (not the Jacobian of a clipped step).

    The firm-cost maximum is nondifferentiable at C3=eta*z; that point requires
    one-sided analysis instead of the smooth eigenvalue test used here.
    """
    q = np.asarray(state, dtype=float)
    x, y, z = q
    values = payoffs(q, p, model)
    gaps = values[:, 0] - values[:, 1]
    if model == "government":
        dg = np.array([[0, (p["S1"] - p["R1"]) * z,
                        (p["S1"] - p["R1"]) * y + p["S2"] - p["R2"]],
                       [-p["S1"] * z, 0, p["E3"] + (1 - x) * p["S1"]],
                       [-p["S2"], 0, 0]])
        rates, mutation = np.ones(3), np.zeros(3)
    elif model == "firm":
        cost_gap = p["C3"] - p["eta"] * z
        if p["eta"] != 0 and abs(cost_gap) < 1e-12:
            raise ValueError("Cost kink: use one-sided derivatives")
        cost_slope = p["eta"] if cost_gap > 0 else 0.0
        dg = np.array([[0, p["S1"], p["S2"]],
                       [-p["S1"], 0, p["E3"] + cost_slope],
                       [-p["S2"], 0, 0]])
        rates = np.array([p["rx"], p["ry"], p["rz"]])
        mutation = np.array([p["mu_x"], p["mu_y"], p["mu_z"]])
    else:
        raise ValueError("Unknown model")
    return (np.diag(rates * (1 - 2 * q) * gaps - mutation)
            + (rates * q * (1 - q))[:, None] * dg)


def time_grid(T, dt):
    if not np.isfinite(T) or not np.isfinite(dt) or T <= 0 or dt <= 0:
        raise ValueError("T and dt must be positive and finite")
    steps = round(T / dt)
    if not np.isclose(steps * dt, T, rtol=1e-12, atol=1e-12):
        raise ValueError("T must be an integer multiple of dt")
    return np.linspace(0.0, T, steps + 1)


def simulate(initial, p, model="government", T=200.0, dt=0.05, eps=1e-9):
    """Historical schemes: clipped classical RK4 (government), Euler (firm).

    Clipping is a numerical safeguard in the supplied scripts, not mutation.
    The unused extra update after the final stored RK4 sample is omitted.
    """
    t = time_grid(T, dt)
    q = np.asarray(initial, dtype=float).copy()
    if q.shape != (3,) or not np.all(np.isfinite(q)) or np.any((q < 0) | (q > 1)):
        raise ValueError("initial must contain three finite shares in [0,1]")
    states = np.empty((len(t), 3))
    states[0] = q
    for i in range(1, len(t)):
        if model == "government":
            def f(v):
                return rhs(np.clip(v, eps, 1 - eps), p, model)
            a = f(q)
            b = f(q + 0.5 * dt * a)
            c = f(q + 0.5 * dt * b)
            d = f(q + dt * c)
            q = q + (dt / 6) * (a + 2*b + 2*c + d)
        elif model == "firm":
            q = q + dt * rhs(q, p, model)
        else:
            raise ValueError("Unknown model")
        q = np.clip(q, eps, 1 - eps)
        states[i] = q
    return t, states


def reference_solution(initial, p, model="government", T=200., dt=.05,
                       rtol=1e-10, atol=1e-12):
    """Adaptive DOP853 cross-check, added in the English repository release.

    The mutation-free model is integrated in logit coordinates. Otherwise tiny
    positive shares can fall below an absolute tolerance, become negative and
    later blow up when their selection advantage changes sign. This method does
    not insert the historical 1e-9 population floor.
    """
    t = time_grid(T, dt)
    if model == "government":
        q0 = np.asarray(initial, dtype=float)
        if np.any((q0 <= 0) | (q0 >= 1)):
            raise ValueError("Logit reference requires strictly interior initial shares")
        def fun(_, u):
            values = payoffs(expit(u), p, model)
            return values[:, 0] - values[:, 1]
        initial_value = logit(q0)
    else:
        fun = lambda _, q: rhs(q, p, model)
        initial_value = initial
    result = solve_ivp(fun, (0, T), initial_value,
                       method="DOP853", t_eval=t, rtol=rtol, atol=atol)
    if not result.success:
        raise RuntimeError(result.message)
    states = expit(result.y.T) if model == "government" else result.y.T
    return t, states


def tail_statistics(states, fraction=.30):
    tail = states[int((1 - fraction) * len(states)):]
    return {f"{name}_tail_{stat}": float(fn(tail[:, j]))
            for j, name in enumerate("xyz")
            for stat, fn in [("min", np.min), ("max", np.max), ("mean", np.mean)]}


def corner_equilibria(p):
    """Eight pure-strategy equilibria of the mutation-free government model."""
    rows = []
    for corner in product((0., 1.), repeat=3):
        eig = np.linalg.eigvals(jacobian(corner, p))
        label = ("locally asymptotically stable" if max(eig.real) < -1e-10 else
                 "unstable" if max(eig.real) > 1e-10 else "nonhyperbolic")
        rows.append(dict(state=list(corner), eigenvalues=eig.real.tolist(), stability=label))
    return rows
