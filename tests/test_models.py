import importlib.util
from pathlib import Path
import numpy as np
import pytest
from immigration_game.models import (parameters, payoffs, rhs, jacobian, simulate,
                                      reference_solution, corner_equilibria, time_grid)

ROOT = Path(__file__).resolve().parents[1]


def load_source(filename):
    spec = importlib.util.spec_from_file_location("archived_source", ROOT / "source_snapshot" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model", ["government", "firm"])
def test_original_payoffs_and_vector_field(model):
    source = load_source("replicator_sim_AB_no_pandas.py" if model == "government"
                         else "replicator_paper_ready_jp.py")
    rng = np.random.default_rng(42)
    p = parameters(model)
    for state in rng.uniform(.05, .95, (20, 3)):
        np.testing.assert_allclose(payoffs(state, p, model).ravel(),
                                   np.array(source.payoffs(*state, p)).ravel(), atol=1e-14)
        if model == "government":
            expected = source.replicator_rhs(state, p)
        else:
            step_p = dict(p, dt=1e-4, eps_clip=1e-9)
            expected = (np.array(source.replicator_step(*state, step_p)) - state) / 1e-4
        np.testing.assert_allclose(rhs(state, p, model), expected, atol=2e-12)


@pytest.mark.parametrize("model", ["government", "firm"])
def test_analytic_jacobian_against_finite_differences(model):
    p = parameters(model)
    for q in np.random.default_rng(7).uniform(.1, .9, (12, 3)):
        h = 1e-6
        numeric = np.column_stack([(rhs(q + np.eye(3)[j]*h, p, model)
                                     - rhs(q - np.eye(3)[j]*h, p, model)) / (2*h)
                                    for j in range(3)])
        np.testing.assert_allclose(jacobian(q, p, model), numeric, atol=1e-9)


def test_government_corners_and_open_policy_equilibrium():
    p = parameters()
    rows = corner_equilibria(p)
    assert len(rows) == 8
    for r in rows:
        np.testing.assert_allclose(rhs(r["state"], p), 0, atol=1e-14)
    np.testing.assert_allclose(np.sort(np.linalg.eigvals(jacobian([0, 1, 1], p))),
                               [-.7, -.6, -.1], atol=1e-14)
    assert sum(r["stability"] == "locally asymptotically stable" for r in rows) == 1


def test_mutation_pushes_inward_at_faces():
    p = parameters("firm")
    for j, key in enumerate(["mu_x", "mu_y", "mu_z"]):
        for edge, sign in [(0., 1), (1., -1)]:
            q = np.full(3, .4)
            q[j] = edge
            assert rhs(q, p, "firm")[j] == pytest.approx(sign*p[key]/2)


def test_cost_kink_requires_one_sided_analysis():
    p = parameters("firm", C3=.125)
    with pytest.raises(ValueError, match="kink"):
        jacobian([.4, .4, .5], p, "firm")
    for z in [.3, .7]:
        q = np.array([.4, .4, z])
        h = 1e-6
        np.testing.assert_allclose(jacobian(q, p, "firm")[:, 2],
                                   (rhs(q+[0, 0, h], p, "firm")-rhs(q-[0, 0, h], p, "firm"))/(2*h),
                                   atol=1e-9)


@pytest.mark.parametrize("model", ["government", "firm"])
def test_source_solver_tracks_adaptive_reference(model):
    p = parameters(model)
    t, q = simulate([.6, .3, .2], p, model, T=20, dt=.025)
    _, ref = reference_solution([.6, .3, .2], p, model, T=20, dt=.025)
    assert np.all((q >= 0) & (q <= 1))
    tolerance = 1e-6 if model == "government" else .01
    np.testing.assert_allclose(q, ref, atol=tolerance, rtol=0)
    assert t[-1] == 20


def test_parameter_and_grid_validation():
    with pytest.raises(ValueError):
        parameters("unknown")
    with pytest.raises(ValueError):
        parameters(C33=1)
    with pytest.raises(ValueError):
        time_grid(1, .3)
    with pytest.raises(ValueError):
        simulate([.5, -.1, .4], parameters())


def test_long_horizon_reference_and_floor_effect():
    p = parameters()
    _, continuous = reference_solution([.6, .3, .2], p)
    _, tiny_floor = simulate([.6, .3, .2], p, eps=1e-60)
    _, historical = simulate([.6, .3, .2], p)
    assert np.isfinite(continuous).all()
    assert np.all((continuous >= 0) & (continuous <= 1))
    np.testing.assert_allclose(tiny_floor, continuous, atol=2e-6, rtol=0)
    assert historical[-1, 1] > .99 and continuous[-1, 1] < 1e-15
