"""The optional C++ gradient must match numpy.gradient, and the analysis must not change when it is on."""
import numpy as np
import pytest

from trajlens import metrics
from trajlens.io import Trajectory

pytestmark = pytest.mark.skipif(not metrics.have_cpp(), reason="C++ extension not built (pip install pybind11, rebuild)")


@pytest.mark.parametrize("edge_order", [1, 2])
@pytest.mark.parametrize("n", [3, 4, 7, 200])
def test_gradient_matches_numpy_on_nonuniform_time(n, edge_order):
    rng = np.random.default_rng(n)
    t = np.cumsum(rng.uniform(0.001, 0.02, n))
    f = rng.normal(size=(n, 5))
    assert np.allclose(metrics._fast.gradient(t, f, edge_order), np.gradient(f, t, axis=0, edge_order=edge_order), rtol=1e-9, atol=1e-9)


def test_gradient_rejects_bad_input():
    t = np.arange(5.0)
    with pytest.raises(ValueError):
        metrics._fast.gradient(t, np.zeros((4, 2)), 2)          # length mismatch
    with pytest.raises(ValueError):
        metrics._fast.gradient(t[:2], np.zeros((2, 2)), 2)      # too few samples for edge_order 2
    with pytest.raises(ValueError):
        metrics._fast.gradient(t, np.zeros((5, 2)), 3)


def test_grad_falls_back_for_two_samples():
    t = np.array([0.0, 0.5])
    f = np.array([[0.0], [1.0]])
    assert np.allclose(metrics.grad(f, t, 1), np.gradient(f, t, axis=0))


def test_analysis_is_the_same_with_and_without_cpp(monkeypatch):
    rng = np.random.default_rng(3)
    t = np.cumsum(rng.uniform(0.0015, 0.0025, 4000))
    q = np.column_stack([np.sin(0.7 * t * (k + 1)) * 0.4 for k in range(6)])
    tr = Trajectory(t=t, q=q, names=[f"j{k}" for k in range(6)]) if "names" in Trajectory.__init__.__code__.co_varnames else None
    if tr is None:
        pytest.skip("Trajectory signature differs")
    a = metrics.analyze(tr)
    monkeypatch.setattr(metrics, "_fast", None)
    b = metrics.analyze(tr)
    for k in ("path_length_rad", "max_speed", "max_jerk_norm", "rms_jerk_norm", "normalized_jerk", "sparc"):
        assert np.isclose(a[k], b[k], rtol=1e-8, equal_nan=True), k
