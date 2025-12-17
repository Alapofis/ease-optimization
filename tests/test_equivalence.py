import numpy as np
import pytest
from scipy.sparse import csr_matrix
from ease import EASE
from ease.baseline_inverse import InverseSolver
from ease.cholesky import CholeskySolver
from ease.incremental.online_ease import IncrementalEASE

@pytest.fixture
def small_X():
    np.random.seed(42)
    X = np.random.randint(0,2,size=(6,5))
    return csr_matrix(X)

def test_ease_fit_and_score(small_X):
    # проверка формы B и score
    for solver in [InverseSolver(), CholeskySolver()]:
        model = EASE(solver=solver, reg=1e-3)
        model.fit(small_X)
        S = model.score(small_X)
        assert S.shape == small_X.shape
        assert np.all(np.isfinite(S))

def test_incremental_update(small_X):
    # incremental проверяем только на одном solver’e
    solver = InverseSolver()
    model = EASE(solver=solver, reg=1e-3)
    model.fit(small_X)

    inc_model = IncrementalEASE(model)
    inc_model.initialize(small_X)
    delta_X = csr_matrix(np.random.randint(0,2,size=small_X.shape))
    inc_model.update(delta_X)

    S_updated = model.score(small_X)
    assert S_updated.shape == small_X.shape
    assert np.all(np.isfinite(S_updated))

def test_incremental_equivalence(small_X):
    # проверка, что incremental без изменений даёт тот же score
    solver = CholeskySolver()
    model_full = EASE(solver=solver, reg=1e-3)
    model_full.fit(small_X)
    S_full = model_full.score(small_X)

    model_inc = EASE(solver=solver, reg=1e-3)
    model_inc.fit(small_X)
    inc_model = IncrementalEASE(model_inc)
    inc_model.initialize(small_X)
    # без новых данных
    inc_model.update(csr_matrix(np.zeros(small_X.shape)))
    S_inc = model_inc.score(small_X)

    assert np.allclose(S_full, S_inc, atol=1e-8)