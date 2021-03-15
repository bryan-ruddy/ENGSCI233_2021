from funcs import *
import pytest

def test_answer():
	tol=1e-10
	assert approx_factorial(0.) < tol
    assert approx_factorial(1) - 0.9221370088957891 < tol
    assert approx_factorial(np.inf) == np.inf
    assert approx_factorial(-np.inf) == 0
	
def test_answer2():
	
	with pytest.raises(ValueError)
        approx_factorial(-3)
    with pytest.raises(TypeError):
        approx_factorial('Colorless green ideas sleep furiously')
	