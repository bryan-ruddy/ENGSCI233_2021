from funcs import *
import pytest

def test_answer():
	tol=1e-10
	assert approx_factorial(0.) < tol
    assert approx_factorial(1) - 0.9221370088957891 < tol
    assert approx_factorial(np.inf) == np.inf
	
def test_answer2():
	
	assert iscomplex(approx_factorial(-1))
    with pytest.raises(TypeError):
        approx_factorial('Colorless green ideas sleep furiously')
	