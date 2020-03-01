# import some pacakges
from scipy.linalg import lu
import numpy as np
from time import time
from multiprocessing import Pool
    
def factorise(N):
	''' create a random matrix and factorise it
	'''
	A = np.random.rand(N,N)
	lu(A)
	return None
	
if __name__ == "__main__":
	# parameters
	N = 3000
	n = 10
	ncpus = 8

	# factorise one matrix using lu()
	t0 = time()
	A = np.random.rand(N,N)
	P,L,U = lu(A)
	t1 = time()
	print('factorising 1 matrix: ',t1-t0, 'seconds')

	# factorise ten matrices using lu()
	t0 = time()
	for i in range(n):
		A = np.random.rand(N,N)
		P,L,U = lu(A)
	t1 = time()
	print('factorising {:d} matrices: '.format(n),t1-t0, 'seconds')

	# factorise ten matrices using factorise()
	t0 = time()
	for i in range(n):
		factorise(N)
	t1 = time()
	print('factorising {:d} matrices: '.format(n),t1-t0, 'seconds')
	
	# factorise ten matrices using factorise() and multiprocessing
	for ncpu in range(2, ncpus+1):
		p = Pool(ncpu)
		t0 = time()
		p.map(factorise, [N for i in range(n)])
		t1 = time()
		print('factorising {:d} matrices with {:d} cpus: '.format(n,ncpu),t1-t0, 'seconds')
	