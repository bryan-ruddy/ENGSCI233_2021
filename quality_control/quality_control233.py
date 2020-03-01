# Supplementary classes and functions for ENGSCI233 notebook quality_control.ipynb
# author: David Dempsey

# module imports
import numpy as np

def harmonic_mean(xs):
	''' Compute the harmonic mean of XS
		
		Parameters:
		-----------
		xs : array-like
			list of values
			
		Returns:
		--------
		xharm : float
			harmonic mean
		
		Notes:
		------
		xs cannot be empty or contain zero values
		
	'''
	# check preconditions
	assert not any([xi == 0. for xi in xs]), 'cannot have zero-values... idiot'
	assert len(xs) > 0, 'xs has zero-length'
	
	# compute harmonic mean
	xharm = 1./(np.sum([1./xi for xi in xs]))
	
	# check number is defined
	if xharm != xharm:
		raise ValueError('Computation returns NaN')
	
	return xharm

def geometric_mean(xs):
	''' Compute the geometric mean of XS
		
		Parameters:
		-----------
		xs : array-like
			list of values
			
		Returns:
		--------
		xgeom : float
			geometric mean
		
		Notes:
		------
		xs cannot be empty
		
	'''
	# check preconditions
	assert len(xs) > 0, 'xs has zero-length'
	
	# compute geometric mean
	xgeom = np.exp(np.mean([np.log(xi) for xi in xs]))
	
	# check number is defined
	if xgeom != xgeom:
		raise ValueError('Computation returns NaN')
	
	return xgeom