# Supplementary classes and functions for ENGSCI233 notebook Sampling.ipynb
# author: David Dempsey

import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv  # function for matrix inverse
from scipy.integrate import trapz
from scipy.interpolate import interp1d
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

TEXTSIZE = 12

# interpolation
def pop_mpl(kwargs):
	color = kwargs.pop('color','k')
	mplkwargs = {'color':color}
	args = ['alpha','marker','mew','mfc','mec','ls','lw','label']
	defs = [1.0,'o',1.0,color,color,'-',0.0,None]
	for arg,default in zip(args,defs):
		mplkwargs.update({arg:kwargs.pop(arg,default)})
	return mplkwargs
def plot_data(xi,yi,ax, **kwargs):
	mplkwargs = pop_mpl(kwargs)
	ax.plot(xi,yi,**mplkwargs)
	ax.set_xlabel('time',size = TEXTSIZE)
	ax.set_ylabel('temperature',size = TEXTSIZE)	
	for t in ax.get_xticklabels()+ax.get_yticklabels():t.set_fontsize(TEXTSIZE)
	ax.legend(loc=2, prop={'size':TEXTSIZE});
def plot_interpolation_lines(xj, ax):
	ylim = ax.get_ylim()
	ax.set_ylim(ylim)
	for xji in xj[:-1]:
		ax.plot([xji,xji], ylim, 'r--', lw=0.5, alpha=0.4)

	# last line added separately so to provide legend label
	ax.plot([xj[-1],xj[-1]], ylim, 'r--', lw=0.5, alpha=0.4, label='interpolate at') 
	ax.legend(loc=2, prop={'size':TEXTSIZE});

	
	
# POLYNOMIAL FITTING
# ------------------
# fit a polynomial and plot data and function
def plot_polynomial_elements(ax, xi, yi, xj, m=1, interpolate=False, extrapolate = False):
	"""Fit polynomial of order M to data XI,YI and plot to axis AX
	"""
	# construct Vandemonde matrix
	A = vandermonde(xi,m)
	
	# construct RHS vector
	b = rhs(xi,yi,m)
	
	# solve Ax=b 
	# (note: I am solving x = A^-1 b, which is not wildly efficient)
	Ainv = inv(A)
	ak = np.dot(Ainv, b)
	
	# plotting
		# i. data
	plot_data(xi,yi,ax,label='data')
		# ii. interpolating function
	if not interpolate:
		if extrapolate:
			xm = (xi[0]+xi[-1])/2.
			xr = -(xi[0]-xi[-1])			
			x = np.linspace(0,2*xi[-1],1001)	# vector of x vals
		else:
			x = np.linspace(xi[0],xi[-1],1001)	# vector of x vals
		fx = polyval(ak,x)					# compute f(x)
		ax.plot(x,fx,'r-',label='{:d} order fit'.format(m))
		# iii. interpolated data
	if interpolate:
		# show lines
		plot_interpolation_lines(xj,ax)
		# evaluate interpolating function at XJ
		fxj = polyval(ak,xj)
		plot_data(xj,fxj,ax,color='r',marker='o',label='interpolated data')
	
	# add residual to plot
	if not interpolate:
		res = np.sum((polyval(ak,xi)-yi)**2)
		ax.annotate('R$^2$={:3.2e}'.format(res), xy=(.05,.7), xycoords='axes fraction', ha='left')

# construct righthandside vector for data XI, YI and polynomial order M
def rhs(xi,yi,m):
	"""Return least-squares righthand side vector for data XI, YI and polynomial order M
	"""
	# preallocate vector
	rhs = np.zeros(m+1)
	# compute terms
	for i in range(m+1):
		rhs[i] = np.sum(xi**i*yi)
	
	return rhs

# construct Vandermonde matrix for data XI and polynomial order M
def vandermonde(xi,m):
	"""Return Vandermonde matrix for data XI and polynomial order M
	"""
	# preallocate matrix
	V = np.zeros((m+1,m+1))
	# loop over rows
	for i in range(m+1):
		# loop over columns
		for j in range(m+1):
			V[i,j] = np.sum(xi**(i+j))
	return V

# evaluate polynomial with coefficients A at locations XI
def polyval(a,xi):
	"""Evaluautes polynomial with coefficients A at points XI.
	"""
	yi = 0.*xi
	for i,ai in enumerate(a):
		yi = yi + ai*xi**i
	return yi

	
	
# PIECEWISE LINEAR INTERPOLATION
# ------------------------------
# perform piecewise linear interpolation
def plot_piecewise_elements(ax, interpolate, xi, yi, xj):
	"""Fit straight line segments between neighbouring data pairs.
	"""
	# for each subinterval
	yj = []
	for xi1, yi1, xi2, yi2 in zip(xi[:-1], yi[:-1], xi[1:], yi[1:]):
		# compute gradient and intercept
		mi,ci = mx_c(xi1,yi1,xi2,yi2)
		
		# find interpolating points in subinterval
		inds = np.where((xj>=xi1)&(xj<xi2))
		
		# evaluate piecewise interpolating function at points
		yj += list(mi*xj[inds] + ci)		
	
	# plot data
	plot_data(xi,yi,ax,label='data')
	
	# other plotting
	if interpolate:
		# plot interpolation points
		plot_interpolation_lines(xj,ax)
		# plot interpolation values
		plot_data(xj,yj,ax,color='r',label='interpolated data')
	else:
		# plot interpolating function
		plot_data(xi,yi,ax,color='r',marker=None,lw=1.,label='piecewise linear interpolation')

# linear interpolation between points 
def mx_c(x1,y1,x2,y2):
	"""Returns gradient and y-intercept for straight line segment between the points (X1,Y1) and (X2,Y2)
	"""
	# gradient
	m = (y2-y1)/(x2-x1)
	# y-intercept
	c = y1-m*x1
	return m,c

	
	
# CUBIC SPLINE INTERPOLATION
# --------------------------
# perform cubic spline interpolation
def plot_spline_elements(ax,interpolate,SubIntEqn,xi,yi,xj):
	"""Fit cubic splines to data using built-in Python functions.
	"""
	# plot data
	plot_data(xi,yi,ax,label='data')
	
	f = interp1d(xi,yi,kind='cubic')
	if interpolate:
		# perform interpolation
		yj = f(xj)		
		# plot interpolation points
		plot_interpolation_lines(xj,ax)
		# plot interpolation values
		plot_data(xj,yj,ax,color='r',label='interpolated data')
	else:
		# plot interpolating function
		xv = np.linspace(xi[0], xi[-1], 1001)
		yv = f(xv)
		plot_data(xv,yv,ax,color='r',lw=1.0,label='cubic spline interpolation', marker=None)
		
	if SubIntEqn>0:
		# this is not going to be very elegant...
			# get subinterval
		x1, x2 = xi[SubIntEqn-1], xi[SubIntEqn]
			# evaluate spline at 1000 points inside interval
		xk = np.linspace(x1,x2,1000)
			# fit best cubic
		a = np.polyfit(xk,f(xk),deg=3)
			# show cubic for subinterval
		poly_str = r'$y$=${:2.1f}x^3$+${:2.1f}x^2$+${:2.1f}x$+${:2.1f}$'.format(*a)
		ls = '-'
		if interpolate:
			ls = '--'
		plot_data(xk,f(xk),ax,color='g',lw=2.0,ls=ls,label=poly_str, marker=None)
		
		
	
# INTEGRATION
# -----------	
# integration
def f_int(x): return (x-2)*(x-5.5)*(x-7)/8+8

def plot_integration_elements(ax, know_gx, subints, area):
	# configure area boolean
	if area == 'None': area = 0
	elif area =='A0': area = 1
	elif area =='A1': area = 2
	elif area =='A2': area = 3
	elif area =='Atot': area = -1
	
	# plot function or data
	if know_gx:
		x = np.linspace(2,8,1001)
		
		y = f_int(x)
		ax.plot(x,y,'r-', label='known function, $g(x)$')
	else:
		xi = np.array([2, 3.5, 6.8, 8.])
		yi = np.array([7.8, 8.5, 8.1, 10.0])
		ax.plot(xi,yi,'kx',ms=5,mew=2,label='known data, $(x_i,y_i)$')
	
	# show subintervals
	if subints:
		if know_gx:
			N=3 	# number of subintervals
			xi = np.linspace(x[0],x[-1],N+1)
			yi = f_int(xi)
			ax.plot(xi,yi,'kx',ms=5,mew=2,label='eval. function, $g(x_i)$')
		ax.plot(xi,yi,'k--')
		# dashed vertical lines
		label = 'three subintervals'
		for xii,yii in zip(xi,yi):
			ax.plot([xii,xii],[0,yii],'k--',label=label)
			label=None
		# subinterval numbering
		if area == 0:
			for xi1,xi2,yi1,yi2,i in zip(xi[:-1],xi[1:],yi[:-1],yi[1:], range(len(xi))):
				ax.text(0.5*(xi1+xi2), 0.25*(yi1+yi2), '$I_'+'{:d}'.format(i+1)+'$', ha = 'center', va = 'center', size = 14)
		
		if area > 0:	
			i = area - 1		
			patches = []
			
			i1 = i
			i2 = i+2
			if i2 == len(xi):
				poly = np.array([list(xi[i1:])+[xi[-1],xi[i1]],list(yi[i1:])+[0,0]]).T
			else:
				poly = np.array([list(xi[i1:i2])+[xi[i2-1],xi[i1]],list(yi[i1:i2])+[0,0]]).T
			xint = xi[i1:i2]
			yint = yi[i1:i2]
				
			area = trapz(yint,xint)
			polygon = Polygon(poly, zorder=1)
			patches.append(polygon)
			p = PatchCollection(patches, color = 'r', alpha = 0.2)
			ax.add_collection(p)
			
			ax.text(np.mean(xint), 0.5*np.mean(yint), '$A_'+'{:d}'.format(i)+'$'+'\n$=$\n${:3.1f}$'.format(area), ha = 'center', va = 'center', size = 12)
		
		if area < 0:	
			patches = []
			area = trapz(yi,xi)
			poly = np.array([list(xi)+[xi[-1],xi[0]],list(yi)+[0,0]]).T
			polygon = Polygon(poly, zorder=1)
			patches.append(polygon)
			p = PatchCollection(patches, color = 'r', alpha = 0.2)
			ax.add_collection(p)
			
			ax.text(np.mean(xi), 0.5*np.mean(yi), '$A_{tot}'+'$'+'\n$=$\n${:3.1f}$'.format(area), ha = 'center', va = 'center', size = 12)
		
	else:
		if area < 0:
			
			patches = []
			if know_gx:
				poly = np.array([list(x)+[x[-1],x[0]],list(y)+[0,0]]).T
				area = trapz(y,x)
			else:
				poly = np.array([list(xi)+[xi[-1],xi[0]],list(yi)+[0,0]]).T
				area = trapz(yi,xi)
				
			polygon = Polygon(poly, zorder=1)
			patches.append(polygon)
			p = PatchCollection(patches, color = 'r', alpha = 0.2)
			ax.add_collection(p)
			
			ax.text(5., 4, 'Area = {:3.1f}'.format(area), ha='center', va = 'center')
		
	
	# plotting
	ax.set_xlabel('time',size = TEXTSIZE)
	ax.set_ylabel('temperature',size = TEXTSIZE)
	ax.set_xlim([0,10])
	ax.set_ylim([0, 15])
	ax.legend(loc=2, prop={'size':TEXTSIZE})
	

	
# NEWTON-COTES METHODS
# --------------------
# interactive trapezium method demo
def plot_trapezium_elements(ax, N):
	# fit polynomial to data
	xi = np.array([2.5, 3.5, 4.5, 5.6, 8.6, 9.9, 13.0, 13.5])
	yi = np.array([24.7, 21.5, 21.6, 22.2, 28.2, 26.3, 41.7, 54.8])
	ak = fit_poly5(xi,yi)
	trapezium(ak,[xi[0], xi[-1]],N,ax)

# fit a fifth order polynomial
def fit_poly5(xi,yi):
	"""Return coefficients of fifth order polynomial fitted to data XI,YI.
	"""
	# construct Vandemonde matrix
	A = vandermonde(xi,5)
	
	# construct RHS vector
	b = rhs(xi,yi,5)
	
	# solve Ax=b 
	# (note: I am solving x = A^-1 b, which is not wildly efficient)
	Ainv = inv(A)
	ak = np.dot(Ainv, b)
	
	return ak

# integrate exactly a fifth order polynomial
def int_poly5(ak, xlim):
	akint = np.array([0.,]+[aki/(i+1) for i,aki in enumerate(ak)])
	return polyval(akint, xlim[1]) - polyval(akint, xlim[0])

# apply Trapezium method
def trapezium(ak,xlim,N,ax):
	"""Apply Trapezium method with N subintervals to polynomial with coefficients
	   AK over the interval XLIM.
	"""
	# construct subintervals and function evaluations
	xin = np.linspace(xlim[0], xlim[1], N+1)
	yin = polyval(ak,xin)
	
	# compute integral
	dx = xin[1]-xin[0]
	area = dx/2*(yin[0] + 2*np.sum(yin[1:-1]) + yin[-1])
	area_true = int_poly5(ak,xlim)
	
	# plotting
		# data
	xi = np.array([2.5, 3.5, 4.5, 5.6, 8.6, 9.9, 13.0, 13.5])
	yi = np.array([24.7, 21.5, 21.6, 22.2, 28.2, 26.3, 41.7, 54.8])
	#ax.plot(xi,yi,'ko',mfc='w',mew=1.5,label='data')
		# interpolating function
	xv = np.linspace(xi[0],xi[-1],1001)
	yv = polyval(ak,xv)
	ax.plot(xv,yv,'r-',label='$g(x)$')
		# subintervals
	ax.plot(xin,yin,'k--x',mec='r',mew=1.5,label='subintervals')
	for xini,yini in zip(xin,yin): 
		ax.plot([xini,xini],[0,yini],'k--')
					
	# plot upkeep
	ax.legend(loc=2, prop={'size': TEXTSIZE})
	ax.set_xlabel('time',size = TEXTSIZE)
	ax.set_ylabel('temperature',size = TEXTSIZE)
	str1 = '$A_{'+'{:d}'.format(N)+'}'+'={:3.1f}$'.format(area)
	str2 = '$A_{\infty}=$'+'${:3.1f}$'.format(area_true)
	str3 = '$\%\,\,err=$'+'${:3.1f}$'.format((area_true-area)/area_true*100)
	ax.annotate(str1+'\n'+str2+'\n'+str3, xy=(.05,.7), xycoords='axes fraction', ha='left', va='top', size = 12)
	
	ylim = ax.get_ylim()
	ax.set_ylim([0, ylim[-1]])




