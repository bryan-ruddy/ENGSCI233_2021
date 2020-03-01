# Supplementary classes and functions for ENGSCI233 notebook Data.ipynb
# author: David Dempsey

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# Module 1: Data
def show_list(ll, ax, highlight=None, label=None, **kwargs):

	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	ax.axis('off')
	
	# check whether popped given
	plot_popped = 'popped' in kwargs.keys()
	
	# width of item
	N = ll.get_length()
	if plot_popped:
		N += 1
	w = 1./(2*N+3)
	y0,y1 = [0.2,0.8]
	
	# draw items
	w0 = w	
	# print popped value first, if present
	if plot_popped:
		# draw box
		poly = np.array([[w0,w0+w,w0+w,w0,w0],[y0,y0,y1,y1,y0]]).T
		polygon = Polygon(poly, zorder=1)
		p = PatchCollection([polygon,], facecolor = 'm', edgecolor = 'k', alpha = 0.3)
		ax.add_collection(p)	
		
		# add text
		ax.text(w0+0.5*w, 0.5*(y0+y1), '{}'.format(kwargs['popped']), ha = 'center', va='center')
		
		w0 += 2*w
		
		N = N-1
		
	for i in range(N):
		# draw box
		poly = np.array([[w0,w0+w,w0+w,w0,w0],[y0,y0,y1,y1,y0]]).T
		polygon = Polygon(poly, zorder=1)
		if i == highlight:
			col = 'g'
		elif i == 0:
			col = 'r'
		else:
			col = 'b'
		p = PatchCollection([polygon,], facecolor = col, edgecolor = 'k', alpha = 0.3)
		ax.add_collection(p)		
		
		# add text
		ax.text(w0+0.5*w, 0.5*(y0+y1), '{}'.format(ll.get_value(i)), ha = 'center', va='center')
		
		w0 += w
				
		# draw arrow
		col = [0.5,0.5,0.5]
		if highlight is not None:
			if i < highlight:
				col = 'k'
		ax.arrow(w0, 0.5*(y0+y1), w, 0, length_includes_head=True, head_length = 0.01, head_width = 0.1, color = col)
		w0 += w
	
	
	# draw null
	poly = np.array([[w0,w0+w,w0+w,w0,w0],[y0,y0,y1,y1,y0]]).T
	polygon = Polygon(poly, zorder=1)
	p = PatchCollection([polygon,], facecolor = 'w', edgecolor = 'k', linestyle = '--')
	ax.add_collection(p)	
	
	ax.text(w0+0.5*w, 0.5*(y0+y1), 'None', ha = 'center', va='center')
	
	# add label
	if label is not None:
		ax.text(w0+1.5*w, 0.5*(y0+y1), label, ha = 'left', va='center')
