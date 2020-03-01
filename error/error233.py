# Supplementary classes and functions for ENGSCI233 notebook Error.ipynb
# author: David Dempsey

# module imports
import numpy as np
from matplotlib import pyplot as plt
from ipywidgets import Checkbox, Dropdown, IntSlider, Layout, HBox, VBox, interact, fixed
from IPython.display import display
from decimal import *
getcontext().prec = 128

# Class and functions for "Fun with Floats" exercise
class Fwf:
	def __init__(self, n=7):
		# create empty axes
		self.fig,self.ax = plt.subplots(1,1)
		self.fig.set_size_inches([9,1])
		self.ax.axis('off')
		self.ax.set_xlim([0,1])
		self.ax.set_ylim([0,1])
		
		# create empty text annotations
		self.text = self.ax.text(-0.05,0.5, '', size = 12, ha = 'left')
				
		# create widgets
		items = [Checkbox(False, description='1/{:d}'.format(2**i)) for i in range(1,n)]
		items = [Dropdown(options = ['-4','-3','-2','-1','0','1','2','3','4'], value = '0', description='$e$')]+items
		self.widgets = items
				
		# set initial state
		self.update(None)
	def update(self, change):
		# get widget state
		fracs = []
		exponent = int(self.widgets[0].value)
		fracs = [w.value for w in self.widgets[1:]]
		
		# assemble new string
		# -------------------
			# exponent
		sum_str = '$\\quad=\\quad 2^{'+'{:d}'.format(exponent)+'}\\times (1\\quad+\\quad$'
		sum_flt = 1.
			# significand
		for i,j in enumerate(fracs):
			if j:
				sum_str += '$\\frac{1}{'+'{:d}'.format(2**(i+1))+'}\\quad+\\quad$'												   
				sum_flt += 1./(2**(i+1))
			# float value
		sum_flt *= 2**exponent
		sum_str = '$'+'{:6.5e}'.format(sum_flt)+'$'+sum_str[:-12]+')$'	
		
		# update annotation
		self.text.set_text(sum_str)
		
		# redraw
		self.fig.canvas.draw()
	def plot(self):
		# check widgets for updates
		for w in self.widgets:
			w.observe(self.update, 'value')
		self.update(None)
		
		# widget layout
		col_layout=Layout(display='flex', flex_flow='row-wrap', width="100%", justify_content='flex-end', align_items='center')
		return HBox(self.widgets, layout=col_layout)
		
def fun_with_floats():
	fwf = Fwf()
	w = fwf.plot()
	display(w)

# Class and functions for "Exponential" exercise
class Exp:
	def __init__(self, delta_e):
		# save delta e function
		self.delta_e = delta_e
		
		# create empty axes
		self.fig,self.ax = plt.subplots(1,1)
		self.fig.set_size_inches([9,3])
		self.ax.axis('off')
		self.ax.set_xlim([0,1])
		self.ax.set_ylim([-0.40,1])
		
		# create empty text annotations
		self.string = ''
		self.text0 = self.ax.text(-0.135,0.90, '', size = 15, ha = 'left')
		self.text1 = self.ax.text(-0.135,0.65, '', size = 15, ha = 'left')
		self.text2 = self.ax.text(-0.105,0.4, '', size = 15, ha = 'left')
		self.text3 = self.ax.text(-0.075,0.05, '', size = 15, ha = 'left', color = 'r', bbox=dict(facecolor='none', edgecolor='red', linestyle='--'))
		
		self.text4 = self.ax.text(-0.135,-0.3, '', size = 15, ha = 'left', color='b')
				
		# create widgets
		items = [
			IntSlider(value = 1, description='terms', min=1, max = 20, step=1),
			Dropdown(options = ['half','single','double'], value = 'single')
			]
		self.widgets = items
			
		# set initial state
		self.update(None)
	def update(self, change):
		# grab current precision from appropriate widget
		precision = self.widgets[1].value
		if precision == 'half':
			prec = np.float16
		elif precision == 'single':
			prec = np.float32
		else:
			prec = np.float64
			
		# grab current term calculation from appropriate widget
		terms = self.widgets[0].value
		
		# calculation
		# -----------
			# initialize term calculation
		sum_terms0 = prec(0.)
		final_term = prec(0.)		
			# iterate over number of terms
		for i in range(0,terms):
			final_term = self.delta_e(i, prec)
			sum_terms0 += final_term
			# compute and append final term
		final_term = self.delta_e(terms, prec)
		sum_terms = sum_terms0 + final_term
		
		# update annotations
		pr = '{:51.50f}'
		self.text0.set_text(r'       e$_{'+'{:d}'.format(terms-1)+'}$'+r'$:$  '+pr.format(sum_terms0))
		self.text1.set_text(r'  $+\Delta$e$_{'+'{:d}'.format(terms)+'}$'+r'$:$  '+pr.format(final_term))
		self.text2.set_text(r'$=$e$_{'+'{:d}'.format(terms)+'}$'+r'$:$  '+pr.format(sum_terms))
		e = '2.71828182845904523536028747135266249775724709369995'
		self.text3.set_text('e$_\infty:$  {}'.format(e))
				
		rd = Decimal(pr.format(sum_terms))-(Decimal(pr.format(final_term))+Decimal(pr.format(sum_terms0)))
		self.text4.set_text('err$_{round}:$'+'  '+pr.format(rd))
		
		# redraw
		self.fig.canvas.draw()
	def plot(self):
		# check widgets for updates
		for w in self.widgets:
			w.observe(self.update, 'value')
		self.update(None)
		
		# widget layout
		col_layout=Layout(display='flex', flex_flow='row-wrap', width="100%", justify_content='center', align_items='center')
		return HBox(self.widgets, layout=col_layout)
	
def exponential_example(delta_e):
	exp = Exp(delta_e)
	w = exp.plot()
	display(w)
	