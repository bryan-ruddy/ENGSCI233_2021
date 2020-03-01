# Supplementary classes and functions for ENGSCI233 notebook Combinatorics.ipynb
# author: David Dempsey

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.collections import PatchCollection
from glob import glob
from copy import copy
from ipywidgets import interact, fixed, IntSlider, VBox, HBox, interactive_output

import traitlets
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Classes and functions for LINKED LIST and NETWORK
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# --- for linked lists
class ListNode(object):
	'''A class with methods for node object.
	'''
	def __init__(self, value, pointer):
		'''Initialise a new node with VALUE and POINTER
		'''
		self.value = value
		self.pointer = pointer
		
	def __repr__(self):
		return "nd: {}".format(self.value)
			
	def next(self):
		'''Returns the next node.
		'''
		return self.pointer
class LinkedList(object):
	'''A class with methods to implement linked list behavior.
	'''
	def __init__(self):
		'''Initialise an empty list.
		'''
		self.head = None
	def __repr__(self):
		'''Print out values in the list.
		'''
		# special case, the list is empty
		if self.head is None:
			return '[]'
		
		# print the head node
		ret_str = '['					   # open brackets
		node = self.head
		ret_str += '{}, '.format(node.value) # add value, comma and white space
		
		# print the nodes that follow, in order
		while node.pointer is not None:	 # stop looping when reach null pointer
			node = node.next()			  # get the next node
			ret_str += '{}, '.format(node.value)
		ret_str = ret_str[:-2] + ']'		# discard final white space and comma, close brackets
		return ret_str
	def append(self, value):
		'''Insert a new node with VALUE at the end of the list.
		'''
		# insert value at final index in list		
		self.insert(self.get_length(), value)	 
	def insert(self, index, value):
		'''Insert a new node with VALUE at position INDEX.
		'''
		# create new node with null pointer
		new_node = ListNode(value, None)
		
		# special case, inserting at the beginning
		if index == 0:
			# new node points to old head
			new_node.pointer = self.head
			# overwrite list head with new node
			self.head = new_node
			return
		
		# get the node immediately prior to index
		node = self.get_node(index-1)
		
		# logic to follow
		if node is None:					# special case, out of range
			print("cannot insert at index {:d}, list only has {:d} items".format(index, self.get_length()))
		elif node.next() is None:		   # special case, inserting as last node
			node.pointer = new_node
		else:
			# point new node to node after new node
			new_node.pointer = node.next()
			# node before new node points to new node
			node.pointer = new_node
	def pop(self, index):
		'''Delete node at INDEX and return its value.
		'''
		# special case, index == 0 (delete head)
		if index == 0:
			# popped value
			pop = self.head.value
			# set new head as second node
			self.head = self.head.next()
			return pop
		
		# get the node immediately prior to index
		node = self.get_node(index-1, verbose=False)
		
		# logic to follow
		if node is None:					# special case, out of range
			print("cannot access index {:d}, list only has {:d} items".format(index, self.get_length()))
			return None
		elif node.next() is None:		  # special case, out of range
			print("cannot access index {:d}, list only has {:d} items".format(index, self.get_length()))
			return None
		elif node.next().next() is None:  # special case, deleting last node
			# popped value
			pop = node.next().value
			
			# make prior node the last node
			node.pointer = None
		else:
			# popped value
			pop = node.next().value
			
			# set this nodes pointer so that it bypasses the deleted node
			node.pointer = node.next().next()
		
		return pop
	def delete(self,index):
		'''Delete node at INDEX.		
		'''
		# use pop method and discard output
		self.pop(index)
	def get_length(self):
		'''Return the length of the linked list.
		'''
		# special case, empty list
		if self.head is None:
			return 0
		
		# initialise counter
		length = 1
		node = self.head
		while node.pointer is not None:
			node = node.next()
			length += 1
			
		return length
	def get_node(self, index):
		'''Return the node at INDEX.
		'''
		# special case: index = -1, retrieve last node
		if index == -1:
			# begin at head
			node = self.head
			
			# loop through until Null pointer
			while node.pointer is not None:
				node = node.next()
			return node
		
		# begin at head, use a counter to keep track of index
		node = self.head
		current_index = 0
		
		# loop through to correct index
		while current_index < index:
			node = node.next()
			if node is None:
				return node
			current_index += 1
		return node
	def get_value(self, index):
		'''Return the value at INDEX.
		'''
		# get the node at INDEX
		node = self.get_node(index)
		
		# return its value (special case if node is None)
		if node is None:
			return None
		else: 
			return node.value
# --- for networks
class Node(object):
	def __init__(self):
		self.name = None
		self.value = None
		self.arcs_in = []
		self.arcs_out = []
	def __repr__(self):
		return '{}:{}'.format(self.name,self.value)
	def get(self,name):
		for arc in self.arcs_out:
			if arc.to_node.name == name:
				return arc.to_node
		raise ValueError("node {} is not a daughter of node {}".format(name, self.name))
class Arc(object):
	def __init__(self):
		self.weight=None
		self.to_node = None
		self.from_node = None
class NetworkError(Exception):
	'''An error to raise when violations occur.
	'''
	pass
class Network(object):
	''' Basic network class - you implemented this in the lab last week.
	'''
	def __init__(self):
		self.nodes = []
		self.arcs = []
	def add_node(self, name, value=None):
		'''Adds a Node with NAME and VALUE to the network.
		'''
		# check node names are unique
		network_names = [nd.name for nd in self.nodes]
		if name in network_names:
			raise NetworkError
		
		# new node, assign values, append to list
		node = Node()
		node.name = name
		node.value = value
		self.nodes.append(node)
	def join_nodes(self, node_from, node_to, weight):
		'''Adds an Arc joining NODE_FROM to NODE_TO with WEIGHT.
		'''
		# new arc
		arc = Arc()
		arc.weight = weight
		arc.to_node = node_to
		arc.from_node = node_from
		# append to list
		self.arcs.append(arc)
		# make sure nodes know about arcs
		node_to.arcs_in.append(arc)
		node_from.arcs_out.append(arc)
	def get_node(self, name):
		''' Loops through the list of nodes and returns the one with NAME.
		
			Returns None if node does not exist.
		'''
		for node in self.nodes:
			if node.name == name:
				return node
		
		raise NetworkError
		
		
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Classes and functions for SEARCH
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Tree(Network):
	''' Derived class of NETWORK. There is one head node, which has daughter nodes.
		Each daughter node may have its own daughter nodes.
	'''
	def build(self, tree_tuple):
		''' Build the tree from recursive TREE_TUPLE
		
			Tuple pairs contain node name first and then either None (indicating no 
			daughters) or another tuple of the same structure.
		'''
		# check that top generation has only one member
		assert(len(tree_tuple)==2)
		
		# build a network tree recursively
		k = tree_tuple[0]
		self.add_daughter(k,tree_tuple[1],None)
		self.head = self.get_node(k)
	def add_daughter(self,name,daughters,mother):
		''' Add new node NAME, link to MOTHER, recursively add DAUGHTERS.
		'''
		# adding the new node, link to mother (unless head node)
		self.add_node(name)
		if mother is not None:
			self.join_nodes(self.get_node(mother), self.get_node(name), 1)
			
		# if additional generation information, recursively add new daughters
		if daughters is not None:
			for daughter in daughters:
				self.add_daughter(daughter[0], daughter[1], name)
	
	def assign_values(self, val_dict):
		''' Assigns values to nodes from VAL_DICT.
		
			Keys of VAL_DICT are node names.
		'''
		for k in val_dict.keys():
			self.get_node(k).value=val_dict[k]

	def show(self, highlight = []):
		''' Don't worry about these rather involved plotting commands.
		
			They are here to give you visual feel of the tree structure.
		'''
		# count generations
		generations = 1
		still_looking = True
		current_generation = [self.head,]
			
		while still_looking:
			next_generation = []
			for node in current_generation:
				for arc in node.arcs_out:
					next_generation.append(arc.to_node)
				
			if len(next_generation)>0:
				generations +=1
			else:
				still_looking = False
			
			current_generation = copy(next_generation)
		
		f,ax = plt.subplots(1,1)
		
		f.set_size_inches(8,generations)
		
		ax.set_xlim([0,1])
		ax.set_ylim([0,generations])
		
		y = generations-0.5
		x = [0,1]
		
		props0 = dict(boxstyle='round', facecolor='white', alpha=1.0)
		still_looking=True
		current_generation = [self.head,]
		locs = {}
		while still_looking:
			next_generation = []
			xnew = []
			for node,x0,x1 in zip(current_generation,x[:-1],x[1:]):
				# plot
				props = copy(props0)
				if node.name in highlight:
					props['facecolor']=[1,0.8,1]
					
				ax.text(0.5*(x0+x1), y, '{}: {}'.format(node.name, node.value), ha='center',va='center',bbox=props,size=14)
				locs.update({node.name:[0.5*(x0+x1), y]})
				
				if len(node.arcs_in)>0:
					frm = node.arcs_in[0].from_node.name
					xa,ya = locs[frm]
					xb,yb = locs[node.name]
					ax.plot([xa,xb],[ya,yb],'k-')
				
				# find next generation
				for arc in node.arcs_out:
					next_generation.append(arc.to_node)
				
				xnew += list(np.linspace(x0,x1,len(node.arcs_out)+1))[:-1]
				
			y = y-1
			xnew += [x1,]
				
			if len(next_generation)>0:
				generations +=1
			else:
				still_looking = False
			
			current_generation = copy(next_generation)
			x = copy(xnew)
			
		ax.axis('off')
def interactive_search(search=0, check=1, tree=None):
	if search==1:
		order = ['A','B','C','D','E','F','G','H']
	else:
		order = ['A','B','D','E','F','C','G','H']
	tree.show(highlight = order[check-1])
	
	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Classes and functions for INSERTION SORT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def insertion_sort(A, plot_step):
	''' Sort array A using insertion sort.
		
		Parameters
		----------
		A : array
			Array of values to be sorted.
		plot_step : int
			Flag indicating which sort step to be displayed.
								
		Notes
		-----
		Calls to SHOW are for plotting and are not part of the sort
		algorithm.
		
	'''
	
	#
	cnt = 0
	cnt = show(A, 0, 0, '   ', -1, cnt, plot_step)
	#
	
	# initialise
	n = len(A)
	
	# for each value in the array, beginning with the second 
	for j in range(1,n):
		# step - assign value to key
		key = A[j]
		cnt = show(A, j, 0, key, 0, cnt, plot_step)
		
		i = j-1
		while i> -1:
			cnt = show(A, j, i, key, 1, cnt, plot_step)
			# step - compare array values to left (of key) against key			
			if not (A[i]> key):
				# condition to stop comparison
				break
			# step - shift value to the right
			A[i+1] = A[i]
			cnt = show(A, j, i, key, 2, cnt, plot_step)
			
			i = i-1
		# step - insert key into new position
		A[i+1] = key
		cnt = show(A, j, i, key, 3, cnt, plot_step)
def show(A, j, i, key, case, step_counter, plot_step):
	''' A plotting function. You do not need to know the details.
	'''
	step_counter +=1
	if step_counter-1 != plot_step: 
		return step_counter
		
	textsize = 12
	
	f,ax = plt.subplots(1,1)
	f.set_size_inches(10,1)
	ax.set_xlim([0,1])
	ax.set_ylim([0,1])
	ax.axis('off')
	
	n = len(A)+3
	dw = 1./n
	
	x0 = dw/2.
	y0,y1,y2,y3,y4,y5 =[0.15, 0.4, 0.65, 0.7, 0.85,0.88]
	
	# plot the key
	if case == 0:
		color = [0.8,1,0.8] # green = key copy
		arrow_text = 'copy key: key=A[j]'
		xa1 = x0+dw/2
	elif case == 1:
		color = [1,0.8,0.8] # red = compare values
		arrow_text = 'compare values: key<A[i]?'
		xa1 = x0+dw/2
	elif case == 2:
		color = 'w'
		arrow_text = 'yes, shift value: A[i]->A[i+1]'
	elif case == 3:
		color = [0.8,0.8,0.8] # grey = insert key
		if i == -1:
			arrow_text = 'insert key: A[i+1]=key'
		else:
			arrow_text = 'no, insert key: A[i+1]=key'
		xa1 = x0+dw/2
	else:
		color = 'w'
	
	poly = np.array([[x0,x0+dw,x0+dw,x0,x0],[y0,y0,y2,y2,y0]]).T
	polygon = Polygon(poly, zorder=1)
	p = PatchCollection([polygon,], color = color, edgecolor = 'k', linewidth=2)
	ax.add_collection(p)
	ax.text(x0+dw/2., y1, '{}'.format(key), ha='center', va = 'center',size = textsize, zorder = 2)
	x0 += 2*dw
	
	# plot the array
	for ia, a in enumerate(A):
		
		if case == 0 and ia == j:   # green = copy key
			color = [0.8,1,0.8]
			xa2 = x0+dw/2.
		elif case == 1 and ia == i:  # red = compare values
			color = [1, 0.8, 0.8]
			xa2 = x0+dw/2.
		elif case == 2 and (ia == i or ia == i+1): # blue = shift value
			color = [0.8,0.8,1]
			xa1 = x0 - dw/2.
			xa2 = xa1 + dw
		elif case == 3 and (ia == i+1):  # grey = insert key
			color = [0.8,0.8,0.8]
			xa2 = x0+dw/2.
		else:
			color = 'w'
			
		
		poly = np.array([[x0,x0+dw,x0+dw,x0,x0],[y0,y0,y2,y2,y0]]).T
		polygon = Polygon(poly, zorder=1)
		p = PatchCollection([polygon,], color = color, edgecolor = 'k', linewidth=2)
		ax.add_collection(p)
	
		ax.text(x0+dw/2.,y1,'{}'.format(a), ha = 'center', va = 'center', size=textsize, zorder = 2)
		x0 += dw
	
	# plot arrows
	if case > -1:
		ax.plot([xa1, xa1, xa2, xa2],[y3, y4, y4, y3],'k-')
		hw, hl = [0.01, 0.05]
		if case == 0:
			ax.arrow(xa1, y4, 0, y3-y4, length_includes_head=True, head_width = hw, head_length = hl)
		elif case == 1:
			ax.arrow(xa1, y4, 0, y3-y4, length_includes_head=True, head_width = hw, head_length = hl)
			ax.arrow(xa2, y4, 0, y3-y4, length_includes_head=True, head_width = hw, head_length = hl)
		elif case == 2:
			ax.arrow(xa2, y4, 0, y3-y4, length_includes_head=True, head_width = hw, head_length = hl)
		elif case == 3:
			ax.arrow(xa2, y4, 0, y3-y4, length_includes_head=True, head_width = hw, head_length = hl)
		ax.text(0.5*(xa1+xa2), y5, arrow_text, ha = 'center', va = 'bottom', size = textsize)		
		
	return step_counter

	
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Classes and functions for HEAP SORT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Heap(Network):
	''' Derived class of NETWORK. A tree structure where each node has at most two daughters.
	'''
	def get_node_with_value(self, value):
		''' Loops through the list of nodes and returns the one with VALUE.
		
			Returns None if node does not exist.
		'''
		for node in self.nodes:
			if node.value == value:
				return node
		
		raise NetworkError
	def initialise(self, A, step_counter, plot_step):
		''' Each of the unordered values in A is assigned as a new node in the network.
		
			STEP_COUNTER and PLOT_STEP are to enable plotting in IPython notebook. Ignore plotting commands.
		'''
		# add first node to top of heap (head of the tree network and first item in queue)
		self.queue = LinkedList()			# use a queue to keep track of which nodes new daughters should be linked to
		self.add_node(name=0, value = A[0]) # name = node number, value = array value
		self.head = self.get_node(0)		# initialise head node
		self.queue.append(self.head)		# head node added to queue
		
		# counter to keep track of nodes added
		i = 1 								
		
		# **plotting: ignore** -------------------------------------------
		step_counter += 1
		if step_counter == plot_step:
			print('allocating unordered values to heap: ', A[i:])
			self.show(self.head, 1)
		# **end ignore** -------------------------------------------------
		
		# loop to keep adding array values to the heap
		n = len(A) 	  			# length of array
		keep_adding = True		# boolean for while loop
		while keep_adding:
			# pull next node from queue: mother to (up to) two new daughter nodes
			mother = self.queue.pop(0)
			
			# add and link first daughter to the mother node
			self.add_node(name=i, value = A[i])
			daughter = self.get_node(i)
			# join to parent
			self.join_nodes(mother, daughter, 1)
			# append the daughter to the queue
			self.queue.append(daughter)
			
			# **plotting: ignore** ---------------------------------------
			step_counter += 1
			if step_counter == plot_step: 
				print('allocating unordered values to heap: ', A[i+1:])
				self.show(daughter, 1)
			# **end ignore** ---------------------------------------------
			
			# assess stopping condition
			i += 1
			if i == n:
				keep_adding = False
				continue
  
			# add and link first daughter to mother node
			self.add_node(name=i, value = A[i])
			daughter = self.get_node(i)
			# join to parent
			self.join_nodes(mother, daughter, 1)
			# append the daughter to the queue
			self.queue.append(daughter)
			
			# **plotting: ignore** ---------------------------------------
			step_counter += 1
			if step_counter == plot_step: 
				print('allocating unordered values to heap: ', A[i+1:])
				self.show(daughter, 1)
			# **end ignore** ---------------------------------------------
			
			# assess stop condition
			i += 1
			if i == n:
				keep_adding = False
				continue
	def build(self, step_counter, plot_step):
		''' Build the heap through promotion and demotion of mother-daughter pairs.
		
			NOTE: THIS IS A HARD-CODED BUILD COMMAND THAT PERTAINS ONLY TO THE SPECIFIC CASE IN THE 
			COMBINATORICS NOTEBOOK. 
			
			YOU WILL HAVE TO REWRITE THIS METHOD AS PART OF THE COMBINATORICS LAB.
			
			STEP_COUNTER and PLOT_STEP are to enable plotting in IPython notebook. Ignore plotting commands.
		'''
		# hard coded sort order with highlighting
		demotions = [
			[2,4,3,6,7,9,8,1,5,10,  0, 10],
			[2,4,3,6,7,9,8,1,5,10,  0,  5],
			[2,4,3,6,7,9,8,1,5,10,  0,  1],
			[2,4,3,6,7,9,8,1,5,10,  0,  8],
			[2,4,3,6,7,9,8,1,5,10,  0,  9],
			[2,4,3,6,7,9,8,1,5,10,  1,  7],
			[2,4,3,6,10,9,8,1,5,7,  2,  7],
			[2,4,3,6,10,9,8,1,5,7,  1,  6],
			[2,4,3,6,10,9,8,1,5,7,  1,  3],
			[2,4,9,6,10,3,8,1,5,7,  2,  3],
			[2,4,9,6,10,3,8,1,5,7,  1,  4],
			[2,10,9,6,4,3,8,1,5,7,  2,  4],
			[2,10,9,6,7,3,8,1,5,4,  2,  4],
			[2,10,9,6,7,3,8,1,5,4,  1,  2],
			[10,2,9,6,7,3,8,1,5,4,  2,  2],
			[10,7,9,6,2,3,8,1,5,4,  2,  2],
			[10,7,9,6,4,3,8,1,5,2,  2,  2],]
		
		for i, demotion in enumerate(demotions):
			self.nodes = []
			self.arcs = []
			self.initialise(demotion[:-2],0,-1)
			if plot_step == (i+step_counter+1):
				if demotion[-2] == 0:
					print('ordering heap: no daughters')
				elif demotion[-2] == 1:
					print('ordering heap: has daughters, attempt demotions') 
				elif demotion[-2] == 2:
					print('ordering heap: demoting...')
				self.show(self.get_node_with_value(demotion[-1]), demotion[-2])
	def demote(self, node):
		''' Implements iterative promotion-demotion of NODE.
		
			Check node value against daughter values. If either daughter exceeds,
			the larger is promoted to NODE's position and NODE is demoted to the 
			daughter position.
			
			In its new position, NODE is checked again for demotion (it has new 
			daughters now). Promotion-demotion checks continue until NODE's value
			exceeds both daughters or it is demoted to a position where it has no
			daughters.
		'''
		# initialise value to exceed as NODE value
		max_val = node.value
		if max_val is None:
			# if node has no value, assign exceedingly small value. Intent is to 
			# guarantee demotion of the node
			max_val = -1.e32
		# currently, not node is scheduled for promotion
		promote_node = None
		
		# check all arcs out of NODE
		for arc_out in node.arcs_out:
			# get daughter
			daughter = arc_out.to_node
			# if daughter has no value, not a candidate for promotion
			if daughter.value is None:
				continue
			# if daughter value larger than NODE or previous daughter
			if daughter.value > max_val:
				# update, new value to exceed for promotion
				max_val = copy(daughter.value)
				# schedule daughter for promotion
				promote_node = daughter
		
		# if no promotion occurs, indicate by returning False
		if promote_node is None:
			return False
		else:
			# NODE assigned value from promoted daughter, daughter assigned node value
			node.value, promote_node.value = promote_node.value, node.value
			# return promoted daughter for subsequent promotion-demotion checks
			return promote_node
	def sorted(self, step_counter, plot_step):
		''' Returns the sorted array by promoting values from the heap.
		
			STEP_COUNTER and PLOT_STEP are to enable plotting in IPython notebook. Ignore plotting commands.
		'''
		# empty list to append sorted values
		A = []
		keep_promoting = True 					# while loop boolean
		while keep_promoting:
			# get head node
			node = self.head
			
			# condition to exit loop
			if node.value is None:
				keep_promoting = False
				continue
			
			# **plotting: ignore** -------------------------------------
			step_counter += 1
			if step_counter == plot_step: 
				self.show(node, 1)
				print('promotion from top of heap: ',A)
			# **end ignore** -------------------------------------------
					
			# copy (promote) value from head node to list
			A.append(node.value)
			
			# head node value set to None and demoted
			node.value = None
			
			# **plotting: ignore** -------------------------------------
			step_counter += 1
			if step_counter == plot_step: 
				self.show(node, 1)
				print('demote empty head node: ',A)
			# **end ignore** -------------------------------------------
			
			# loop to demote to bottom - calls DEMOTE method
			keep_demoting = True
			while keep_demoting:
				node = self.demote(node)
				if not node:
					keep_demoting = False
				
				# **plotting: ignore** --------------------------------
				else:
					step_counter += 1
					if step_counter == plot_step: 
						self.show(node, 2)
						print('demote empty head node: ',A)
				# **end ignore** --------------------------------------
		
		# return sorted list
		return A,step_counter
	def show(self, nd, case):
		''' Another rather long plotting method I don't expect you to digest.
		'''
		# count generations
		generations = 1
		still_looking = True
		current_generation = [self.head,]
			
		while still_looking:
			next_generation = []
			for node in current_generation:
				for arc in node.arcs_out:
					next_generation.append(arc.to_node)
				
			if len(next_generation)>0:
				generations +=1
			else:
				still_looking = False
			
			current_generation = copy(next_generation)
		
		f,ax = plt.subplots(1,1)
		
		f.set_size_inches(8,generations)
		
		ax.set_xlim([0,1])
		ax.set_ylim([0,generations])
		
		y = generations-0.5
		x = [0,1]
		
		gen = 1
		node = self.nodes[-1]
		keep_searching_up = True
		while keep_searching_up:
			if not node.arcs_in:
				keep_searching_up = False
			else:
				gen += 1
				node = node.arcs_in[0].from_node
		
		dx = 1./(2**gen)
		dy = 0.5
		
		still_looking=True
		current_generation = [self.head,]
		locs = {}
		generation_counter = -1
		while still_looking:
			next_generation = []
			generation_counter += 1
			xs = np.linspace(0,1,2**generation_counter+1)
			for node,x0,x1 in zip(current_generation,xs[:-1],xs[1:]):
					
				xm = 0.5*(x0+x1)
				poly = np.array([[xm-dx/2.,xm+dx/2.,xm+dx/2,xm-dx/2,xm-dx/2],[y-dy/2,y-dy/2,y+dy/2,y+dy/2,y-dy/2]]).T
				polygon = Polygon(poly, zorder=1)
				color = 'w'
				if node.name == nd.name and case == 0:
					color = [0.8,0.8,0.8]
				elif node.name == nd.name and case == 1:
					color = [1., 0.8, 0.8]
				elif node.name == nd.name and case == 2:
					color = [0.8, 1., 0.8]
				elif nd.arcs_in:
					if node.name == nd.arcs_in[0].from_node.name and case == 2:
						color = [0.8, 1., 0.8]
					
				p = PatchCollection([polygon,], color = color, zorder = 1, edgecolor = 'k', linewidth=2)
				ax.add_collection(p)
				nm = node.value
				if nm is None:
					nm = '-'
				ax.text(0.5*(x0+x1), y, '{}'.format(nm), ha='center',va='center',size=14)
				locs.update({node.name:[0.5*(x0+x1), y]})
				
				if len(node.arcs_in)>0:
					frm = node.arcs_in[0].from_node.name
					xa,ya = locs[frm]
					xb,yb = locs[node.name]
					ax.plot([xa,xb],[ya,yb],'k-', zorder = -1)
				
				# find next generation
				for arc in node.arcs_out:
					next_generation.append(arc.to_node)
				
			y = y-1
				
			if len(next_generation)>0:
				generations +=1
			else:
				still_looking = False
			
			current_generation = copy(next_generation)
			
		ax.axis('off')

			
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Classes and functions for DIJKSTRA
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SPath_Node(object):
	def __init__(self):
		self.name = None
		self.xpos = None
		self.ypos = None
		self.dist = None
		self.pre = None
		self.cset = None
		self.color = None
		self.arcs_in = []
		self.arcs_out = []
	def __repr__(self):
		return "nd: name {} xpos {} ypos {}".format(self.name, self.xpos, self.ypos)
class SPath_Arc(object):
	def __init__(self):
		self.weight=None
		self.to_node = None
		self.from_node = None
		self.color = None
	def __repr__(self):
		if self.to_node is None:
			to_nd = 'None'
		else:
			to_nd = self.to_node.name
		if self.from_node is None:
			from_nd = 'None'
		else:
			from_nd = self.from_node.name
		return "arc: {}->{}".format(from_nd, to_nd)
class SPath_NetworkError(Exception):
	'''An error to raise when violations occur.
	'''
	pass
class SPathNetwork(object):
	''' Modified form of Network class that includes x,y positions
		of each node, rather than a value. Intended for easier plotting
		of the network for use in a lecture example
	'''
	def __init__(self):
		self.nodes = []
		self.arcs = []
	 
	def __repr__(self):
		return ("ntwk(" + ''.join([len(self.nodes)*'{},'])[:-1]+")").format(*[nd.name for nd in self.nodes])
	 
	def add_node(self, name, xpos=None, ypos=None, dist=None, pre=None, cset=None, color=None):
		'''Adds a Node with NAME and XPOS, YPOS to the network.
		'''
		# check node names are unique
		network_names = [nd.name for nd in self.nodes]
		if name in network_names:
			raise NetworkError("Node with name \'{}\' already exists.".format(name))
		 
		# new node, assign values, append to list
		node = SPath_Node()
		node.name = name
		node.xpos = xpos
		node.ypos = ypos
		node.dist = dist
		node.pre = pre
		node.cset = cset
		node.color = color
		self.nodes.append(node)
		 
	def join_nodes(self, node_from, node_to, weight):
		'''Adds an Arc joining NODE_FROM to NODE_TO with WEIGHT.
		'''
		# new arc
		arc = SPath_Arc()
		arc.weight = weight
		arc.to_node = node_to
		arc.from_node = node_from
		arc.color = None
		# append to list
		self.arcs.append(arc)
		# make sure nodes know about arcs
		node_to.arcs_in.append(arc)
		node_from.arcs_out.append(arc)
 
	def read_nodes(self, filename):
		'''Read data from FILENAME and construct the network.
		'''
		# open network file
		fp = open(filename, 'r')
		 
		# get first line
		ln = fp.readline().strip()
		while ln is not '':
			# node name, xpos, ypos
			ln2 = ln.split(',')
			name = ln2[0]
			xpos = float(ln2[1])
			ypos = float(ln2[2])
			
			# create each node and location
			self.add_node(name, xpos, ypos)
			
			# get next line
			ln = fp.readline().strip()

	def read_arcs(self, filename):
		# open network file
		fp = open(filename, 'r')
		 
		# get first line
		ln = fp.readline().strip()
		while ln is not '':
			# node name
			ln2 = ln.split(',')
			from_node_name = ln2[0]
			from_node = self.get_node(from_node_name)
			arcs = ln2[1:]
		
			# read arcs
			for arc in arcs:
				to_node_name, weight = arc.split(';')
				weight = int(weight)
				 
				# check if to_node defined
				self.get_node(to_node_name)
				 
				# get to node
				to_node = self.get_node(to_node_name)
				 
				# add arc
				self.join_nodes(from_node, to_node, weight)
			 
			# get next line
			ln = fp.readline().strip()
			 
		fp.close()
	 
	def get_node(self, name):
		''' Loops through the list of nodes and returns the one with NAME.
		 
			Raises NetworkError if node does not exist.
		'''
		for node in self.nodes:
			if node.name == name:
				return node
		 
		raise NetworkError('Node \'{}\' does not exist.'.format(name))
		
def spath(network, iteration, stepno):
	''' Find shortest path between source and destination nodes in network.
		Hard-coded for in-lecture demonstration
	'''

	# first node is same regardless of iteration or step
	network.nodes[0].dist = 0.
	network.nodes[0].pre = 'None'
	network.nodes[0].cset = 'visited'
	network.nodes[0].color = 'yellow'

	# initial state of other nodes
	for nd in network.nodes[1:]:
		nd.dist = float("Inf")
		nd.pre = 'None'
		nd.cset = 'unvisited'
		nd.color = 'white'
	
	# set initial color of destination node
	network.nodes[-1].color = 'lightblue'

	# initial state of arcs
	for arc in network.arcs[:]:
		arc.color = 'white'

	# update node properties based on iteration and step
	
	# zeroth iteration
	if iteration == 0 and stepno == 2:
		network.arcs[0].color = 'yellow'
		network.arcs[1].color = 'yellow'

	if iteration == 0 and stepno == 3:
		network.arcs[0].color = 'lightgreen'
		network.arcs[1].color = 'lightgreen'
		
	if (iteration == 0 and stepno == 3) or iteration > 0:
		network.nodes[1].dist = 2
		network.nodes[1].pre = 'A'
		network.nodes[2].dist = 4
		network.nodes[2].pre = 'A'


	# iteration 1
	if iteration > 0:
		network.nodes[1].cset = 'visited'
		network.nodes[1].color = 'yellow'
		network.arcs[0].color = 'yellow'
		
	if iteration == 1 and stepno == 2:
		network.arcs[2].color = 'yellow'
		network.arcs[3].color = 'yellow'	  

	if iteration == 1 and stepno == 3:
		network.arcs[2].color = 'lightgreen'
		network.arcs[3].color = 'lightgreen'

	if (iteration == 1 and stepno == 3) or iteration > 1:
		network.nodes[2].dist = 3
		network.nodes[2].pre = 'B'
		network.nodes[3].dist = 6
		network.nodes[3].pre = 'B'

		
	# iteration number 2
	if iteration > 1:
		network.nodes[2].cset = 'visited'
		network.nodes[2].color = 'yellow'
		network.arcs[2].color = 'yellow'
		
	if iteration == 2 and stepno == 2:
		network.arcs[4].color = 'yellow'
		network.arcs[5].color = 'yellow'	  

	if iteration == 2 and stepno == 3:
		network.arcs[4].color = 'lightgreen'
		network.arcs[5].color = 'lightgreen'

	if (iteration == 2 and stepno == 3) or iteration > 2:
		network.nodes[3].dist = 5
		network.nodes[3].pre = 'C'
		network.nodes[4].dist = 4
		network.nodes[4].pre = 'C'

		
	# iteration number 3
	if iteration > 2:
		network.nodes[4].cset = 'visited'
		network.nodes[4].color = 'yellow'
		network.arcs[5].color = 'yellow'
		
	if iteration == 3 and stepno == 2:
		network.arcs[8].color = 'yellow'	  

	if iteration == 3 and stepno == 3:
		network.arcs[8].color = 'lightgreen'

	if (iteration == 3 and stepno == 3) or iteration > 3:
		network.nodes[5].dist = 7
		network.nodes[5].pre = 'E'

	# iteration number 4
	if iteration > 3:
		network.nodes[3].cset = 'visited'
		network.nodes[3].color = 'yellow'
		network.nodes[4].color = 'white'
		network.arcs[4].color = 'yellow'
		network.arcs[5].color = 'white'
		
	if iteration == 4 and stepno == 2:
		network.arcs[6].color = 'yellow'
		network.arcs[7].color = 'yellow'

	# iteration number 5
	if iteration > 4:
		network.nodes[3].color = 'white'
		network.nodes[4].color = 'yellow'
		network.nodes[5].cset = 'visited'
		network.nodes[5].color = 'yellow'
		network.arcs[4].color = 'white'
		network.arcs[8].color = 'yellow'
		network.arcs[5].color = 'yellow'		
def spath_demo(iteration, stepno):

	# set the files to read network data from
	network = SPathNetwork()
	network.read_nodes('spath_nodes.txt')
	network.read_arcs('spath_arcs.txt')
	source_name = 'A'
	destination_name= 'F'
	
	# perform algorithm for iterations
	spath(network, iteration, stepno)
	
	# set up basic plot structure
	#fig, (ax1, ax2) = plt.subplots(2,1)
	#fig.set_size_inches(20,12)
	fig = plt.figure(figsize=(20,10))
	ax1 = plt.axes([0.1,0.4,0.8,0.65])
	ax2 = plt.axes([0.1,0.1,0.8,0.35])

	# box style for node overlays
	nd_bbox_default = dict(boxstyle='round',facecolor='white',alpha=1.0)
	nd_bbox_source = dict(boxstyle='round',facecolor='yellow',alpha=1.0)
	nd_bbox_dest = dict(boxstyle='round',facecolor='lightblue',alpha=1.0)
	
	# box styles for arc overlays
	arc_bbox_default = dict(facecolor='white',alpha=0.99)
	arc_bbox_connected = dict(facecolor='yellow',alpha=0.99)
	arc_bbox_updated = dict(facecolor='lightgreen',alpha=0.99)

	# count number of nodes in network, use to assign table size
	num_nd = 0
	for nd in network.nodes:
		num_nd = num_nd+1
	table_data = np.chararray((num_nd,4),itemsize=10)

	# some initialisation of variables
	col_label = ('Node','Distance','Predecessor', 'Set')
	i = 0
	
	for nd in network.nodes:

		# choose correct colour box for node
		if nd.color == 'yellow':
			nd_bbox = copy(nd_bbox_source)
		elif nd.color == 'lightblue':
			nd_bbox = copy(nd_bbox_dest)
		elif nd.color == 'white':
			nd_bbox = copy(nd_bbox_default)	
		
		# plot each arc in network
		for arc in nd.arcs_out[:]:
			
			# set start/end location of arc line
			tmpx = [arc.from_node.xpos,arc.to_node.xpos]
			tmpy = [arc.from_node.ypos,arc.to_node.ypos]
			dx = (arc.to_node.xpos - arc.from_node.xpos)*0.9
			dy = (arc.to_node.ypos - arc.from_node.ypos)*0.9
			midx = 0.5*(arc.from_node.xpos+arc.to_node.xpos)
			midy = 0.5*(arc.from_node.ypos+arc.to_node.ypos)
			
			# plot the arc line
#			 ax1.plot(tmpx,tmpy,'k-')
			ax1.arrow(arc.from_node.xpos,arc.from_node.ypos,dx,dy, width=0.005, fill=False)
			
			# choose correct colour box for arc
			if arc.color == 'white':
				arc_bbox = copy(arc_bbox_default)
			elif arc.color == 'yellow':
				arc_bbox = copy(arc_bbox_connected)
			elif arc.color == 'lightgreen':
				arc_bbox = copy(arc_bbox_updated)
			
			# plot arc text box
			ax1.text(midx, midy, '{}'.format(arc.weight), ha='center', va='center', bbox=arc_bbox, fontsize=14)			   
		
		# plot node text box
		ax1.text(nd.xpos, nd.ypos, nd.name, ha='center', va='center', bbox=nd_bbox, fontsize=20)
		
		# create new row of table data
		table_data[i,:] = [nd.name, '%1.0f'%nd.dist, nd.pre, nd.cset]
		i=i+1

	# tidy up display of text table
	table_data = table_data.decode('UTF-8')

	# display text table as plot
	tb = ax2.table(cellText=table_data,cellColours=None,
				   colWidths=[0.12,0.1,0.12,0.07],
				   colLabels=col_label,loc='center')
	tb.auto_set_font_size(False)
	tb.set_fontsize(18)
	tb.scale(2, 2)
	
	# some final axis adjustments
	ax1.axis('off')
	ax2.axis('tight')
	ax2.axis('off')
	plt.show()
	
def dijkstra_example():
	int1 = IntSlider(min=0,max=5,step=1,value=0, description='iteration')
	int2 = IntSlider(min=1,max=3,step=1,value=1, description='step number')
	
	def int_change(change):
		int2.value=1
	int1.observe(int_change)
	
	controls = {'iteration':int1,'stepno':int2} 
	return VBox([HBox([int1,int2]), interactive_output(spath_demo, controls)])
	
	
	
	
	
	