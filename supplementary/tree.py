'''
Created on Aug 9, 2017
@author: arnon
source: https://github.com/Acrisel/references/blob/master/osutils/ptree.py
'''

import os
from fnmatch import fnmatch

def realname(path, root=None):
    if root is not None:
        path=os.path.join(root, path)
    result=os.path.basename(path)
    if os.path.islink(path):
        realpath=os.readlink(path)
        result= '%s -> %s' % (os.path.basename(path), realpath)
    return result

def tree(startpath='.', depth=-1, skip = None):
    prefix=0
    if startpath != '/':
        if startpath.endswith('/'): startpath=startpath[:-1]
        prefix=len(startpath)
    for root, dirs, files in os.walk(startpath):
        if fnmatch(root,skip):
            continue
      	# remove dirs with skip string
        if skip is not None:
        	dirs = [dir for dir in dirs if not fnmatch(dir,skip)]
        level = root[prefix:].count(os.sep)
        if depth >-1 and level > depth: continue
        indent=subindent =''
        if level > 0:
            indent = '|   ' * (level-1) + '|-- '
        subindent = '|   ' * (level) + '|-- '
        print('{}{}/'.format(indent, realname(root)))
        # print dir only if symbolic link; otherwise, will be printed as root
        for d in dirs:
            if os.path.islink(os.path.join(root, d)):
                print('{}{}'.format(subindent, realname(d, root=root)))
        for f in files:
            print('{}{}'.format(subindent, realname(f, root=root)))