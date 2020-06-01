# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""

import numpy as np
from tool.auxdata import auxstructure

def squaremesh(square,h1,h2):
    #Squaremesh uniform mesh of a square
    #
    # square = [a1,b1,a2,b2] for rectangle [a1,b1]*[a2,b2]
    
    # ----------- Generate nodes ---------
    a1 = square[0]; b1 = square[1]; a2 = square[2]; b2 = square[3];
    x,y = np.mgrid[a1:b1+h1:h1, a2:b2+h2:h2];  # 对应 matlab 的 ndgrid
        
    x1 = x.ravel("F"); y1 = y.ravel("F");
    node = np.vstack( (x1,y1) ); node = node.transpose();
    
    #-------- Generate elements ---------
    nx = x.shape[0]; ny = y.shape[1]; # number of columns and rows
    
    # 7 --- 8 --- 9               # 6 --- 7 --- 8
    # |     |     |               # |     |     |
    # 4 --- 5 --- 6      --->     # 3 --- 4 --- 5  
    # |     |     |               # |     |     |
    # 1 --- 2 --- 3               # 0 --- 1 --- 2
    
    # 4 k+nx --- k+1+nx 3             # 3 k+nx --- k+1+nx 2
    #    |        |                   #    |        |
    # 1  k  ---  k+1    2             # 0  k  ---  k+1    1
    
    # indices of k
    N = node.shape[0];
    k = np.arange(0,N-1-nx+1);   cut = nx*np.arange(1,ny-1+1) - 1;  
    k = np.delete(k,cut);
    
    elem = np.hstack( ( [k+1,k+1+nx,k], [k+nx,k,k+1+nx] ) );
    elem = elem.transpose();

    return node, elem


def uniformrefine(node,elem):

    # auxiliary mesh data
    auxT = auxstructure(node,elem);
    edge = auxT.edge; elem2edge = auxT.elem2edge;
    N = node.shape[0]; NT = elem.shape[0]; #NE = edge.shape[0];
    
    # Add new nodes: middle points of all edges
    node1 = np.vstack( (node,  (node[edge[:,0],:] + node[edge[:,1],:])/2) );
    
    # Refine each triangle into four triangles as follows
    # 3                      # 2
    # | \                    # | \
    # 5- 4                   # 4- 3
    # |\ |\                  # |\ |\
    # 1- 6- 2                # 0- 5- 1
    
    t = np.arange(NT);  p = np.zeros( (NT,6) );
    p[:,0:3] = elem;
    p[:,3:6] = elem2edge + N;
    
    elem1 = np.zeros( (4*NT,3), dtype = np.int );
    elem1[t,:] = np.hstack( (p[t,0,np.newaxis], p[t,5,np.newaxis], p[t,4,np.newaxis]) );
    elem1[NT:2*NT,:] = np.hstack( (p[t,5,np.newaxis], p[t,1,np.newaxis], p[t,3,np.newaxis]) );
    elem1[2*NT:3*NT,:] = np.hstack( (p[t,4,np.newaxis], p[t,3,np.newaxis], p[t,2,np.newaxis]) );
    elem1[3*NT:4*NT,:] = np.hstack( (p[t,3,np.newaxis], p[t,4,np.newaxis], p[t,5,np.newaxis]) );
    
    return node1, elem1;