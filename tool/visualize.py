# -*- coding: utf-8 -*-
"""
Created on Wed May 20 21:20:17 2020

@author: Terenceyuyue
"""
import matplotlib.pyplot as plt
import numpy as np
# matlab 
from matlab.basic import m_sparse, m_find

def showmesh(node,elem):
    NT = elem.shape[0];
    #plt.figure(figsize=(6,6))
    for iel in range(NT):
        x = node[elem[iel,:],0]; y = node[elem[iel,:],1];
        x = x[ [0,1,2,0] ]; y = y[ [0,1,2,0] ];
        plt.plot(x,y,'k',linewidth = 0.5);
        plt.fill(x, y, color = [0.5,0.9,0.45]);
    plt.show()
    
def findnode(node):
    N = node.shape[0];
    for i in range(N):
        x = node[i,0]; y = node[i,1];
        plt.text(x,y, str(i),fontsize = 8, fontweight = 'bold');
 

def findelem(node,elem):
    NT = elem.shape[0];
    center = np.zeros( (NT,2) );
    for iel in range(NT):
        index = elem[iel,:];
        verts = node[index,:]; verts1 = verts[[1,2,0],:];
        area_components = verts[:,0]*verts1[:,1] - verts1[:,0]*verts[:,1];
        area_components = area_components.reshape((-1,1));
        area = 0.5*abs(sum(area_components));
        center[iel,:] = sum((verts+verts1)*area_components)/(6*area);
        
    plt.plot( center[:,0],center[:,1],'o', linewidth = 1, markeredgecolor = 'k', \
             markerfacecolor = 'y', markersize = 18);
    for iel in range(NT):
        plt.text( center[iel,0]-0.015, center[iel,1]-0.01, str(iel));
        

def findedge(node,elem, *args):
    
    bdInd = args; 

    # edge matrix
    allEdge = np.vstack( ( elem[:,[1,2]], elem[:,[2,0]], elem[:,[0,1]]  ) );
    totalEdge = np.sort(allEdge);
    N = totalEdge.shape[0];
    ii = totalEdge[:,1];  jj = totalEdge[:,0];  ss = np.ones(N);
    A = m_sparse(ii,jj,ss,N,N);  
    i,j,s = m_find(A);
    edge = np.array([j,i]); edge = edge.transpose();
    
    # range
    if len(bdInd)==0:  # bdInd = ()
       rg = range(edge.shape[0]);
    else:
        rg = np.where(s==1); rg = rg[0];
    
    # edge index
    midEdge = ( node[edge[rg,0],:] + node[edge[rg,1],:]   )/2;
    plt.plot( midEdge[:,0], midEdge[:,1], 's', linewidth = 1, \
             markeredgecolor = 'k', markerfacecolor = [0.6,0.5,0.8],\
            markersize = 20);
    for i in range(len(rg)):
        plt.text( midEdge[i,0]-0.025, midEdge[i,1]-0.01, str(i), \
                 fontsize = 12, fontweight = 'bold', color = 'k');
            
            
def showsolution(node,elem,u,*args):
    #from mpl_toolkits import mplot3d
    if len(args)==0:        
       ax = plt.axes(projection='3d');
    else:
        ax = args[0];
    x = node[:,0]; y = node[:,1]; 
    ax.plot_trisurf(x, y, u, triangles = elem, cmap = 'viridis', \
                    edgecolor = 'k', linewidth=0.2)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('u');   
    plt.show()
        