# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""

import numpy as np
from matlab.basic import m_unique,m_sparse,m_find
from scipy.spatial.distance import pdist

def auxgeometry(node,elem):
    
    NT = elem.shape[0];   
    centroid = np.zeros((NT,2)); area = np.zeros((NT,)); diameter = np.zeros((NT,));
    for iel in range(NT):
        index = elem[iel,:];
        verts = node[index,:]; verts1 = verts[[1,2,0],:];
        area_components = verts[:,0]*verts1[:,1] - verts1[:,0]*verts[:,1];
        area_components = area_components.reshape((-1,1));
        ar = 0.5*abs(sum(area_components));
        area[iel] = ar;
        centroid[iel,:] = sum((verts+verts1)*area_components)/(6*ar);
        diameter[iel] = max(pdist(verts));
    
    aux = {'node':node, 'elem':elem, 'centroid':centroid, \
           'area':area, 'diameter':diameter};
        
    return aux;



def auxstructure(node,elem):
    
    NT = elem.shape[0];    
    # totalEdge
    allEdge = np.vstack( ( elem[:,[1,2]], elem[:,[2,0]], elem[:,[0,1]]  ) );
    totalEdge = np.sort(allEdge);
    
    # --------- elem2edge: elementwise edges ---------------
    _, i1, totalJ = m_unique(totalEdge);
    elem2edge = totalJ.reshape( (3,NT) );
    elem2edge = elem2edge.transpose();
    
    # -------- edge, bdEdge --------
    N = totalEdge.shape[0];
    ii = totalEdge[:,1];  jj = totalEdge[:,0];  ss = np.ones(N);
    A = m_sparse(ii,jj,ss,N,N);
    i, j, s = m_find(A);
    edge = np.array([j,i]); edge = edge.transpose();
    bdEdge = edge[s==1,:]; # not counterclockwise
    
    # ------- edge2elem --------
    totalJelem = np.tile(np.arange(NT),3); # np.tile 等价于 matlab 的 repmat
    
    _,i2,_ = m_unique(totalJ[::-1])
    i2 = len(totalEdge)-1-i2
    edge2elem = totalJelem[np.array([i1,i2])].transpose();
    
    
    # --------- neighbor ---------
    NE = edge.shape[0];
    ii1 = edge2elem[:,0]; jj1 = np.arange(NE); ss1 = edge2elem[:,1];
    ii2 = edge2elem[:,1]; jj2 = np.arange(NE); ss2 = edge2elem[:,0];
    label = (ii2!=ss2);
    ii2 = ii2[label]; jj2 = jj2[label]; ss2 = ss2[label];
    ii = np.hstack( (ii1,ii2) ); 
    jj = np.hstack( (jj1,jj2) );
    ss = np.hstack( (ss1,ss2) );
    neighbor = m_sparse(ii,jj,ss,NT,NE);
    
    aux = {'node':node, 'elem':elem, 'elem2edge':elem2edge, \
            'edge':edge, 'bdEdge':bdEdge, 'edge2elem':edge2elem,
            'neighbor':neighbor};
    
    return aux;

