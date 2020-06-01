# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""

import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve

def FEM1D(node,elem1D,pde,bdStruct):
    
    # ----------------- preparation -------------------------
    N = node.size; nel = elem1D.shape[0]; Ndof = 2;
    f = pde.f; Du = pde.Du; g_D = pde.g_D; para = pde.para;
    Neumann = bdStruct.Neumann; Dirichlet = bdStruct.Dirichlet;
    
    
    # -------------- Sparse assembling indices --------------
    nnz = nel*Ndof**2;
    ii = np.array( range(nnz) ); jj = np.array( range(nnz) );  
    id = 0;
    for i in range(Ndof):
        for j in range(Ndof):
           ii[id:id+nel] = elem1D[:,i]; 
           jj[id:id+nel] = elem1D[:,j]; 
           id = id + nel; 
    
    # ----------------- Stiffness matrix -----------------
    a = para[0]; b = para[1]; c = para[2];
    # All element matrices
    h = np.diff(node);
    k11 = a/h+b/2*(-1)+c*h/6*2;
    k12 = a/h*(-1)+b/2+c*h/6;
    k21 = a/h*(-1)+b/2*(-1)+c*h/6;
    k22 = a/h+b/2+c*h/6*2;
    K = np.hstack((k11,k12,k21,k22)); 
    # stiffness matrix             
    kk = csc_matrix((K,(ii,jj)),shape=(N,N)); # sparse in matlab  

    # ------------------ Load vector ------------------
    x1 = node[0:N-1]; x2 = node[1:N];
    xc = (x1+x2)/2;
    F1 = f(xc)*h/2; F2 = F1; 
    F = np.hstack((F1,F2));
    subs = np.hstack((elem1D[...,0],elem1D[...,1])); 
    ff = np.bincount(subs, weights=F, minlength=N);  # accumarray in matlab
    
    # ------- Neumann boundary conditions ----------
    if Neumann != [] :
       nvec = 1;  
       ind =  (elem1D[:,0] == Neumann);    
       if ind.any(): nvec = -1; 
       Dnu = Du(node[Neumann]) * nvec;
       ff[Neumann] = ff[Neumann] + a*Dnu;
       
    # ------- Dirichlet boundary conditions ----------
    uh = np.zeros((N,)); 
    bdDof = Dirichlet; 
    freeDof = np.array( range(N), dtype = np.int); 
    freeDof = np.delete(freeDof, bdDof);
    uh[bdDof] = g_D(node[Dirichlet]);
    ff = ff - kk*uh;

    uh[freeDof] = spsolve(kk[ np.ix_(freeDof,freeDof) ],ff[freeDof]);
    
    return uh;