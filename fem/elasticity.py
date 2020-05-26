# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

# fem
from fem.base import gradbasis
# tool
from tool.GaussQuad import quadpts

def elasticity1(node,elem,pde,bdStruct):

   
    N = node.shape[0]; NT = elem.shape[0]; Ndof = 3;
    lambda_ = pde['lambda_']; mu = pde['mu']; f = pde['f'];
    
    # -------------- Compute (Dibase,Djbase) --------------------
    Dphi, area = gradbasis(node,elem);
    Dbase = [1,2,3,4];  # initialization
    s = 0;
    for i in range(2):
        for j in range(2):
            k11 = Dphi[0][:,i]*Dphi[0][:,j]*area;
            k12 = Dphi[0][:,i]*Dphi[1][:,j]*area;
            k13 = Dphi[0][:,i]*Dphi[2][:,j]*area;
            k21 = Dphi[1][:,i]*Dphi[0][:,j]*area;
            k22 = Dphi[1][:,i]*Dphi[1][:,j]*area;
            k23 = Dphi[1][:,i]*Dphi[2][:,j]*area;
            k31 = Dphi[2][:,i]*Dphi[0][:,j]*area;
            k32 = Dphi[2][:,i]*Dphi[1][:,j]*area;
            k33 = Dphi[2][:,i]*Dphi[2][:,j]*area;
            K = np.hstack((k11,k12,k13,k21,k22,k23,k31,k32,k33)); # straigthen
            Dbase[s] = K; s = s+1;
            
    # -------------- Sparse assembling indices --------------
    nnz = NT*Ndof**2;
    ii = np.array( range(nnz), dtype = np.int); 
    jj = np.array( range(nnz), dtype = np.int);  
    id = 0;
    for i in range(Ndof):
        for j in range(Ndof):
            ii[id:id+NT] = elem[:,i]; 
            jj[id:id+NT] = elem[:,j]; 
            id = id + NT; 
    
    ii11 = ii;   jj11 = jj;  ii12 = ii;   jj12 = jj+N;
    ii21 = ii+N; jj21 = jj;  ii22 = ii+N; jj22 = jj+N;
    
    # ----------- Assemble stiffness matrix -----------
    ss11 = (lambda_+2*mu)*Dbase[0] + mu*Dbase[3];
    ss12 = lambda_*Dbase[1] + mu*Dbase[2];
    ss21 = lambda_*Dbase[2] + mu*Dbase[1];
    ss22 = (lambda_+2*mu)*Dbase[3] + mu*Dbase[0];
    ii = np.hstack((ii11, ii12, ii21, ii22));
    jj = np.hstack((jj11, jj12, jj21, jj22));
    ss = np.hstack((ss11, ss12, ss21, ss22));
    kk = csc_matrix( (ss,(ii,jj)), shape = (2*N,2*N), dtype = np.float);
    
    # ------------- Assemble load vector ------------
    # Gauss quadrature rule
    [_lambda,weight] = quadpts(2);
    F1 = np.zeros((NT,3), dtype = np.float);
    F2 = np.zeros((NT,3), dtype = np.float);
    for p in range(len(weight)):
        pxy = _lambda[p,0]*node[elem[:,0],:] \
            + _lambda[p,1]*node[elem[:,1],:] \
            + _lambda[p,2]*node[elem[:,2],:];
        fxy = f(pxy); # fxy = [f1xy, f2xy]
        F1 = F1 + weight[p]*fxy[:,0,np.newaxis]*_lambda[p,:];
        F2 = F2 + weight[p]*fxy[:,1,np.newaxis]*_lambda[p,:];
    F1 = area[:,np.newaxis]*F1;
    F2 = area[:,np.newaxis]*F2;
    
    ff = np.bincount( np.hstack( ( elem.ravel("F"),elem.ravel("F")+N ) ), \
                     weights = np.hstack( ( F1.ravel("F"),F2.ravel("F") ) ), \
                     minlength = 2*N);
    
    # ------------ Neumann boundary condition ---------------- 
    elemN = bdStruct['elemN'];
    if elemN.any():
       g_N = pde['g_N'];
       z1 = node[elemN[:,0],:]; z2 = node[elemN[:,1],:];
       e = z1-z2; # e = z2-z1
       ne = np.array( [ -e[:,1], e[:,0] ] ).transpose(); # scaled ne
       Sig1 = g_N(z1); Sig2 = g_N(z2);
       F11 = np.sum( ne*Sig1[:,[0,2]], axis=1 )/2;
       F12 = np.sum( ne*Sig2[:,[0,2]], axis=1 )/2;
       F21 = np.sum( ne*Sig1[:,[2,1]], axis=1 )/2;
       F22 = np.sum( ne*Sig2[:,[2,1]], axis=1 )/2;
       FN = np.hstack( (F11,F12,F21,F22) );
       ff = ff + np.bincount( np.hstack( (elemN.ravel("F"), elemN.ravel("F")+N) ) ,\
                              FN,  minlength = 2*N);
           
    # ------------ Dirichlet boundary condition ----------------
    g_D = pde['g_D'];  eD = bdStruct['eD'];
    id = np.hstack((eD,eD+N));
    isBdNode = np.array([False]*2*N); isBdNode[id] = True;
    bdDof = isBdNode; freeDof = ~isBdNode;
    pD = node[eD,:];
    uh = np.zeros((2*N,)); 
    uD = g_D(pD); uh[bdDof] = uD.ravel("F");
    ff = ff - kk@uh;   
    
    # ---------------------- Solver -------------------------
    uh[freeDof] = spsolve( kk[ np.ix_(freeDof,freeDof) ],ff[freeDof] );
    
    return uh;   


def elasticity2(node,elem,pde,bdStruct):

   
    N = node.shape[0]; NT = elem.shape[0]; Ndof = 3;
    lambda_ = pde['lambda_']; mu = pde['mu']; f = pde['f'];
    
    # -------------- Compute (Dibase,Djbase) --------------------
    Dphi, area = gradbasis(node,elem);
    Dbase = [1,2,3,4];  # initialization
    s = 0;
    for i in range(2):
        for j in range(2):
            k11 = Dphi[0][:,i]*Dphi[0][:,j]*area;
            k12 = Dphi[0][:,i]*Dphi[1][:,j]*area;
            k13 = Dphi[0][:,i]*Dphi[2][:,j]*area;
            k21 = Dphi[1][:,i]*Dphi[0][:,j]*area;
            k22 = Dphi[1][:,i]*Dphi[1][:,j]*area;
            k23 = Dphi[1][:,i]*Dphi[2][:,j]*area;
            k31 = Dphi[2][:,i]*Dphi[0][:,j]*area;
            k32 = Dphi[2][:,i]*Dphi[1][:,j]*area;
            k33 = Dphi[2][:,i]*Dphi[2][:,j]*area;
            K = np.hstack((k11,k12,k13,k21,k22,k23,k31,k32,k33)); # straigthen
            Dbase[s] = K; s = s+1;
            
    # -------------- Sparse assembling indices --------------
    nnz = NT*Ndof**2;
    ii = np.array( range(nnz), dtype = np.int); 
    jj = np.array( range(nnz), dtype = np.int);  
    id = 0;
    for i in range(Ndof):
        for j in range(Ndof):
            ii[id:id+NT] = elem[:,i]; 
            jj[id:id+NT] = elem[:,j]; 
            id = id + NT; 
    
    ii11 = ii;   jj11 = jj;  ii12 = ii;   jj12 = jj+N;
    ii21 = ii+N; jj21 = jj;  ii22 = ii+N; jj22 = jj+N;
    
    # ----------- Assemble stiffness matrix -----------
    # (Eij(u):Eij(v))
    ss11 = Dbase[0] + 0.5*Dbase[3];
    ss12 = 0.5*Dbase[2];
    ss21 = 0.5*Dbase[1];
    ss22 = Dbase[3] + 0.5*Dbase[0];
    ii = np.hstack((ii11, ii12, ii21, ii22));
    jj = np.hstack((jj11, jj12, jj21, jj22));
    ss = np.hstack((ss11, ss12, ss21, ss22));
    A = csc_matrix( (ss,(ii,jj)), shape = (2*N,2*N), dtype = np.float);
    A = 2*mu*A;
    
    # (div u,div v)
    ss11 = Dbase[0];            ss12 = Dbase[1];
    ss21 = Dbase[2];            ss22 = Dbase[3];
    ss = np.hstack((ss11, ss12, ss21, ss22));
    B = csc_matrix( (ss,(ii,jj)), shape = (2*N,2*N), dtype = np.float);
    B = lambda_*B;
    
    # stiffness matrix
    kk = A + B;
    
    # ------------- Assemble load vector ------------
    # Gauss quadrature rule
    [_lambda,weight] = quadpts(2);
    F1 = np.zeros((NT,3), dtype = np.float);
    F2 = np.zeros((NT,3), dtype = np.float);
    for p in range(len(weight)):
        pxy = _lambda[p,0]*node[elem[:,0],:] \
            + _lambda[p,1]*node[elem[:,1],:] \
            + _lambda[p,2]*node[elem[:,2],:];
        fxy = f(pxy); # fxy = [f1xy, f2xy]
        F1 = F1 + weight[p]*fxy[:,0,np.newaxis]*_lambda[p,:];
        F2 = F2 + weight[p]*fxy[:,1,np.newaxis]*_lambda[p,:];
    F1 = area[:,np.newaxis]*F1;
    F2 = area[:,np.newaxis]*F2;
    
    ff = np.bincount( np.hstack( ( elem.ravel("F"),elem.ravel("F")+N ) ), \
                     weights = np.hstack( ( F1.ravel("F"),F2.ravel("F") ) ), \
                     minlength = 2*N);
    
    # ------------ Neumann boundary condition ---------------- 
    elemN = bdStruct['elemN'];
    if elemN.any():
       g_N = pde['g_N'];
       z1 = node[elemN[:,0],:]; z2 = node[elemN[:,1],:];
       e = z1-z2; # e = z2-z1
       ne = np.array( [ -e[:,1], e[:,0] ] ).transpose(); # scaled ne
       Sig1 = g_N(z1); Sig2 = g_N(z2);
       F11 = np.sum( ne*Sig1[:,[0,2]], axis=1 )/2;
       F12 = np.sum( ne*Sig2[:,[0,2]], axis=1 )/2;
       F21 = np.sum( ne*Sig1[:,[2,1]], axis=1 )/2;
       F22 = np.sum( ne*Sig2[:,[2,1]], axis=1 )/2;
       FN = np.hstack( (F11,F12,F21,F22) );
       ff = ff + np.bincount( np.hstack( (elemN.ravel("F"), elemN.ravel("F")+N) ) ,\
                              FN,  minlength = 2*N);
           
    # ------------ Dirichlet boundary condition ----------------
    g_D = pde['g_D'];  eD = bdStruct['eD'];
    id = np.hstack((eD,eD+N));
    isBdNode = np.array([False]*2*N); isBdNode[id] = True;
    bdDof = isBdNode; freeDof = ~isBdNode;
    pD = node[eD,:];
    uh = np.zeros((2*N,)); 
    uD = g_D(pD); uh[bdDof] = uD.ravel("F");
    ff = ff - kk@uh;   
    
    # ---------------------- Solver -------------------------
    uh[freeDof] = spsolve( kk[ np.ix_(freeDof,freeDof) ],ff[freeDof] );
    
    return uh;   


def elasticity3(node,elem,pde,bdStruct):

   
    N = node.shape[0]; NT = elem.shape[0]; Ndof = 3;
    lambda_ = pde['lambda_']; mu = pde['mu']; f = pde['f'];
    
    # -------------- Compute (Dibase,Djbase) --------------------
    Dphi, area = gradbasis(node,elem);
    Dbase = [1,2,3,4];  # initialization
    s = 0;
    for i in range(2):
        for j in range(2):
            k11 = Dphi[0][:,i]*Dphi[0][:,j]*area;
            k12 = Dphi[0][:,i]*Dphi[1][:,j]*area;
            k13 = Dphi[0][:,i]*Dphi[2][:,j]*area;
            k21 = Dphi[1][:,i]*Dphi[0][:,j]*area;
            k22 = Dphi[1][:,i]*Dphi[1][:,j]*area;
            k23 = Dphi[1][:,i]*Dphi[2][:,j]*area;
            k31 = Dphi[2][:,i]*Dphi[0][:,j]*area;
            k32 = Dphi[2][:,i]*Dphi[1][:,j]*area;
            k33 = Dphi[2][:,i]*Dphi[2][:,j]*area;
            K = np.hstack((k11,k12,k13,k21,k22,k23,k31,k32,k33)); # straigthen
            Dbase[s] = K; s = s+1;
            
    # -------------- Sparse assembling indices --------------
    nnz = NT*Ndof**2;
    ii = np.array( range(nnz), dtype = np.int); 
    jj = np.array( range(nnz), dtype = np.int);  
    id = 0;
    for i in range(Ndof):
        for j in range(Ndof):
            ii[id:id+NT] = elem[:,i]; 
            jj[id:id+NT] = elem[:,j]; 
            id = id + NT; 
    
    ii11 = ii;   jj11 = jj;  ii12 = ii;   jj12 = jj+N;
    ii21 = ii+N; jj21 = jj;  ii22 = ii+N; jj22 = jj+N;
    
    # ----------- Assemble stiffness matrix -----------
    # (grad u,grad v)
    ss11 = Dbase[0] + Dbase[3];  ss22 = ss11;
    ii = np.hstack((ii11, ii22));
    jj = np.hstack((jj11, jj22));
    ss = np.hstack((ss11, ss22));
    A = csc_matrix( (ss,(ii,jj)), shape = (2*N,2*N), dtype = np.float);
    A = mu*A;
    
    # (div u,div v)
    ss11 = Dbase[0];            ss12 = Dbase[1];
    ss21 = Dbase[2];            ss22 = Dbase[3];
    ii = np.hstack((ii11, ii12, ii21, ii22));
    jj = np.hstack((jj11, jj12, jj21, jj22));
    ss = np.hstack((ss11, ss12, ss21, ss22));
    B = csc_matrix( (ss,(ii,jj)), shape = (2*N,2*N), dtype = np.float);
    B = (lambda_+mu)*B;
    
    # stiffness matrix
    kk = A + B;
    
    # ------------- Assemble load vector ------------
    # Gauss quadrature rule
    [_lambda,weight] = quadpts(2);
    F1 = np.zeros((NT,3), dtype = np.float);
    F2 = np.zeros((NT,3), dtype = np.float);
    for p in range(len(weight)):
        pxy = _lambda[p,0]*node[elem[:,0],:] \
            + _lambda[p,1]*node[elem[:,1],:] \
            + _lambda[p,2]*node[elem[:,2],:];
        fxy = f(pxy); # fxy = [f1xy, f2xy]
        F1 = F1 + weight[p]*fxy[:,0,np.newaxis]*_lambda[p,:];
        F2 = F2 + weight[p]*fxy[:,1,np.newaxis]*_lambda[p,:];
    F1 = area[:,np.newaxis]*F1;
    F2 = area[:,np.newaxis]*F2;
    
    ff = np.bincount( np.hstack( ( elem.ravel("F"),elem.ravel("F")+N ) ), \
                     weights = np.hstack( ( F1.ravel("F"),F2.ravel("F") ) ), \
                     minlength = 2*N);
    
    # ------------ Neumann boundary condition ---------------- 
    elemN = bdStruct['elemN'];
    if elemN.any():
       g_N = pde['g_N'];
       z1 = node[elemN[:,0],:]; z2 = node[elemN[:,1],:];
       e = z1-z2; # e = z2-z1
       ne = np.array( [ -e[:,1], e[:,0] ] ).transpose(); # scaled ne
       Sig1 = g_N(z1); Sig2 = g_N(z2);
       F11 = np.sum( ne*Sig1[:,[0,2]], axis=1 )/2;
       F12 = np.sum( ne*Sig2[:,[0,2]], axis=1 )/2;
       F21 = np.sum( ne*Sig1[:,[2,1]], axis=1 )/2;
       F22 = np.sum( ne*Sig2[:,[2,1]], axis=1 )/2;
       FN = np.hstack( (F11,F12,F21,F22) );
       ff = ff + np.bincount( np.hstack( (elemN.ravel("F"), elemN.ravel("F")+N) ) ,\
                              FN,  minlength = 2*N);
           
    # ------------ Dirichlet boundary condition ----------------
    g_D = pde['g_D'];  eD = bdStruct['eD'];
    id = np.hstack((eD,eD+N));
    isBdNode = np.array([False]*2*N); isBdNode[id] = True;
    bdDof = isBdNode; freeDof = ~isBdNode;
    pD = node[eD,:];
    uh = np.zeros((2*N,)); 
    uD = g_D(pD); uh[bdDof] = uD.ravel("F");
    ff = ff - kk@uh;   
    
    # ---------------------- Solver -------------------------
    uh[freeDof] = spsolve( kk[ np.ix_(freeDof,freeDof) ],ff[freeDof] );
    
    return uh;   


