# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix
# matlab
#from matlab.basic import m_sparse, m_accumarray
# tool
from tool.GaussQuad import quadpts, quadpts1
from tool.auxdata import auxstructure
# fem
from fem.base import gradbasis



def Poisson(node,elem,pde,bdStruct,*args):  
    
    if len(args) == 0: quadOrder = 3;
    if len(args) == 1: quadOrder = args[0];    
    
    N = node.shape[0];  NT = elem.shape[0];  Ndof = 3;
    f = pde['f'];
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
        #end
    #end
    
    # ---------------------- gradbasis --------------------
    Dphi, area = gradbasis(node,elem);
    
    # -------------- Stiffness matrix -------------
    K = np.zeros( (NT, Ndof**2), dtype = np.float);
    s = 0;
    for i in range(Ndof):
        for j in range(Ndof):
            K[:,s] =  np.sum(Dphi[i]*Dphi[j], axis = 1) * area;
            s = s+1; 
        #end
    #end
    kk = csc_matrix( ( K.ravel("F"), (ii,jj) ), shape = (N,N), dtype = np.float);
    
    # ------------- Load vector ------------
    _lambda, weights = quadpts(quadOrder);
    F = np.zeros( (NT,3), dtype = np.float ); # straighten
    z1 = node[elem[:,0],:];   
    z2 = node[elem[:,1],:]; 
    z3 = node[elem[:,2],:];
    for p in range( len(weights) ):
        # quadrature points in the x-y coordinate
        pxy = _lambda[p,0]*z1 + _lambda[p,1]*z2 + _lambda[p,2]*z3;
        fxy = f(pxy).reshape((-1,1)); 
        fv = fxy*_lambda[p,:];
        F += weights[p]*fv;
    #end
       
    F = area[:,np.newaxis]*F; # area.reshape((-1,1))*F;
    ff = np.bincount(elem.ravel("F"), weights = F.ravel("F"), minlength = N);
    
    # ------------ Neumann boundary conditions --------------
    elemN = bdStruct['elemN'];
    if elemN.any():
        z1 = node[elemN[:,0],:]; z2 = node[elemN[:,1],:];
        e = z1-z2; # e = z2-z1
        ne = np.array( [ -e[:,1], e[:,0] ] ).transpose(); # scaled ne
        Du = pde['Du'];
        gradu1 = Du(z1); gradu2 = Du(z2);
        F1 = np.sum(ne*gradu1,axis=1)/2;  F2 = np.sum(ne*gradu2,axis=1)/2; 
        FN = np.zeros((len(F1),2)); FN[:,0] = F1; FN[:,1] = F2;
        ff = ff + np.bincount(elemN.ravel("F"), FN.ravel("F"), minlength=N);
    #end
    
    # --------- Dirichlet boundary conditions ---------------
    eD = bdStruct['eD'];  g_D = pde['g_D'];
    isBdNode = np.array( [False]*N );  isBdNode[eD] = True;
    bdNode = isBdNode;  freeNode = ~bdNode;  # np.array 的逻辑数组可以这样取反，但列表不可以
    pD = node[bdNode,:];
    uh = np.zeros( (N,), dtype = np.float );   uh[bdNode] = g_D(pD); 
    ff = ff - kk@uh; 
    
    # ---------------------- Solver -------------------------
    uh[freeNode] = spsolve( kk[ np.ix_(freeNode,freeNode) ],ff[freeNode] );
    
    return uh;


def PoissonP2(node,elem,pde,bdStruct,*args):  
    
    if len(args) == 0: quadOrder = 4;
    if len(args) == 1: quadOrder = args[0];    
    
    # -------------- Sparse assembling indices --------------
    # auxstructure
    auxT = auxstructure(node, elem);
    edge = auxT['edge']; 
    elem2edge = auxT['elem2edge'];
    # numbers
    N = node.shape[0]; NT = elem.shape[0]; NE = edge.shape[0];
    NNdof = N + NE; Ndof = 6; #global and local d.o.f. numbers
    # elem2dof
    elem2 = np.hstack( (elem, elem2edge+N) );
    # ii,jj    
    nnz = NT*Ndof**2;
    ii = np.array( range(nnz), dtype = np.int); 
    jj = np.array( range(nnz), dtype = np.int);  
    id = 0;
    for i in range(Ndof):
        for j in range(Ndof):
            ii[id:id+NT] = elem2[:,i]; 
            jj[id:id+NT] = elem2[:,j]; 
            id = id + NT; 
        #end
    #end
    
    # -------------- Stiffness matrix -------------
    # _lambda and Dlambda
    _lambda, weight = quadpts(quadOrder); nG = len(weight);
    Dlambda,area = gradbasis(node,elem);
    # stiffness matrix
    K = np.zeros((NT,Ndof**2)); # straighten
    for p in range(nG):
        Dphip = [i for i in range(nG)]; # initialization
        # Dphi at quadrature points
        Dphip[0] = (4*_lambda[p,0]-1)*Dlambda[0];
        Dphip[1] = (4*_lambda[p,1]-1)*Dlambda[1];
        Dphip[2] = (4*_lambda[p,2]-1)*Dlambda[2];
        Dphip[3] = 4*(_lambda[p,1]*Dlambda[2]+_lambda[p,2]*Dlambda[1]);
        Dphip[4] = 4*(_lambda[p,2]*Dlambda[0]+_lambda[p,0]*Dlambda[2]);
        Dphip[5] = 4*(_lambda[p,0]*Dlambda[1]+_lambda[p,1]*Dlambda[0]);
        s = 0;
        for i in range(Ndof):
            for j in range(Ndof):
                K[:,s] +=  weight[p]*np.sum(Dphip[i]*Dphip[j],axis = 1) * area;
                s = s+1;
            #end
        #end
    #end
    kk = csc_matrix( ( K.ravel("F"), (ii,jj) ), shape = (NNdof,NNdof), dtype = np.float);
    
    # ----------------------- Load vector --------------------------
    # basis function
    phi = np.zeros((nG,6));
    phi[:,0] = _lambda[:,0]*(2*_lambda[:,0]-1);
    phi[:,1] = _lambda[:,1]*(2*_lambda[:,1]-1);
    phi[:,2] = _lambda[:,2]*(2*_lambda[:,2]-1);
    phi[:,3] = 4*_lambda[:,1]*_lambda[:,2];
    phi[:,4] = 4*_lambda[:,0]*_lambda[:,2];
    phi[:,5] = 4*_lambda[:,1]*_lambda[:,0];
    # load vector
    f = pde['f'];
    F = np.zeros( (NT,Ndof) ); # straighten
    for p in range(nG):
        # quadrature points in the x-y coordinate
        pxy = _lambda[p,0]*node[elem[:,0],:] \
            + _lambda[p,1]*node[elem[:,1],:] \
            + _lambda[p,2]*node[elem[:,2],:];
        fxy = f(pxy).reshape((-1,1));
        F = F + weight[p]*fxy*phi[p,:];
    #end
    F = area[:,np.newaxis]*F; # area.reshape((-1,1))*F;
    ff = np.bincount(elem2.ravel("F"), weights = F.ravel("F"), minlength = NNdof);
    
    # ------- Neumann boundary conditions -----------
    bdIndexN = bdStruct['bdIndexN']; elemN = bdStruct['elemN'];
    if elemN.any():
        # Sparse assembling index
        elem1 = np.hstack( (elemN, bdIndexN[:,np.newaxis] + N) ) ;  ndof = 3;
        # Gauss quadrature rule
        _lambda, weight = quadpts1(quadOrder); ng = len(weight);
        # basis function
        phi1 = np.zeros( (ng,3) );
        phi1[:,0] = _lambda[:,0]*(2*_lambda[:,0]-1);
        phi1[:,1] = _lambda[:,1]*(2*_lambda[:,1]-1);
        phi1[:,2] = 4*_lambda[:,0]*_lambda[:,1];
        # nvec
        z1 = node[elemN[:,0],:]; z2 = node[elemN[:,1],:]; nel = elemN.shape[0];
        e = z1-z2;  he = np.sqrt(np.sum(e**2,axis=1));
        nvec = np.zeros( (nel,2), dtype = np.float );
        nvec[:,0] = -e[:,1]/he; nvec[:,1] =  e[:,0]/he;
        # assemble
        FN = np.zeros( (nel,ndof) );
        Du = pde['Du'];
        for p in range(ng):
            pz = _lambda[p,0]*z1 + _lambda[p,1]*z2;
            Dnu = np.sum( Du(pz)*nvec, axis=1 ).reshape((-1,1)); 
            FN = FN + weight[p]*Dnu*phi1[p,:];
        #end
        FN = he[:,np.newaxis]*FN;
        ff = ff + np.bincount(elem1.ravel("F"), weights = FN.ravel("F"), minlength = NNdof);    
    # --------- Dirichlet boundary conditions ---------------
    eD = bdStruct['eD']; bdIndexD = bdStruct['bdIndexD'];
    id = np.hstack( (eD, bdIndexD+N) ); 
    g_D = pde['g_D']; elemD = bdStruct['elemD'];
    isBdNode = np.array( [False]*NNdof ); isBdNode[id] = True; 
    bdDof = isBdNode; freeDof = ~isBdNode;
    z1 = node[elemD[:,0],:]; z2 = node[elemD[:,1],:]; zc = (z1+z2)/2;
    pD = node[eD,:];
    wD = g_D(pD); wc = g_D(zc);   
    uh = np.zeros((NNdof,)); uh[bdDof] = np.hstack( (wD,wc) );
    ff = ff - kk@uh;
    
    # ---------------------- Solver -------------------------
    uh[freeDof] = spsolve( kk[ np.ix_(freeDof,freeDof) ],ff[freeDof] );
    return uh;

def PoissonP3(node,elem,pde,bdStruct,*args):  
    
    if len(args) == 0: quadOrder = 5;
    if len(args) == 1: quadOrder = args[0];    
    
    # -------------- Sparse assembling indices --------------
    # auxstructure
    auxT = auxstructure(node, elem);
    edge = auxT['edge']; 
    elem2edge = auxT['elem2edge'];
    # numbers
    N = node.shape[0]; NT = elem.shape[0]; NE = edge.shape[0];
    NNdof = N + 2*NE + NT; Ndof = 10; #global and local d.o.f. numbers
    # sgnelem
    v1 = [1,2,0]; v2 = [2,0,1]; # L1: 1-2 ( matlab 2-3 )
    bdIndex = bdStruct['bdIndex']; E = np.array([False]*NE); E[bdIndex] = 1;
    sgnelem = np.sign(elem[:,v2] - elem[:,v1]);
    sgnbd = E[elem2edge];  sgnelem[sgnbd] = 1;
    sgnelem[sgnelem==-1] = 0; 
    elema = elem2edge + N*sgnelem + (N+NE)*(sgnelem==0); # 1/3 point
    elemb = elem2edge + (N+NE)*sgnelem + N*(sgnelem==0); # 2/3 point
    # local --> global
    elem2 = np.hstack( (elem, elema, elemb, np.arange(NT).reshape((-1,1))+N+2*NE) );
    # ii,jj    
    nnz = NT*Ndof**2;
    ii = np.array( range(nnz), dtype = np.int); 
    jj = np.array( range(nnz), dtype = np.int);  
    id = 0;
    for i in range(Ndof):
        for j in range(Ndof):
            ii[id:id+NT] = elem2[:,i]; 
            jj[id:id+NT] = elem2[:,j]; 
            id = id + NT; 
        #end
    #end
    
    # -------------- Stiffness matrix -------------
    # _lambda and Dlambda
    _lambda, weight = quadpts(quadOrder); nG = len(weight);
    Dlambda,area = gradbasis(node,elem);
    # stiffness matrix
    K = np.zeros((NT,Ndof**2)); # straighten
    for p in range(nG):
        Dphip = [i for i in range(nG)]; # initialization
        # Dphi at quadrature points
        Dphip[0] = 0.5*(3*Dlambda[0])*(3*_lambda[p,0]-2)*_lambda[p,0] \
                 + 0.5*(3*_lambda[p,0]-1)*(3*Dlambda[0])*_lambda[p,0] \
                 + 0.5*(3*_lambda[p,0]-1)*(3*_lambda[p,0]-2)*Dlambda[0];    
        Dphip[1] = 0.5*(3*Dlambda[1])*(3*_lambda[p,1]-2)*_lambda[p,1] \
                 + 0.5*(3*_lambda[p,1]-1)*(3*Dlambda[1])*_lambda[p,1] \
                 + 0.5*(3*_lambda[p,1]-1)*(3*_lambda[p,1]-2)*Dlambda[1];             
        Dphip[2] = 0.5*(3*Dlambda[2])*(3*_lambda[p,2]-2)*_lambda[p,2] \
                 + 0.5*(3*_lambda[p,2]-1)*(3*Dlambda[2])*_lambda[p,2] \
                 + 0.5*(3*_lambda[p,2]-1)*(3*_lambda[p,2]-2)*Dlambda[2];             
        Dphip[3] = 9/2*Dlambda[2]*_lambda[p,1]*(3*_lambda[p,1]-1) \
                 + 9/2*_lambda[p,2]*Dlambda[1]*(3*_lambda[p,1]-1) \
                 + 9/2*_lambda[p,2]*_lambda[p,1]*(3*Dlambda[1]);             
        Dphip[4] = 9/2*Dlambda[0]*_lambda[p,2]*(3*_lambda[p,2]-1) \
                 + 9/2*_lambda[p,0]*Dlambda[2]*(3*_lambda[p,2]-1) \
                 + 9/2*_lambda[p,0]*_lambda[p,2]*(3*Dlambda[2]);    
        Dphip[5] = 9/2*Dlambda[0]*_lambda[p,1]*(3*_lambda[p,0]-1) \
                 + 9/2*_lambda[p,0]*Dlambda[1]*(3*_lambda[p,0]-1) \
                 + 9/2*_lambda[p,0]*_lambda[p,1]*(3*Dlambda[0]);    
        Dphip[6] = 9/2*Dlambda[2]*_lambda[p,1]*(3*_lambda[p,2]-1) \
                 + 9/2*_lambda[p,2]*Dlambda[1]*(3*_lambda[p,2]-1) \
                 + 9/2*_lambda[p,2]*_lambda[p,1]*(3*Dlambda[2]);             
        Dphip[7] = 9/2*Dlambda[0]*_lambda[p,2]*(3*_lambda[p,0]-1) \
                 + 9/2*_lambda[p,0]*Dlambda[2]*(3*_lambda[p,0]-1) \
                 + 9/2*_lambda[p,0]*_lambda[p,2]*(3*Dlambda[0]);
        Dphip[8] = 9/2*Dlambda[0]*_lambda[p,1]*(3*_lambda[p,1]-1) \
                 + 9/2*_lambda[p,0]*Dlambda[1]*(3*_lambda[p,1]-1) \
                 + 9/2*_lambda[p,0]*_lambda[p,1]*(3*Dlambda[1]);
        Dphip[9] = 27*_lambda[p,0]*_lambda[p,1]*Dlambda[2] \
                 + 27*_lambda[p,0]*_lambda[p,2]*Dlambda[1] \
                 + 27*_lambda[p,2]*_lambda[p,1]*Dlambda[0];
        s = 0;
        for i in range(Ndof):
            for j in range(Ndof):
                K[:,s] +=  weight[p]*np.sum(Dphip[i]*Dphip[j],axis = 1) * area;
                s = s+1;
            #end
        #end
    #end
    kk = csc_matrix( ( K.ravel("F"), (ii,jj) ), shape = (NNdof,NNdof), dtype = np.float);
    
    # ----------------------- Load vector --------------------------
    # basis function
    phi = np.zeros((nG,Ndof));
    phi[:,0] = 0.5*(3*_lambda[:,0]-1)*(3*_lambda[:,0]-2)*_lambda[:,0];
    phi[:,1] = 0.5*(3*_lambda[:,1]-1)*(3*_lambda[:,1]-2)*_lambda[:,1];
    phi[:,2] = 0.5*(3*_lambda[:,2]-1)*(3*_lambda[:,2]-2)*_lambda[:,2];
    phi[:,3] = 9/2*_lambda[:,2]*_lambda[:,1]*(3*_lambda[:,1]-1);
    phi[:,4] = 9/2*_lambda[:,0]*_lambda[:,2]*(3*_lambda[:,2]-1);
    phi[:,5] = 9/2*_lambda[:,1]*_lambda[:,0]*(3*_lambda[:,0]-1);
    phi[:,6] = 9/2*_lambda[:,1]*_lambda[:,2]*(3*_lambda[:,2]-1);
    phi[:,7] = 9/2*_lambda[:,2]*_lambda[:,0]*(3*_lambda[:,0]-1);
    phi[:,8] = 9/2*_lambda[:,1]*_lambda[:,0]*(3*_lambda[:,1]-1);
    phi[:,9] = 27*_lambda[:,0]*_lambda[:,1]*_lambda[:,2];
    # load vector
    f = pde['f'];
    F = np.zeros( (NT,Ndof) ); # straighten
    for p in range(nG):
        # quadrature points in the x-y coordinate
        pxy = _lambda[p,0]*node[elem[:,0],:] \
            + _lambda[p,1]*node[elem[:,1],:] \
            + _lambda[p,2]*node[elem[:,2],:];
        fxy = f(pxy).reshape((-1,1));
        F = F + weight[p]*fxy*phi[p,:];
    #end
    F = area[:,np.newaxis]*F; # area.reshape((-1,1))*F;
    ff = np.bincount(elem2.ravel("F"), weights = F.ravel("F"), minlength = NNdof);
    
    # ------- Neumann boundary conditions -----------
    bdIndexN = bdStruct['bdIndexN']; elemN = bdStruct['elemN'];
    if elemN.any():
        # Sparse assembling index
        bdIndexN1 = bdIndexN[:,np.newaxis];
        elem1 = np.hstack( (elemN, bdIndexN1+N, bdIndexN1+N+NE) ) ;  ndof = 4;
        # Gauss quadrature rule
        _lambda, weight = quadpts1(quadOrder); ng = len(weight);
        # basis function
        phi1 = np.zeros( (ng,ndof) );
        phi1[:,0] = 0.5*(3*_lambda[:,0]-1)*(3*_lambda[:,0]-2)*_lambda[:,0];
        phi1[:,1] = 0.5*(3*_lambda[:,1]-1)*(3*_lambda[:,1]-2)*_lambda[:,1];
        phi1[:,2] = 9/2*_lambda[:,0]*_lambda[:,1]*(3*_lambda[:,0]-1);
        phi1[:,3] = 9/2*_lambda[:,1]*_lambda[:,0]*(3*_lambda[:,1]-1);
        # nvec
        z1 = node[elemN[:,0],:]; z2 = node[elemN[:,1],:]; nel = elemN.shape[0];
        e = z1-z2;  he = np.sqrt(np.sum(e**2,axis=1));
        nvec = np.zeros( (nel,2), dtype = np.float );
        nvec[:,0] = -e[:,1]/he; nvec[:,1] =  e[:,0]/he;
        # assemble
        FN = np.zeros( (nel,ndof) );
        Du = pde['Du'];
        for p in range(ng):
            pz = _lambda[p,0]*z1 + _lambda[p,1]*z2;
            Dnu = np.sum( Du(pz)*nvec, axis=1 ).reshape((-1,1)); 
            FN = FN + weight[p]*Dnu*phi1[p,:];
        #end
        FN = he[:,np.newaxis]*FN;
        ff = ff + np.bincount(elem1.ravel("F"), weights = FN.ravel("F"), minlength = NNdof);    
    # --------- Dirichlet boundary conditions ---------------
    eD = bdStruct['eD']; bdIndexD = bdStruct['bdIndexD'];
    id = np.hstack( (eD, bdIndexD+N, bdIndexD+N+NE) ); 
    g_D = pde['g_D']; elemD = bdStruct['elemD'];
    isBdNode = np.array( [False]*NNdof ); isBdNode[id] = True; 
    bdDof = isBdNode; freeDof = ~isBdNode; # only valid for bool ndarray
    z1 = node[elemD[:,0],:]; z2 = node[elemD[:,1],:]; 
    za = z1+(z2-z1)/3;  zb = z1+2*(z2-z1)/3;
    pD = node[eD,:];
    uD = g_D(pD);  uDa = g_D(za);  uDb = g_D(zb);
    uh = np.zeros((NNdof,)); uh[bdDof] = np.hstack( (uD,uDa,uDb) );
    ff = ff - kk@uh;
    
    # ---------------------- Solver -------------------------
    uh[freeDof] = spsolve( kk[ np.ix_(freeDof,freeDof) ],ff[freeDof] );
    return uh;