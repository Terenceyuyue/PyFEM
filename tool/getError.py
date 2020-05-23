# -*- coding: utf-8 -*-
"""
Created on Thu May 21 22:48:20 2020

@author: Terenceyuyue
"""

import numpy as np
import matplotlib.pyplot as plt
from tool.GaussQuad import quadpts
from tool.auxdata import auxstructure
from tool.setboundary import setboundary

def getL2error(node,elem,uh,u,*args):
    
    if len(args) == 0: feSpace = 'P1'; quadOrder = 3;
    if len(args) == 1: feSpace = args[0]; quadOrder = 3;
    if len(args) == 2: feSpace = args[0]; quadOrder = args[1];

    N = node.shape[0]; NT = elem.shape[0];
    # Gauss quadrature rule
    _lambda, weight = quadpts(quadOrder); nG = len(weight);
    # area of triangles
    z1 = node[elem[:,0],:]; z2 = node[elem[:,1],:]; z3 = node[elem[:,2],:];
    ve2 = z1-z3; ve3 = z2-z1;
    area = 0.5*abs( -ve3[:,0]*ve2[:,1] + ve3[:,1]*ve2[:,0] );
    # auxstructure
    auxT = auxstructure(node, elem);
    elem2edge = auxT['elem2edge'];
    
    ## P1-Lagrange
    if feSpace == 'P1':
        elem2dof = elem;
        phi = _lambda; # basis functions
    
    ## P2-Lagrange
    if feSpace == 'P2':
        elem2dof = np.hstack( (elem, elem2edge+N) );
        phi = np.zeros((nG,6));
        phi[:,0] = _lambda[:,0]*(2*_lambda[:,0]-1);
        phi[:,1] = _lambda[:,1]*(2*_lambda[:,1]-1);
        phi[:,2] = _lambda[:,2]*(2*_lambda[:,2]-1);
        phi[:,3] = 4*_lambda[:,1]*_lambda[:,2];
        phi[:,4] = 4*_lambda[:,0]*_lambda[:,2];
        phi[:,5] = 4*_lambda[:,1]*_lambda[:,0];
    
    
    ## P3-Lagrange
    if feSpace == 'P3':
        # sgnelem
        edge = auxT['edge']; NE = edge.shape[0];
        bdStruct = setboundary(node,elem);
        v1 = [1,2,0]; v2 = [2,0,1]; # L1: 1-2 ( matlab 2-3 )
        bdIndex = bdStruct['bdIndex']; E = np.array([False]*NE); E[bdIndex] = 1;
        sgnelem = np.sign(elem[:,v2] - elem[:,v1]);
        sgnbd = E[elem2edge];  sgnelem[sgnbd] = 1;
        sgnelem[sgnelem==-1] = 0; 
        elema = elem2edge + N*sgnelem + (N+NE)*(sgnelem==0); # 1/3 point
        elemb = elem2edge + (N+NE)*sgnelem + N*(sgnelem==0); # 2/3 point
        # local --> global
        elem2dof = np.hstack( (elem, elema, elemb, np.arange(NT).reshape((-1,1))+N+2*NE) );
        phi = np.zeros((nG,10));
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
        
    Ndof = elem2dof.shape[1];
    
    ## elementwise error
    err = np.zeros( (NT,) );
    for p in range(nG):
        uhp = 0;
        for j in range(Ndof):
            uhp = uhp + uh[elem2dof[:,j]]*phi[p,j];
        pz = _lambda[p,0]*z1 + _lambda[p,1]*z2 + _lambda[p,2]*z3;
        err = err + weight[p]*(u(pz)-uhp)**2;    
    err = area*err;
    # Modification
    err[np.isnan(err)] = 0; # singular values, i.e. uexact(p) = infty, are excluded
    err = np.sqrt(abs(sum(err)));
    
    return err;
    
def getH1error(node,elem,uh,Du,*args):
    
    if len(args) == 0: feSpace = 'P1'; quadOrder = 3;
    if len(args) == 1: feSpace = args[0]; quadOrder = 3;
    if len(args) == 2: feSpace = args[0]; quadOrder = args[1];
    
    N = node.shape[0]; NT = elem.shape[0];
    uh = uh.reshape((-1,1));
    # Gauss quadrature rule
    _lambda, weight = quadpts(quadOrder); nG = len(weight);
    # auxstructure
    auxT = auxstructure(node, elem);
    elem2edge = auxT['elem2edge'];
    # area
    z1 = node[elem[:,0],:]; z2 = node[elem[:,1],:]; z3 = node[elem[:,2],:];
    xi = np.array( [ z2[:,0]-z3[:,0], z3[:,0]-z1[:,0], z1[:,0]-z2[:,0] ] ).transpose();
    eta = np.array( [z2[:,1]-z3[:,1], z3[:,1]-z1[:,1], z1[:,1]-z2[:,1] ] ).transpose();
    area = 0.5*(xi[:,0]*eta[:,1]-xi[:,1]*eta[:,0]);
    area1 = area.reshape((-1,1));
    # gradbasis
    D_lambdax = eta/np.tile(2*area1,3); # np.tile 可以去掉，相当于 matlab 的 np.tile
    D_lambday = -xi/np.tile(2*area1,3);
    D_lambda1 = np.array( [ D_lambdax[:,0], D_lambday[:,0] ] ).transpose();
    D_lambda2 = np.array( [ D_lambdax[:,1], D_lambday[:,1] ] ).transpose();
    D_lambda3 = np.array( [ D_lambdax[:,2], D_lambday[:,2] ] ).transpose();
        
    ## P1-Lagrange
    if feSpace == 'P1':
        # elementwise d.o.f.s
        elem2dof = elem;
        # numerical gradient  (at p-th quadrature point)
        Duh =  np.tile(uh[elem2dof[:,0]],2)*D_lambda1 \
            +  np.tile(uh[elem2dof[:,1]],2)*D_lambda2 \
            +  np.tile(uh[elem2dof[:,2]],2)*D_lambda3;
        # elementwise error
        err = np.zeros( (NT,) );
        for p in range(nG):
            pz =  _lambda[p,0]*z1 + _lambda[p,1]*z2 + _lambda[p,2]*z3;
            err = err + weight[p]*np.sum((Du(pz)-Duh)**2,axis = 1);
        
    if feSpace == 'P2':
        # elementwise d.o.f.s
        elem2dof = np.hstack( (elem, elem2edge+N) );
        # numerical gradient (at p-th quadrature point)
        err = np.zeros( (NT,) );
        for p in range(nG):
            Dphip1 = (4*_lambda[p,0]-1)*D_lambda1;
            Dphip2 = (4*_lambda[p,1]-1)*D_lambda2;
            Dphip3 = (4*_lambda[p,2]-1)*D_lambda3;
            Dphip4 = 4*(_lambda[p,1]*D_lambda3+_lambda[p,2]*D_lambda2);
            Dphip5 = 4*(_lambda[p,2]*D_lambda1+_lambda[p,0]*D_lambda3);
            Dphip6 = 4*(_lambda[p,0]*D_lambda2+_lambda[p,1]*D_lambda1);
            Duh =  np.tile(uh[elem2dof[:,0]],2)*Dphip1 \
                +  np.tile(uh[elem2dof[:,1]],2)*Dphip2 \
                +  np.tile(uh[elem2dof[:,2]],2)*Dphip3 \
                +  np.tile(uh[elem2dof[:,3]],2)*Dphip4 \
                +  np.tile(uh[elem2dof[:,4]],2)*Dphip5 \
                +  np.tile(uh[elem2dof[:,5]],2)*Dphip6;
            # elementwise error
            pz =  _lambda[p,0]*z1 + _lambda[p,1]*z2 + _lambda[p,2]*z3;
            err = err + weight[p]*np.sum((Du(pz)-Duh)**2,axis = 1);
            
    ## P3-Lagrange
    if feSpace == 'P3':
        # sgnelem
        edge = auxT['edge']; NE = edge.shape[0];
        bdStruct = setboundary(node,elem);
        v1 = [1,2,0]; v2 = [2,0,1]; # L1: 1-2 ( matlab 2-3 )
        bdIndex = bdStruct['bdIndex']; E = np.array([False]*NE); E[bdIndex] = 1;
        sgnelem = np.sign(elem[:,v2] - elem[:,v1]);
        sgnbd = E[elem2edge];  sgnelem[sgnbd] = 1;
        sgnelem[sgnelem==-1] = 0;
        elema = elem2edge + N*sgnelem + (N+NE)*(sgnelem==0); # 1/3 point
        elemb = elem2edge + (N+NE)*sgnelem + N*(sgnelem==0); # 2/3 point
        # local --> global
        elem2dof = np.hstack( (elem, elema, elemb, np.arange(NT).reshape((-1,1))+N+2*NE) );       
        # numerical gradient (at p-th quadrature point)
        err = np.zeros((NT,));
        for p in range(nG):
            Dphip1 = (27/2*_lambda[p,0]*_lambda[p,0]-9*_lambda[p,0]+1)*D_lambda1;           
            Dphip2 = (27/2*_lambda[p,1]*_lambda[p,1]-9*_lambda[p,1]+1)*D_lambda2; 
            Dphip3 = (27/2*_lambda[p,2]*_lambda[p,2]-9*_lambda[p,2]+1)*D_lambda3;
            Dphip4 = 9/2*((3*_lambda[p,1]*_lambda[p,1]-_lambda[p,1])*D_lambda3+\
                    _lambda[p,2]*(6*_lambda[p,1]-1)*D_lambda2);
            Dphip5 = 9/2*((3*_lambda[p,2]*_lambda[p,2]-_lambda[p,2])*D_lambda1+\
                     _lambda[p,0]*(6*_lambda[p,2]-1)*D_lambda3);        
            Dphip6 = 9/2*((3*_lambda[p,0]*_lambda[p,0]-_lambda[p,0])*D_lambda2+\
                     _lambda[p,1]*(6*_lambda[p,0]-1)*D_lambda1);
            Dphip7 = 9/2*((3*_lambda[p,2]*_lambda[p,2]-_lambda[p,2])*D_lambda2+\
                     _lambda[p,1]*(6*_lambda[p,2]-1)*D_lambda3);
            Dphip8 = 9/2*((3*_lambda[p,0]*_lambda[p,0]-_lambda[p,0])*D_lambda3+\
                     _lambda[p,2]*(6*_lambda[p,0]-1)*D_lambda1);  
            Dphip9 = 9/2*((3*_lambda[p,1]*_lambda[p,1]-_lambda[p,1])*D_lambda1+\
                     _lambda[p,0]*(6*_lambda[p,1]-1)*D_lambda2);  
            Dphip10 = 27*(_lambda[p,0]*_lambda[p,1]*D_lambda3+_lambda[p,0]*_lambda[p,2]*D_lambda2+\
                     _lambda[p,2]*_lambda[p,1]*D_lambda1);
            Duh =  np.tile(uh[elem2dof[:,0]],2)*Dphip1 \
                +  np.tile(uh[elem2dof[:,1]],2)*Dphip2 \
                +  np.tile(uh[elem2dof[:,2]],2)*Dphip3 \
                +  np.tile(uh[elem2dof[:,3]],2)*Dphip4 \
                +  np.tile(uh[elem2dof[:,4]],2)*Dphip5 \
                +  np.tile(uh[elem2dof[:,5]],2)*Dphip6 \
                +  np.tile(uh[elem2dof[:,6]],2)*Dphip7 \
                +  np.tile(uh[elem2dof[:,7]],2)*Dphip8 \
                +  np.tile(uh[elem2dof[:,8]],2)*Dphip9 \
                +  np.tile(uh[elem2dof[:,9]],2)*Dphip10;
            # elementwise error
            pz =  _lambda[p,0]*z1 + _lambda[p,1]*z2 + _lambda[p,2]*z3;
            err = err + weight[p]*np.sum((Du(pz)-Duh)**2, axis = 1);    
            
    err = area*err; 
    # Modification
    err[np.isnan(err)] = 0; # singular values, i.e. uexact(p) = infty, are excluded
    err = np.sqrt(abs(sum(err))); 
    
    return err;


def showrate(h,err,opt1,opt2,strs):
    
    err[err == 0] = 1e-16; # Prevent the case err = 0, log(err) = -Inf.
    p = np.polyfit( np.log(h), np.log(err), deg = 1 );
    r = p[0];
    s = 0.75*err[0]/h[0]**r;
    
    plt.loglog(h,err,opt1, label = strs, linewidth = 2);
    plt.loglog(h,s*h**r,opt2, label = '$O (h^{%0.2f})$'%(r), linewidth = 1);    
    

def showrateh(h,ErrL2,ErrH1,*args):
    
    if len(args) == 0:
       str1 = '||u - u_h||';   # L2 \
       str2 = '||Du - Du_h||'; # H1  
    else:
       str1 = args[0]; str2 = args[1];
    
    showrate(h,ErrH1,'r-*','k.-', str2);
    showrate(h,ErrL2,'b-s','k--', str1);   
    plt.legend(loc = 'best')
    plt.show()
     