# -*- coding: utf-8 -*-
"""
@author: Terenceyuyue
"""

import sys
sys.path.append("../")  # 导入上一级目录的模块

import numpy as np
import time

# pde
from pdedata.elasticitydata import elasticitydata

# tool
from tool.mesh import squaremesh,uniformrefine
from tool.setboundary import setboundary
from tool.getError import getL2error, getH1error, showrateh

# func
from fem.elasticity import elasticity1

tic = time.process_time();

#  -------------- Mesh and boundary conditions --------------
a1 = 0; b1 = 1; a2 = 0; b2 = 1;
N = 4;
Nx = N; Ny = N; h1 = (b1-a1)/Nx; h2 = (b2-a2)/Ny;
node,elem = squaremesh([a1,b1,a2,b2],h1,h2);

bdNeumann = " abs(y-0)<1e-4 or abs(x-1)<1e-4 "; # string for Neumann

# ------------------------ PDE data ------------------------
lambda_ = 1; mu = 1;  
pde = elasticitydata(lambda_, mu);
uexact = pde['uexact']; Du = pde['Du'];

# ----------------- elasticity1 ---------------------
maxIt = 4;
h = np.zeros((maxIt,), dtype = np.float);
ErrL2 = np.zeros((maxIt,), dtype = np.float); 
ErrH1 = np.zeros((maxIt,), dtype = np.float);
for k in range(maxIt):
    node,elem = uniformrefine(node,elem);
    bdStruct = setboundary(node,elem,bdNeumann);
    uh = elasticity1(node,elem,pde,bdStruct);
    uh = uh.reshape((2,-1)).transpose();
    NT = elem.shape[0]; h[k] = 1/np.sqrt(NT);
    
    errL2 = np.zeros((2,), dtype = np.float); 
    errH1 = np.zeros((2,), dtype = np.float); 
    for id in range(2):
         uid = uh[:,id];
         ue = lambda pz: uexact(pz)[:,id]; 
         Due = lambda pz: Du(pz)[:,2*id:2*id+2];
         errL2[id] = getL2error(node,elem,uid,ue);
         errH1[id] = getH1error(node,elem,uid,Due);
    
    ErrL2[k] = np.linalg.norm(errL2);
    ErrH1[k] = np.linalg.norm(errH1);
    

# -------------------- Show rate -----------------------    
showrateh(h,ErrL2,ErrH1)


toc = (time.process_time() - tic);
print('process_time = %.4f s' % toc)    