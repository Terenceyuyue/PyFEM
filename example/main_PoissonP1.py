# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""

import sys
sys.path.append("../")  # 导入上一级目录的模块

import numpy as np
import time

# pde
from pdedata.Poissondata import Poissondata
# tool
from tool.setboundary import setboundary
from tool.mesh import squaremesh
from tool.getError import getL2error, getH1error, showrateh
from tool.mesh import uniformrefine

# Poisson func
from fem.Poisson import Poisson

"""
############################ main_Poisson #############################
"""
tic = time.process_time();

#  -------------- Mesh and boundary conditions --------------
a1 = 0; b1 = 1; a2 = 0; b2 = 1;
N = 2;
Nx = N; Ny = N; h1 = (b1-a1)/Nx; h2 = (b2-a2)/Ny;
node,elem = squaremesh([a1,b1,a2,b2],h1,h2);

bdNeumann = "abs(x-1)<=1e-4";

# ------------------ PDE data -------------------
pde = Poissondata();
u = pde['uexact']; Du = pde['Du'];

# ----------------- Poisson ---------------------
maxIt = 5;
h = np.zeros((maxIt,), dtype = np.float);
ErrL2 = np.zeros((maxIt,), dtype = np.float); 
ErrH1 = np.zeros((maxIt,), dtype = np.float);

for k in range(maxIt):
    node,elem = uniformrefine(node, elem);
    bdStruct = setboundary(node,elem,bdNeumann);
    uh = Poisson(node, elem, pde, bdStruct);
    NT = elem.shape[0];
    h[k] = 1/np.sqrt(NT);
    ErrL2[k] = getL2error(node, elem, uh, u);
    ErrH1[k] = getH1error(node, elem, uh, Du);    

# -------------------- Show rate -----------------------    
showrateh(h,ErrL2,ErrH1)


toc = (time.process_time() - tic);
print('process_time = %.4f s' % toc)




