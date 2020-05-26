# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""
import sys
sys.path.append("../")  # 导入上一级目录的模块


import time
import matplotlib.pyplot as plt
from tool.visualize import showmesh, findnode, showsolution #, findelem, findedge

# pde
from pdedata.Poissondata import Poissondata
# tool
from tool.setboundary import setboundary
from tool.mesh import squaremesh
from tool.getError import getL2error, getH1error

# Poisson func
from fem.Poisson import Poisson

"""
############################ main_Poisson #############################
"""
tic = time.process_time();

#  -------------- Mesh and boundary conditions --------------
a1 = 0; b1 = 1; a2 = 0; b2 = 1;
N = 5;
Nx = N; Ny = N; h1 = (b1-a1)/Nx; h2 = (b2-a2)/Ny;
node,elem = squaremesh([a1,b1,a2,b2],h1,h2);

bdNeumann = "abs(x-1)<=1e-4";
bdStruct = setboundary(node, elem, bdNeumann);

# ---------------- showmesh -------------------
plt.figure(figsize=(6,6))
showmesh(node,elem);
findnode(node);
# findelem(node,elem);
# findedge(node,elem);

# ------------------ PDE data -------------------
pde = Poissondata();

# ----------------- Poisson ---------------------
uh = Poisson(node, elem, pde, bdStruct);


# --------------------- error analysis ----------------------
u = pde['uexact'];  Du = pde['Du'];
ue = u(node);
plt.figure()
ax = plt.subplot(1,2,1, projection = '3d'); 
showsolution(node,elem, uh, ax);
ax = plt.subplot(1,2,2, projection = '3d');
showsolution(node,elem, ue, ax);

# L2 and H1 errors
ErrL2 = getL2error(node, elem, uh, u);
ErrH1 = getH1error(node, elem, uh, Du);  
print('ErrL2 = %.2e, ErrH1 = %.2e' % (ErrL2, ErrH1) )  


toc = (time.process_time() - tic);
print('process_time = %.4f s' % toc)





