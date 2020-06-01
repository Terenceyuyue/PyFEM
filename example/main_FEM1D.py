# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""

import sys
sys.path.append('../') # 导入上一级目录的模块

import numpy as np
import matplotlib.pyplot as plt

# pde
from pdedata.pdedata1D import pde1D
# func
from fem.FEM1D import FEM1D


#  --------------------- Mesh ----------------------
a = 0; b = 1;
nel = 10;  N = nel+1; # numbers of elements and nodes
node = np.linspace(a,b,nel+1);
elem1D = np.zeros( (nel,2) , dtype = np.int); 
elem1D[:,0] = range(0,N-1);  elem1D[:,1] = range(1,N); 

class bdStruct: Neumann = [0];   Dirichlet = [N-1];

#  ---------------------- PDE ----------------------
a = 1;  b = 0;  c = 0;  para = [a,b,c]
pde = pde1D(para);
f = pde.f; u = pde.uexact; 


# ----------------------- FEM1D --------------------
uh = FEM1D(node,elem1D ,pde,bdStruct);

#print(uh)


# ------------ error analysis -----------
plt.figure(figsize=(6,4))

plt.plot(node,u(node),"r-",label="$u$",linewidth=2) # exact
plt.plot(node,uh,"k--",label="$u_h$",linewidth=2)

plt.legend()
plt.xlabel("$x$"); 
plt.title("FEM1D");
plt.show()



    