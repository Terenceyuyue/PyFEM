# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""

# pde data module for 1-D problems

import numpy as np
import sympy as sym

def pde1D(para):
    a = para[0]; b = para[1]; c = para[2];
    x = sym.Symbol('x');
    c1 = 0.5/np.exp(1); c2 = -0.5*(1+1/np.exp(1));
    # exact solution
    u = c1*sym.exp(2*x)+c2*sym.exp(x)+1/2;
    Du = sym.diff(u);
    f = -a*sym.diff(u,x,2)+b*sym.diff(u,x,1)+c*u;
    # 转化为 lambda 函数
    u = sym.lambdify(x,u,"numpy"); # lambda 表达式
    Du = sym.lambdify(x,Du,"numpy"); # lambda 表达式
    f = sym.lambdify(x,f,"numpy"); # lambda 表达式
    g_D = u;
    pde = {'f':f, 'uexact':u, 'Du':Du, 'g_D':g_D, 'para':para};
    
    return pde;