# -*- coding: utf-8 -*-
"""
Created on Sun May 17 21:20:52 2020

@author: Terenceyuyue
"""

import numpy as np

def Poissondata():
    def uexact(p):
        x = p[:,0]; y = p[:,1];
        return y**2*np.sin(np.pi*x);
    
    def f(p):
        x = p[:,0]; y = p[:,1];
        return (np.pi**2*y**2-2)*np.sin(np.pi*x);
    
    def Du(p):
        x = p[:,0]; y = p[:,1];
        val = np.array( [np.pi*y**2*np.cos(np.pi*x), 2*y*np.sin(np.pi*x)] );
        return val.transpose();
    
    def g_D(p):
        return uexact(p);

    pde = {'uexact':uexact, 'f':f, 'Du':Du, 'g_D':g_D};
    return pde;
