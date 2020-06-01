# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""

import numpy as np

class Poissondata:
    def __init__(self):
        pass
    
    def uexact(self,p):
        x = p[:,0]; y = p[:,1];
        return y**2*np.sin(np.pi*x);
    
    def f(self,p):
        x = p[:,0]; y = p[:,1];
        return (np.pi**2*y**2-2)*np.sin(np.pi*x);
    
    def Du(self,p):
        x = p[:,0]; y = p[:,1];
        val = np.array( [np.pi*y**2*np.cos(np.pi*x), 2*y*np.sin(np.pi*x)] );
        return val.transpose();
    
    def g_D(self,p):
        return self.uexact(p);

