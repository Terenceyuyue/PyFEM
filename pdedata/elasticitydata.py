# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""

import numpy as np

def elasticitydata(*args):
    
    # Lame constants
    if len(args)==0: 
        lambda_ = 1; mu = 1;
    else: 
        lambda_ = args[0]; mu = args[1];
    
    sin = np.sin; cos = np.cos; pi = np.pi;
    def uexact(p):
        x = p[:,0]; y = p[:,1];
        u1 = cos(pi*x)*cos(pi*y); u2 = sin(pi*x)*sin(pi*y);
        u1 = u1[:,np.newaxis]; u2 = u2[:,np.newaxis];
        return np.hstack((u1,u2));
    
    def f(p):
        return mu*2*pi**2*uexact(p);
    
    def g_D(p):
        return uexact(p);
    
    def Du(p):
        x = p[:,0]; y = p[:,1];
        u1 = -pi*sin(pi*x)*cos(pi*y); # u1x
        u2 = -pi*cos(pi*x)*sin(pi*y); # u1y
        u3 = pi*cos(pi*x)*sin(pi*y);  # u2x
        u4 = pi*sin(pi*x)*cos(pi*y);  # u2y
        u1 = u1[:,np.newaxis]; u2 = u2[:,np.newaxis];
        u3 = u3[:,np.newaxis]; u4 = u4[:,np.newaxis];
        return np.hstack((u1,u2,u3,u4));
    
    def g_N(p):
        x = p[:,0]; y = p[:,1];
        E11 = -pi*sin(pi*x)*cos(pi*y); E22 = pi*sin(pi*x)*cos(pi*y);
        E12 = 0.5*(-pi*cos(pi*x)*sin(pi*y) + pi*cos(pi*x)*sin(pi*y));
        sig11 = (lambda_+2*mu)*E11 + lambda_*E22;
        sig22 = (lambda_+2*mu)*E22 + lambda_*E11;
        sig12 = 2*mu*E12;
        sig11 = sig11[:,np.newaxis]; 
        sig22 = sig22[:,np.newaxis];
        sig12 = sig12[:,np.newaxis];
        return np.hstack( (sig11,sig22,sig12) );
    
    pde = {'lambda_':lambda_, 'mu':mu, 'f':f, 'uexact':uexact, 'g_D':g_D, \
           'g_N':g_N, 'Du':Du};
    return pde