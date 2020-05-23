# -*- coding: utf-8 -*-
"""
Created on Thu May 21 17:18:11 2020

@author: Terenceyuyue
"""

import numpy as np

def gradbasis(node,elem):
    
    z1 = node[elem[:,0],:];   z2 = node[elem[:,1],:]; z3 = node[elem[:,2],:];
    e1 = z2-z3; e2 = z3-z1; e3 = z1-z2; # ei = [xi; etai]
    area = 0.5*(-e3[:,0]*e2[:,1]+e3[:,1]*e2[:,0]);
    
    grad1 = np.array( [e1[:,1], -e1[:,0]] )/(2*area); grad1 = grad1.transpose();
    grad2 = np.array( [e2[:,1], -e2[:,0]] )/(2*area); grad2 = grad2.transpose();
    grad3 = -(grad1+grad2);  
    
    # # The following can be replaced by  Dphi = [grad1, grad2, grad3];
    # Dphi = np.zeros( (3,len(area),2) );
    # Dphi[0] = grad1;  # Dphi[0] is short for Dphi[0,:,:]
    # Dphi[1] = grad2;
    # Dphi[2] = grad3;
    Dphi = [grad1, grad2, grad3]
    
    return Dphi, area