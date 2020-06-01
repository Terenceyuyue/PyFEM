"""

@author: Terenceyuyue
"""

import numpy as np
#from scipy.sparse import csc_matrix
# matlab 
from matlab.basic import m_sparse, m_find, m_unique


class setboundary(object):
    def __init__(self,node,elem,*args):
        
        if len(args)==0: 
            bdNeumann = [];
        else:
            bdNeumann = args[0];           
        
        # -------------- totalEdge ----------------
        allEdge = np.vstack( ( elem[:,[1,2]], elem[:,[2,0]], elem[:,[0,1]]  ) );
        totalEdge = np.sort(allEdge);
        
        # --------  counterclockwise bdEdge --------
        N = totalEdge.shape[0];
        ii = totalEdge[:,1];  jj = totalEdge[:,0];  ss = np.ones(N);
        A = m_sparse(ii,jj,ss,N,N);
        
        _, _, s = m_find(A);
        _, i1, _ = m_unique(totalEdge);
        bdEdge = allEdge[i1[s==1],:];
        
        # --------- set boundary --------
        nE = bdEdge.shape[0];
        # initial as Dirichlet (True for Dirichlet, False for Neumann)
        bdFlag = np.array( [True]*nE );
        nodebdEdge = ( node[bdEdge[:,0],:] + node[bdEdge[:,1],:] )/2;
        x1 = nodebdEdge[:,0]; y1 = nodebdEdge[:,1];    
        if bdNeumann != []:
            # id = np.array([False]*nE);
            # for i in range(nE):
            #     x = x1[i]; y = y1[i];
            #     if eval(bdNeumann):
            #        id[i] = True;
            id = [eval(bdNeumann) for x,y in zip(x1,y1)];
            bdFlag[id] = False;
        
        elemD = bdEdge[bdFlag, :];
        elemN = bdEdge[~bdFlag, :]; # np.array 型的逻辑数组可取反
        eD = np.unique(elemD);
        bdIndex = np.where(s==1)[0]; # 注意 np.where 给出的是元组
        bdIndexD = bdIndex[bdFlag]; 
        bdIndexN = bdIndex[~bdFlag];
        
        self.node = node;  self.elem = elem;
        self.bdNeumann = bdNeumann;
        self.elemD = elemD; self.elemN = elemN;
        self.eD = eD; 
        self.bdIndex = bdIndex; 
        self.bdIndexD = bdIndexD; 
        self.bdIndexN = bdIndexN;
            