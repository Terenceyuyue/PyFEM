# -*- coding: utf-8 -*-
"""

@author: Terenceyuyue
"""

import numpy as np
from scipy.sparse import csc_matrix, issparse

# sparse matrix
def m_sparse(i,j,s,m,n):
    A = csc_matrix((s,(i,j)),shape = (m,n)); # sparse in matlab
    return A;

def m_issparse(A):
    return issparse(A);

def m_accumarray(subs, vals, N):
    return np.bincount(subs, weights = vals, minlength = N);

# find nonzero entries
def m_find(A,*args): 
    
    if len(args)==0:
        cond = '>0';
    else:
        cond = args[0];
        
    if len(A.shape)>1:     # matrix
        if not issparse(A):
            A = A.transpose(); # np.where 是含行查找，matlab 是按列查找
            _str = 'A'+cond;
            index = np.where(eval(_str));
            i = index[1]; j = index[0];  s = A[j,i]; # 注意转置了要还原
            return i,j,s
        if issparse(A):
           ip = A.indices; indptr = A.indptr; sp = A.data;
           jp = np.zeros((len(ip),), dtype = np.int);
           for i in range(A.shape[0]):
               jp[indptr[i]:indptr[i+1]] = i;
           _str = 'sp'+cond; 
           i = ip[eval(_str)]; 
           j = jp[eval(_str)]; 
           s = sp[eval(_str)];
           return i,j,s;          
    else: # 向量
        _str = 'A' + cond; # 字符串拼接
        i = np.where(eval(_str)); i = i[0];
        return i;
        

# unique(A, 'rows')
def m_unique(A):    
    C,IA,IC,In = np.unique(A, return_index=True, return_inverse=True,  \
                            return_counts=True, axis=0); # axis = 0 行
    return C,IA,IC


def m_repmat(A,ind):
    A = np.tile(A,ind);
    return A;
        
      