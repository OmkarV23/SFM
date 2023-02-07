import cupy as cp

def Essential_mat(K,F):
    K = cp.asarray(K)
    print(K)
    E = K.T @ F @ K
    u,_,vt = cp.linalg.svd(E)
    E = u @ cp.diag([1,1,0]) @ vt
    return E