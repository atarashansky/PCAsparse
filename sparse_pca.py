import scipy as sp

def sparse_pca(X,npcs,mu = None):
    # X -- scipy sparse data matrix
    # npcs -- number of principal components
    # mu -- precomputed feature means. if None, calculates them from X.

    # compute mean of data features
    if mu is None: 
        mu = X.mean(0).A.flatten()[None,:]

    # dot product operator for the means
    mmat = mdot = mu.dot 
    # dot product operator for the transposed means
    mhmat = mhdot = mu.T.dot 
    # dot product operator for the data
    Xmat = Xdot = X.dot 
    # dot product operator for the transposed data
    XHmat = XHdot = X.T.conj().dot 
    # dot product operator for a vector of ones
    ones = np.ones(X.shape[0])[None,:].dot 

    # modify the matrix/vector dot products to subtract the means
    def matvec(x): 
        return Xdot(x) - mdot(x)
    def matmat(x): 
        return Xmat(x) - mmat(x)
    def rmatvec(x): 
        return XHdot(x) - mhdot(ones(x))
    def rmatmat(x): 
        return XHmat(x) - mhmat(ones(x))
    
    # construct the LinearOperator
    XL = sp.sparse.linalg.LinearOperator(matvec = matvec, dtype = X.dtype,
                                         matmat = matmat,
                                         shape = X.shape,
                                        rmatvec = rmatvec, rmatmat = rmatmat)
     
    # I chose to use 'arpack' in this case as it is 2x faster for my specific application
    u,s,v = sp.sparse.linalg.svds(XL,solver='arpack',k=npcs)
    
    # i like my eigenvalues sorted in decreasing order
    idx = np.argsort(-s)
    S = np.diag(s[idx])
    # principal components
    pcs = u[:,idx].dot(S) 
    # equivalent to PCA.components_ in sklearn 
    components_ = v[idx,:] 
    return pcs,components_
