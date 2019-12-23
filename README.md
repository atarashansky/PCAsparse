# PCAsparse

A function that can run PCA on scipy sparse inputs using LinearOperators to perform implicit mean centering prior to SVD.

This function was written thanks to helpful discussions found in this scikit-learn issue:
https://github.com/scikit-learn/scikit-learn/issues/12794

This function is a little slower than `sklearn.decomposition.TruncatedSVD` as `TruncatedSVD` is run using the randomized SVD solver, which does not accept as input a `scipy.sparse.linalg.LinearOperator`. `scipy.sparse.linalg.svds` does accept a `LinearOperator` as input but does not have randomized SVD as an available solver (only `lobpcg` and `arpack`).
