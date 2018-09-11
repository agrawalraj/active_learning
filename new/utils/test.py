import numpy as np
from scipy.linalg import ldl
import operator as op
from sksparse.cholmod import cholesky


def permute(mat, perm):
    m = mat.copy()
    m = m[perm]
    m = m[:, perm]
    return m


def inv_perm(permutation):
    return [i for i, j in sorted(enumerate(permutation), key=op.itemgetter(1))]


def prec2adj(prec, node_order):
    p = prec.shape[0]
    prec = permute(prec, node_order)
    u, d, perm_ = ldl(prec, lower=False)
    u = u[perm_]
    d = d[perm_]
    u[np.isclose(u, 0)] = 0
    inv_node_order = inv_perm(node_order)
    adj_mat = np.eye(p) - permute(u, inv_node_order)
    omega = np.linalg.inv(permute(d, inv_node_order))
    adj_mat[np.isclose(adj_mat, 0)] = 0
    return adj_mat, omega


if __name__ == '__main__':
    adj1 = np.array([
        [0, 2, 3],
        [0, 0, 5],
        [0, 0, 0]
    ])
    prec1 = (np.eye(3) - adj1) @ (np.eye(3) - adj1).T
    prec = np.array([
        [14, 13, -3],
        [13, 26, -5],
        [-3, -5, 1],
    ])
    node_order = [0, 1, 2]
    prec = permute(prec, [2, 1, 0])
    adj, omega = prec2adj(prec, node_order)

    from scipy import sparse
    factor = cholesky(sparse.csc_matrix(prec))
    l, d = factor.L_D()
    l = l.todense()
    d = d.todense()
