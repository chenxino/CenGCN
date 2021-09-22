import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from math import ceil
from sklearn.preprocessing import normalize


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_data_new(dataset_str):
    dataset='data_new/'+dataset_str+'/'
    adj=sp.csr_matrix(np.loadtxt(dataset+'adj.txt',dtype=np.float32))
    features=sp.csr_matrix(np.loadtxt(dataset+'features.txt',dtype=np.float32))
    # features=sp.csr_matrix(np.eye(adj.shape[0]),dtype=np.float32)
    labels =np.loadtxt(dataset + 'labels.txt', dtype=np.int32)
    idx_train=np.loadtxt(dataset+'train.txt',dtype=np.int32)
    idx_val = np.loadtxt(dataset + 'val.txt', dtype=np.int32)
    idx_test = np.loadtxt(dataset + 'test.txt', dtype=np.int32)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_data_link(dataset_str):
    dataset='data_link/'+dataset_str+'/'
    adj=sp.csr_matrix(np.loadtxt(dataset+'adj_link.txt',dtype=np.float32))
    features=sp.csr_matrix(np.loadtxt(dataset+'features.txt',dtype=np.float32))
    negative_edges=np.loadtxt(dataset+'negative_edges.txt',dtype=np.int32)
    positive_edges = np.loadtxt(dataset + 'positive_edges.txt',dtype=np.int32)
    return adj,features,negative_edges,positive_edges

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocoo()

def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_eig_layer(adj,w,v,percent=0.02,p=1,q=-1):
    re_adj=np.asarray(adj.toarray(),dtype=np.float32)
    # w,v=np.linalg.eig(re_adj)
    sort_ord=np.argmax(np.abs(w))
    eig_cen=np.asarray(np.abs(v[:,sort_ord]),dtype=np.float32)
    center_node=np.sort(np.argsort(eig_cen)[-ceil(re_adj.shape[0]*percent):])

    eig_cen+=0.001
    eig_cen=eig_cen/np.min(eig_cen)

    wdegree=np.power(eig_cen,p)
    reci_degree=np.power(eig_cen,q)
    weight_adj=np.copy(re_adj)
    label_pro=label_propagation(re_adj,center_node)
    cut_adj=re_adj[:,center_node]
    cur_degree=np.sum(cut_adj,axis=1)
    for i in range(cut_adj.shape[0]):
        sort_label=list(reversed(np.argsort(label_pro[i,:])))
        for j in range(len(center_node)):
            cen=center_node[j]
            if cut_adj[i,j]>0:
                pos=sort_label.index(j)
                if pos<=cur_degree[i]-1:
                    weight_adj[i,cen]=weight_adj[i,cen]*wdegree[i]*wdegree[cen]
                    weight_adj[cen, i] = weight_adj[cen, i] * wdegree[i] * wdegree[cen]
                else:
                    weight_adj[i,cen] = weight_adj[i, cen] * reci_degree[i] * reci_degree[cen]
                    weight_adj[cen, i] = weight_adj[cen, i] * reci_degree[i] * reci_degree[cen]
    deg_one=np.array([1]*re_adj.shape[0])
    deg_one[center_node]=eig_cen[center_node]
    weight_adj+=np.diag(deg_one)
    adj_normalized = normalize_adj(weight_adj)
    return sparse_to_tuple(adj_normalized),center_node

def preprocess_eig(adj,w,v,percent=0.02,p=1,q=-1):
    re_adj=np.asarray(adj.toarray(),dtype=np.float32)#+np.eye(adj.shape[0],dtype=np.float32)
    # w,v=np.linalg.eig(re_adj)
    sort_ord=np.argmax(np.abs(w))
    eig_cen=np.asarray(np.abs(v[:,sort_ord]),dtype=np.float32)
    center_node=np.sort(np.argsort(eig_cen)[-ceil(re_adj.shape[0]*percent):])

    eig_cen+=0.001
    eig_cen=eig_cen/np.min(eig_cen)

    weight=np.array([1]*re_adj.shape[0])
    weight[center_node]=eig_cen[center_node]
    diag_d=np.diag(np.power(weight,p))
    adj_normalized = normalize_adj(np.dot(np.dot(diag_d,re_adj),diag_d)+np.diag(eig_cen))
    return sparse_to_tuple(adj_normalized),center_node

def preprocess_inv_eig(adj,w,v,percent=0.02,p=1,q=-1):
    re_adj=np.asarray(adj.toarray(),dtype=np.float32)#+np.eye(adj.shape[0],dtype=np.float32)
    w,v=np.linalg.eig(re_adj)
    sort_ord=np.argmax(np.abs(w))
    eig_cen=np.asarray(np.abs(v[:,sort_ord]),dtype=np.float32)
    center_node=np.sort(np.argsort(eig_cen)[-ceil(re_adj.shape[0]*percent):])

    eig_cen+=0.001
    eig_cen=eig_cen/np.min(eig_cen)

    weight=np.array([1]*re_adj.shape[0])*1.0
    weight[center_node]=eig_cen[center_node]
    diag_d=np.diag(np.power(weight,q))

    adj_normalized = normalize_adj(np.dot(np.dot(diag_d,re_adj),diag_d)+np.diag(eig_cen))
    return sparse_to_tuple(adj_normalized),center_node

def preprocess_inv_degree(adj,percent=0.02,p=1,q=-1):
    re_adj = np.asarray(adj.toarray(), dtype=np.float32)
    degree=np.sum(re_adj,axis=1)
    center_node=np.sort(np.argsort(degree)[-ceil(re_adj.shape[0]*percent):])
    weight=np.array([1]*re_adj.shape[0])*1.0
    weight[center_node]=degree[center_node]
    diag_d=np.diag(np.power(weight,q))
    adj_normalized = normalize_adj(np.dot(np.dot(diag_d,re_adj),diag_d)+np.diag(degree))
    return sparse_to_tuple(adj_normalized),center_node

def preprocess_degree(adj,percent=0.02,p=1,q=-1):
    re_adj = np.asarray(adj.toarray(), dtype=np.float32)
    degree=np.sum(re_adj,axis=1)
    center_node=np.sort(np.argsort(degree)[-ceil(re_adj.shape[0]*percent):])
    weight=np.array([1]*re_adj.shape[0])
    weight[center_node]=degree[center_node]
    diag_d=np.diag(np.power(weight,p))
    tran_adj=np.dot(np.dot(diag_d,re_adj),diag_d)+np.diag(degree)
    adj_normalized = normalize_adj(tran_adj)
    return sparse_to_tuple(adj_normalized),center_node

def preprocess_degree_layer(adj,percent=0.02,p=1,q=-1):
    re_adj = np.asarray(adj.toarray(), dtype=np.float32)
    degree=np.sum(re_adj,axis=1)

    wdegree=np.power(degree,p)
    reci_degree=np.power(degree,q)

    center_node=np.sort(np.argsort(degree)[-ceil(re_adj.shape[0]*percent):])
    weight_adj=np.copy(re_adj)
    label_pro=label_propagation(re_adj,center_node)
    cut_adj=re_adj[:,center_node]
    cur_degree=np.sum(cut_adj,axis=1)

    for i in range(cut_adj.shape[0]):
        sort_label=list(reversed(np.argsort(label_pro[i,:])))
        for j in range(len(center_node)):
            cen=center_node[j]
            if cut_adj[i,j]>0:
                pos=sort_label.index(j)
                if pos<=cur_degree[i]-1:
                    weight_adj[i,cen]=weight_adj[i,cen]*wdegree[i]*wdegree[cen]
                    weight_adj[cen, i] = weight_adj[cen, i] * wdegree[i] * wdegree[cen]
                else:
                    weight_adj[i,cen] = weight_adj[i, cen] * reci_degree[i] * reci_degree[cen]
                    weight_adj[cen, i] = weight_adj[cen, i] * reci_degree[i] * reci_degree[cen]

    deg_one=np.array([1]*re_adj.shape[0])
    deg_one[center_node]=degree[center_node]
    weight_adj+=np.diag(deg_one)
    adj_normalized = normalize_adj(weight_adj)
    return sparse_to_tuple(adj_normalized),center_node

def label_propagation(adj,center,iter=5):
    sort_c=center
    cen_label=np.eye(len(sort_c),dtype=np.float32)
    norm_adj=normalize(adj,axis=1)
    cen_adj=np.copy(adj)
    for c in sort_c:
        cen_adj[c,c]=1.0
    cen_adj=normalize(cen_adj,norm='l1',axis=1)
    cen_adj=cen_adj[:,sort_c]
    label_pro=np.dot(cen_adj,cen_label)
    label_pro=normalize(label_pro,norm='l1',axis=1)
    for _ in range(iter-1):
        label_pro=np.dot(norm_adj,label_pro)
        # label_pro = normalize(label_pro, norm='l1', axis=1)
    return label_pro

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def construct_feed_dict_link(features, support,placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
