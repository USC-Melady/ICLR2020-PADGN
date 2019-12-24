# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import os
import sys
import multiprocessing as mp
from argparse import ArgumentParser
import functools

import numpy as np
import scipy as sp
import scipy.io as spio
import pandas as pd
import networkx as nx
import torch
import dgl

import numpy as np
from numpy import *
import numpy.fft as fft
import torch
import torch.utils.data
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from sklearn.neighbors import kneighbors_graph
import networkx as nx
import geopy.distance as distance

parser = ArgumentParser()
parser.add_argument('source_data', type=str, help='sst-0')
parser.add_argument('target_data', type=str, help='sst-0_4nn_42_250sample')
parser.add_argument('nodenum', type=int, help='250')
parser.add_argument('graph_seed', type=int, help='42')
args = parser.parse_args()



# %%
datadir = '../data/sst/{}'.format(args.source_data)
simdata = np.load(os.path.join(datadir, '000_simdata.npz'))
print(simdata['frames'].shape)
framedata = simdata['frames']
all_node_meta = np.load(os.path.join(datadir, 'node_meta.npy'))

N = simdata['frames'].shape[1]

# %%
from sklearn.neighbors import kneighbors_graph
import networkx as nx

def distance_latlong(x, y):
    return distance.distance(x,y).km

#### Build Graph (kNN)
def build_graph_kNN(X, num_neighbors):
    A = kneighbors_graph(X, num_neighbors, mode='connectivity', metric=distance_latlong)
    A = (A + A.transpose()) / 2  # symmetry
    G = nx.from_numpy_matrix(A.todense())
    edge_index = np.array([indices for indices in np.nonzero(A)])  # graph
    return {'A': A, 'edge_index': edge_index}
    

def sample_and_build_graph(pointl, num_neighbors, seed): # select pointl x pointl points
    random_state = np.random.RandomState(seed)
    x_ind = random_state.choice(N, pointl, replace=True)
    y_ind = random_state.choice(N, pointl, replace=True)
    X_ind = np.stack((x_ind, y_ind), axis=1).astype(np.int64)
    X = all_node_meta[x_ind, y_ind]
    built_graph = build_graph_kNN(X, num_neighbors)
    return {'X_ind': X_ind, 'X': X, 'edge_index': built_graph['edge_index']}


def plot_graph(X, edge_index):
    plt.figure(figsize=(5, 5))
    plt.plot(X[:, 1], X[:, 0], 'o')
    for ei in range(edge_index.shape[1]):
        plt.plot([X[edge_index[0, ei], 1], X[edge_index[1, ei], 1]], 
                 [X[edge_index[0, ei], 0], X[edge_index[1, ei], 0]], 'r-')


# built_graph = sample_and_build_graph(250, 4, seed=42)
built_graph = sample_and_build_graph(args.nodenum, 4, seed=args.graph_seed)
plot_graph(built_graph['X'], built_graph['edge_index'])


# %%
X_ind = built_graph['X_ind']


# %%
sampled_datadir = '../data/sst/{}'.format(args.target_data)
if not os.path.exists(sampled_datadir):
    os.makedirs(sampled_datadir)
    
# sample_num = 12
sample_num = len([x for x in os.listdir(datadir) if x.endswith('_simdata.npz')])
for t in range(sample_num):
    simdata_t = np.load(os.path.join(datadir, '{:03d}_simdata.npz'.format(t)))
    frames = simdata_t['frames']
    sampled_frames = frames[:, X_ind[:, 0], X_ind[:, 1]]
    resdict = {
        'frames': sampled_frames
    }
    np.savez(os.path.join(sampled_datadir, '{:03d}_simdata.npz'.format(t)), **resdict)
    print('Finished simulation {:03d}'.format(t))
    
sampled_edges_array = built_graph['edge_index'].astype(np.int64)
print(sampled_edges_array.shape)
np.save(os.path.join(sampled_datadir, 'edge_index.npy'), sampled_edges_array)
np.save(os.path.join(sampled_datadir, 'edge_index_4nn.npy'), sampled_edges_array)

np.save(os.path.join(sampled_datadir, 'node_meta.npy'), built_graph['X'])


# %%
train_list = np.array(['{:03d}_simdata.npz'.format(k) for k in range(0, 12)])
valid_list = np.array(['{:03d}_simdata.npz'.format(k) for k in range(12, 24)])
test_list = np.array(['{:03d}_simdata.npz'.format(k) for k in range(12, 24)])

np.savez(os.path.join(sampled_datadir, 'split.npz'), train=train_list, valid=valid_list, test=test_list)


# %%
train_list = np.array(['{:03d}_simdata.npz'.format(k) for k in range(0, 12)])
valid_list = np.array(['{:03d}_simdata.npz'.format(k) for k in range(12, 24)])
test_list = np.array(['{:03d}_simdata.npz'.format(k) for k in range(12, 24)])

np.savez(os.path.join(sampled_datadir, 'small_split.npz'), train=train_list, valid=valid_list, test=test_list)


import os
import sys
import multiprocessing as mp
import functools

import numpy as np
import scipy as sp
import scipy.io as spio
import scipy.sparse as sparse
import pandas as pd
import networkx as nx
import torch
import dgl
import pymap3d
import utm

import numpy as np
from numpy import *
import numpy.fft as fft
import torch
import torch.utils.data
from torch.autograd import Variable

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

import igl
from meshplot import plot, subplot, interact


# %%
datadir = sampled_datadir

edge_index = np.load(os.path.join(datadir, 'edge_index_4nn.npy'))
node_meta_lla = np.load(os.path.join(datadir, 'node_meta.npy'))
print(edge_index.shape, node_meta_lla.shape)


# %%
# get 3d node_meta points, used for building mesh
node_meta_ecef = pymap3d.geodetic2ecef(node_meta_lla[:, 0], node_meta_lla[:, 1], np.zeros_like(node_meta_lla[:, 0]))
node_meta_ecef = np.stack(node_meta_ecef, axis=-1)
node_meta_ecef /= 1000
node_meta_ecef


# %%
# get enu coords, used for sorting neighbors clock-wisely and for triangulation
node_meta_utm = []
for lla_t in node_meta_lla:
    node_meta_utm.append(utm.from_latlon(lla_t[0], lla_t[1])[:2])
node_meta_utm = np.array(node_meta_utm, dtype=np.float32) / 1000
node_meta_utm


# %%
np.save(os.path.join(datadir, 'node_meta_utm.npy'), node_meta_utm)
np.save(os.path.join(datadir, 'node_meta_ecef.npy'), node_meta_ecef)


# %%
def cross_prod_k(x, y):
    return x[0] * y [1] - x[1] * y[0]

def same_or_right(x, y):
    k = cross_prod_k(x, y)
    inner = np.dot(x, y)
    return (k < 0) or (k == 0 and inner > 0)

def make_vec_comparator(u):
    '''
    return a function for direction comparision: x < y if x->y is clockwise
    '''
    def comparator(x, y):
        x, y = x[1], y[1]
        sor_x, sor_y = same_or_right(u, x), same_or_right(u, y)
        cpk_xy = cross_prod_k(x, y)
        inner_xy = np.dot(x, y)
        if ((sor_x and not sor_y) or ((sor_x == sor_y) and cpk_xy < 0)):
            return -1
        elif (cpk_xy == 0 and inner_xy > 0):
            return 0
        else:
            return 1
    return comparator

def build_mesh(edge_index, node_meta):
    '''
    build mesh-like graph for arbitrary graph: for each node, build a new graph via connecting its neighbors clockwisely
    assume input graph is symmetric
    '''
    edge_index = edge_index.transpose(1, 0) # E x 2
    node_coords = node_meta[:, :2] # N x 2
    triangles = []
    node_num = node_coords.shape[0]
    edge_num = edge_index.shape[0]
    
    for t in range(node_num):
        dst_t_edge_indices = np.where(edge_index[:, 1] == t)[0]
        dst_t_edges = edge_index[dst_t_edge_indices, :]
        t_neighbors = dst_t_edges[:, 0]
        
        dst_t_edge_vecs = node_meta[dst_t_edges[:, 1]] - node_meta[dst_t_edges[:, 0]] # E x 2
        dst_t_edge_vecs_tosort = []
        for ei in range(dst_t_edge_vecs.shape[0]):
            dst_t_edge_vecs_tosort.append((ei, dst_t_edge_vecs[ei]))
        dst_t_edge_vecs_sorted = sorted(dst_t_edge_vecs_tosort, key=functools.cmp_to_key(make_vec_comparator(dst_t_edge_vecs[0])))
        dst_t_edge_i_sorted = [x[0] for x in dst_t_edge_vecs_sorted]
        for ei in range(len(dst_t_edge_i_sorted) - 1):
            currp = t_neighbors[dst_t_edge_i_sorted[ei]]
            nextp = t_neighbors[dst_t_edge_i_sorted[ei + 1]]
            triangles.append((t, currp, nextp))
        currp = nextp
        nextp = t_neighbors[dst_t_edge_i_sorted[0]]
        triangles.append((t, currp, nextp))
        
#     v = np.concatenate((node_coords, np.zeros((node_num, 1))), axis=-1) # N x 3, lastdim=0
    v = node_meta
    f = np.array(triangles, dtype=np.int64)
#     f = np.unique(np.sort(f, axis=1), axis=0)
    return {'v': v, 'f': f}

res = build_mesh(edge_index, node_meta=node_meta_utm)
res['v'] = node_meta_ecef
print(res['v'].shape, res['f'].shape)


# %%
for t in range(191):
    print(t, res['f'][res['f'][:, 0] == t].shape[0], edge_index[:, edge_index[1, :] == t].shape[1])
    assert(res['f'][res['f'][:, 0] == t].shape[0] == edge_index[:, edge_index[1, :] == t].shape[1])


# %%
def calc_alpha(kc, ic, jc):
    v_ki = ic - kc
    v_kj = jc - kc
    v_ki_norm, v_kj_norm = np.linalg.norm(v_ki), np.linalg.norm(v_kj)
    if (v_ki_norm < 1e-6 and v_kj_norm < 1e-6):
        sinalpha = 0. # alpha=0
        cosalpha = 1.
    elif (v_ki_norm < 1e-6 or v_kj_norm < 1e-6):
        sinalpha = 1. # alpha = pi / 2
        cosalpha = 0.
    else:
        sinalpha = np.linalg.norm(np.cross(v_ki, v_kj)) / (v_ki_norm * v_kj_norm)
        cosalpha = np.linalg.norm(np.dot(v_ki, v_kj)) / (v_ki_norm * v_kj_norm)
    return sinalpha, cosalpha

def cotmatrix_firstvonly(vertices, faces):
    '''
    cot will only contribute to first vertex in face
    '''
    v_num, f_num = vertices.shape[0], faces.shape[0]
    cotmatrix_value_dict = {}
    for face in faces:
        edge_ps = [(face[0], face[1], face[2]), (face[0], face[2], face[1])]
        for edge_p in edge_ps:
            i, j, k = edge_p
            if (i, j) not in cotmatrix_value_dict.keys():
                cotmatrix_value_dict[(i, j)] = 0.
            if (i, i) not in cotmatrix_value_dict.keys():
                cotmatrix_value_dict[(i, i)] = 0.
            sinalpha, cosalpha = calc_alpha(vertices[k], vertices[i], vertices[j])
            if np.abs(sinalpha) > 0:
                cotalpha = cosalpha / sinalpha
            else:
                cotalpha = 1e6 # truncate inf to 1e6
            cotmatrix_value_dict[(i, j)] += cotalpha
            cotmatrix_value_dict[(i, i)] -= cotalpha
    xs, ys, vs = [], [], []
    for k, v in cotmatrix_value_dict.items():
        xs.append(k[0])
        ys.append(k[1])
        vs.append(v)
    xs = np.array(xs, dtype=np.int64)
    ys = np.array(ys, dtype=np.int64)
    vs = np.array(vs, dtype=np.float32)
    L = sparse.csc_matrix((vs, (xs, ys)), shape=(v_num, v_num))
#     return L
    degrees = np.array([np.sum((faces[:, 0] == t).astype(np.float32)) for t in range(v_num)], dtype=np.float32)
    print(degrees)
    return sparse.csc_matrix(L.toarray() / (-np.diag(L.toarray()).reshape(-1, 1)) * degrees.reshape(-1, 1))


# %%
# v = np.array([
#     [0, 0, 0],
#     [1, 0, 0],
#     [1, 1, 0],
#     [2, 1, 0]
# ], dtype=np.float32)
# f = np.array([
#     [0, 1, 2],
#     [1, 3, 2]
# ], dtype=np.int64)
# u = np.array([[0, 0, 1, 1]], dtype=np.float32).reshape(4, 1)

def construct_mesh_matrices(vertices, faces, lat):
    v_num, f_num = vertices.shape[0], faces.shape[0]
    G = igl.grad(vertices, faces)
#     L = igl.cotmatrix(vertices, faces)
    L = cotmatrix_firstvonly(vertices, faces)
    A = igl.doublearea(vertices, faces)
    N = igl.per_face_normals(vertices, faces, vertices)
#     XN = np.array([1, 0, 0], dtype=np.float32)
#     YN = np.array([0, 1, 0], dtype=np.float32)
    YN = np.reshape(G.dot(lat), [f_num, 3], order='F')
    YN = YN / (np.linalg.norm(YN, axis=1)[:, np.newaxis]+1e-6)
    XN = np.cross(YN, N)
    i = faces[:, 0].ravel() # each triangle only belongs to the first vertex!
    j = np.arange(f_num)
    one = np.ones(f_num)
    adj = sparse.csc_matrix((one, (i, j)), shape=(v_num, f_num))
    tot_area = adj.dot(A)
    norm_area = A.ravel() / np.squeeze(tot_area[i] + 1e-6)
    F2V = sparse.csc_matrix((norm_area, (i, j)), shape=(v_num, f_num))
    return {
        'G': G, 'L': L, 'A': A, 'XN': XN, 'YN': YN, 'F2V': F2V, 'N': N
    }

mesh_matrices = construct_mesh_matrices(res['v'], res['f'], lat=node_meta_lla[:, 0])
np.savez(os.path.join(datadir, 'mesh_matrices.npz'), **mesh_matrices)
np.savez(os.path.join(datadir, 'mesh_matrices_4nn.npz'), **mesh_matrices)


# %%
res['v']


# %%
for k, v in mesh_matrices.items():
    print(k, v.shape)


# %%
x = np.load(os.path.join(datadir, 'mesh_matrices_4nn.npz'))
x['F2V'].item().toarray().sum(axis=1)


# %%
# Build Delaunay Triangulation and graph

from scipy.spatial import Delaunay


tri = Delaunay(node_meta_utm)
plt.figure(figsize=(10, 30))
plt.triplot(node_meta_utm[:,0], node_meta_utm[:,1], tri.simplices.copy())


# %%
edge_index_de = []
indices, indptr = tri.vertex_neighbor_vertices
for t in range(node_meta_utm.shape[0]):
    neighbors_t = indptr[indices[t]:indices[t+1]]
    for nt in neighbors_t:
        edge_index_de.append((t, nt))
edge_index_de = np.array(edge_index_de, dtype=np.int64).transpose(1, 0)
np.save(os.path.join(datadir, 'edge_index_de.npy'), edge_index_de)

tri.simplices

def construct_mesh_matrices_de(vertices, faces, lat):
    v_num, f_num = vertices.shape[0], faces.shape[0]
    G = igl.grad(vertices, faces)
    L = igl.cotmatrix(vertices, faces)
    A = igl.doublearea(vertices, faces)
    N = igl.per_face_normals(vertices, faces, vertices)
    YN = np.reshape(G.dot(lat), [f_num, 3], order='F')
    YN = YN / (np.linalg.norm(YN, axis=1)[:, np.newaxis]+1e-6)
    XN = np.cross(YN, N)
    i = faces.ravel()
    j = np.arange(f_num).repeat(3)
    one = np.ones(f_num * 3)
    adj = sparse.csc_matrix((one, (i, j)), shape=(v_num, f_num))
    tot_area = adj.dot(A)
    norm_area = A.ravel().repeat(3) / np.squeeze(tot_area[i] + 1e-6)
    F2V = sparse.csc_matrix((norm_area, (i, j)), shape=(v_num, f_num))
    return {
        'G': G, 'L': L, 'A': A, 'XN': XN, 'YN': YN, 'F2V': F2V
    }

mesh_de = construct_mesh_matrices_de(vertices=node_meta_ecef,
                                    faces=tri.simplices.astype(np.int64), lat=node_meta_lla[:, 0])
np.savez(os.path.join(datadir, 'mesh_matrices_de.npz'), **mesh_de)


# %%
mesh_de['F2V'].toarray()