import pandas as pd
import geopandas
from sklearn.neighbors import kneighbors_graph

import torch
import numpy as np
import networkx as nx

from torch_geometric.data import Data


def get_xy_from_ind(xy_array, ind):
    return xy_array[ind]

def get_graph(xy_array, num_neighbors=4, one_component=True):
    
    adj_dict = {}
    A = kneighbors_graph(xy_array, num_neighbors, mode='connectivity')    # anti-symmetry
    A = (A + A.transpose()) / 2    # symmetry
    A = (A>0)*1.0

    G = nx.from_numpy_matrix(A.todense())
    if one_component:
        assert nx.number_connected_components(G)==1
    for n in range(G.number_of_nodes()):
        adj_dict[n] = [nn for nn in G.neighbors(n)]

    edge_index = np.array([indices for indices in np.nonzero(A)])
    num_nodes = A.shape[0]
    num_edges = edge_index.shape[1]

    start_pts = get_xy_from_ind(xy_array, edge_index[0])
    end_pts = get_xy_from_ind(xy_array, edge_index[1])

    return A.todense(), adj_dict, start_pts, end_pts

def edge_to_two_points(edge, F, X_coord):
    """
    edge: [sender, receiver]
    F: CustomFunction
    X_coord: coordinates
    """
    sender, receiver = edge
    s_coord = X_coord[sender]
    s_value = F(s_coord[0], s_coord[1])
    r_coord = X_coord[receiver]
    r_value = F(r_coord[0], r_coord[1])
    
    return s_coord, r_coord, s_value, r_value

def decompose_graph(graph):
    # graph: torch_geometric.data.data.Data
    # TODO: make it more robust
    x, edge_index, edge_attr, global_attr = None, None, None, None
    for key in graph.keys:
        if key=="x":
            x = graph.x
        elif key=="edge_index":
            edge_index = graph.edge_index
        elif key=="edge_attr":
            edge_attr = graph.edge_attr
        elif key=="global_attr":
            global_attr = graph.global_attr
        else:
            pass
    return (x, edge_index, edge_attr, global_attr)


def graph_concat(graph1, graph2, 
                 node_cat=True, edge_cat=True, global_cat=False):
    """
    Args:
        graph1: torch_geometric.data.data.Data
        graph2: torch_geometric.data.data.Data
        node_cat: True if concat node_attr
        edge_cat: True if concat edge_attr
        global_cat: True if concat global_attr
    Return:
        new graph: concat(graph1, graph2)
    """
    # graph2 attr is used for attr that is not concated.
    _x = graph2.x
    _edge_attr = graph2.edge_attr
    _global_attr = graph2.global_attr
    _edge_index = graph2.edge_index
    
    if node_cat:
        try:
            _x = torch.cat([graph1.x, graph2.x], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'x' key.")
    
    if edge_cat:
        try:
            _edge_attr = torch.cat([graph1.edge_attr, graph2.edge_attr], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'edge_attr' key.")
            
    if global_cat:
        try:
            _global_attr = torch.cat([graph1.global_attr, graph2.global_attr], dim=-1)
        except:
            raise ValueError("Both graph1 and graph2 should have 'global_attr' key.")

    ret = Data(x=_x, edge_attr=_edge_attr, edge_index=_edge_index)
    ret.global_attr = _global_attr
    
    return ret


def copy_geometric_data(graph):
    """return a copy of torch_geometric.data.data.Data
    This function should be carefully used based on
    which keys in a given graph.
    """
    node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)
    
    ret = Data(x=node_attr, edge_index=edge_index, edge_attr=edge_attr)
    ret.global_attr = global_attr
    
    return ret

def normalize_vector(vec):
    '''return normalized vector'''
    norm=np.linalg.norm(vec, ord=2)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return vec/norm

def get_projection(vec1, vec2):
    '''projection of vec1 on vec2'''
    vec2 = normalize_vector(vec2)
    return np.dot(vec1, vec2)

def edge_to_directional_vector(edge_index, X):
    spts, ret = [], []
    for edge in edge_index.transpose():
        sind, rind = edge
        s_coord = X[sind, :]
        r_coord = X[rind, :]
        ret.append([r_coord[0]-s_coord[0], r_coord[1]-s_coord[1]])
        spts.append(s_coord)
    return np.array(spts), np.array(ret)

def sample_edges(edge_index, nb_edges=10):
    nb_total_edges = edge_index.shape[1]
    
    if nb_edges > nb_total_edges:
        raise ValueError("nb_edges ({}) should be less than nb_total_edges ({})".format(nb_edges, nb_total_edges))
    
    else:
        indices = np.random.choice(nb_total_edges, nb_edges, replace=False)
        return edge_index[:, indices], indices
    
def sample_xy(xx, yy, nb_points=100):
    '''sample random coordinates in meshgrid
    return np.array shape=(nb_points, 2)
    '''
    assert xx.shape == yy.shape
    
    nx, ny = xx.shape
    nb_total_points = nx * ny
    if nb_points > nb_total_points:
        raise ValueError("nb_points ({}) should be less than nb_total_points ({})".format(nb_points, nb_total_points))
    
    else:
        indices = np.random.choice(nb_total_points, nb_points, replace=False)
        row_ind, col_ind = np.unravel_index(indices, xx.shape)
        return xx[0, col_ind], yy[row_ind, 0]
