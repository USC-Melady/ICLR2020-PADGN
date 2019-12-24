import os
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)
import itertools
import datetime
import pickle

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data, Dataset
import torch.nn.functional as F


SERIES_FEATURES_LIST = ['spd', 'time_in_day']
SERIES_FEATURES_NAME2ID = {}
for fi, f in enumerate(SERIES_FEATURES_LIST):
    SERIES_FEATURES_NAME2ID[f] = fi


def load_traffic(target_features=('spd',), given_input_features=('time_in_day',)):
    datapath = '../data/traffic/data'

    # load sensor graphs
    adj_mx_path = os.path.join(datapath, 'sensor_graph/adj_mx.pkl')
    with open(adj_mx_path, 'rb') as f:
        adj_mx_data = pickle.load(f, encoding='latin1')
    # adj distance mx
    sensor_ids, sensor_id_to_ind, adj_mx = adj_mx_data

    # adj connection mx
    A = (adj_mx > 0).astype(np.int)
    A = (A + A.transpose()) / 2
    edge_index = np.array([indices for indices in np.nonzero(A)])
    edge_index = torch.from_numpy(edge_index).contiguous().long()

    target_feature_indices = np.array([SERIES_FEATURES_NAME2ID[fn] for fn in target_features]).astype(np.int)
    given_feature_indices = np.array([SERIES_FEATURES_NAME2ID[fn] for fn in given_input_features]).astype(np.int)
    new_feature_indices = np.concatenate((given_feature_indices, target_feature_indices))

    def load_data_graphs(datafilepath, normalization=None):
        raw_data = np.load(datafilepath)
        vals = raw_data['x']
        if normalization is None:
            flatten_vals = vals.reshape((-1, vals.shape[-1]))
            flatten_vals = flatten_vals[:, new_feature_indices]
            normalization = {
                'mean': np.mean(flatten_vals, axis=0), 'std': np.std(flatten_vals, axis=0)
            }
        data_graphs = []
        for sample_i in range(vals.shape[0]):
            val_i = (vals[sample_i][..., new_feature_indices] - normalization['mean']) / normalization['std']
            val_i = torch.from_numpy(val_i).contiguous().float()
            data_graph = Data(x=val_i.transpose(0, 1), edge_index=edge_index)
            data_graph.target = val_i.transpose(0, 1)[..., -len(target_feature_indices):]
            data_graphs.append(data_graph)

        return data_graphs, normalization

    train_datalist, train_normalization = load_data_graphs(os.path.join(datapath, 'METR-LA/train.npz'),
                                                           normalization=None)
    valid_datalist, _ = load_data_graphs(os.path.join(datapath, 'METR-LA/val.npz'),
                                         normalization=train_normalization)
    test_datalist, _ = load_data_graphs(os.path.join(datapath, 'METR-LA/test.npz'),
                                        normalization=train_normalization)
    return train_datalist, valid_datalist, test_datalist, train_normalization