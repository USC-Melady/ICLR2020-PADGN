import os
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)
import itertools
import datetime

import numpy as np
import torch
from torch_geometric.data import Data


def load_synthetic(dataset, graph=None):
    # datapath = '../data/noaa'
    datapath = os.path.join('../data', dataset)

    if graph is None:
        graph_suffix = ''
    else:
        graph_suffix = '_{}'.format(graph)
    edge_index = np.load(os.path.join(datapath, 'edge_index{}.npy'.format(graph_suffix)))
    edge_index = torch.from_numpy(edge_index).contiguous().long()

    distpath = os.path.join(datapath, 'edge_dist{}.npy'.format(graph_suffix))
    if os.path.exists(distpath):
        edge_dist = np.load(distpath).reshape(-1, 1)
        edge_dist = torch.from_numpy(edge_dist).contiguous().float()
    else:
        edge_dist = None

    # target_id = 1 # Temp
    def load_data_graphs(normalization=None):
        raw_datas = np.load(os.path.join(datapath, 'diffusion_sampled.npy'))[:, :30, :, :]

        if normalization is None:
            flatten_vals = raw_datas.reshape(-1, raw_datas.shape[-1])
            normalization = {
                'mean': np.mean(flatten_vals, axis=0), 'std': np.std(flatten_vals, axis=0)
            }
        data_graphs = []
        for data in raw_datas:
            vals = (data - normalization['mean']) / normalization['std']
            vals = torch.from_numpy(vals).contiguous().float()
            data_graph = Data(x=vals.transpose(0, 1), edge_index=edge_index)
            if edge_dist is not None:
                data_graph.edge_dist = edge_dist
            data_graph.target = vals.transpose(0, 1)
            data_graphs.append(data_graph)
        return data_graphs, normalization

    train_datalist, train_normalization = load_data_graphs(normalization=None)
    valid_datalist, _ = load_data_graphs(normalization=train_normalization)
    test_datalist, _ = load_data_graphs(normalization=train_normalization)
    return train_datalist, valid_datalist, test_datalist, train_normalization
