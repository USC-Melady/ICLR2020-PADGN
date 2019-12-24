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


def load_sst(datasetname, graph=None):
    # datapath = '../data/noaa'
    datapath = os.path.join('../data/sst', datasetname)

    if graph is None:
        graph_suffix = ''
    else:
        graph_suffix = '_{}'.format(graph)
    edge_index = np.load(os.path.join(datapath, 'edge_index{}.npy'.format(graph_suffix)))
    edge_index = torch.from_numpy(edge_index).contiguous().long()

    # distpath = os.path.join(datapath, 'edge_dist{}.npy'.format(graph_suffix))
    # if os.path.exists(distpath):
    #     edge_dist = np.load(distpath).reshape(-1, 1)
    #     edge_dist = torch.from_numpy(edge_dist).contiguous().float()
    # else:
    #     edge_dist = None

    node_meta = np.load(os.path.join(datapath, 'node_meta.npy'))
    # node_meta = (node_meta - np.mean(node_meta, axis=0)) / np.std(node_meta, axis=0)
    node_meta = torch.from_numpy(node_meta).contiguous().float()

    mesh_matrices_path = os.path.join(datapath, 'mesh_matrices{}.npz'.format(graph_suffix))
    print(mesh_matrices_path)
    if os.path.exists(mesh_matrices_path):
        mesh_matrices = np.load(mesh_matrices_path, allow_pickle=True)
        print('loaded matrices from {}'.format(mesh_matrices_path))
    else:
        mesh_matrices = None

    # split = np.load(os.path.join(datapath, 'split.npz'))
    split = np.load(os.path.join(datapath, 'small_split.npz'))
    train_files, valid_files, test_files = split['train'], split['valid'], split['test']

    # target_id = 1 # Temp
    def load_data_graphs(files, normalization=None):
        raw_datas = [np.load(os.path.join(datapath, filename)) for filename in files]
        frame_datas = [d['frames'][:, :, np.newaxis] for d in raw_datas]
        if normalization is None:
            flatten_vals = np.stack(frame_datas, axis=0).reshape(-1, frame_datas[0].shape[-1])
            normalization = {
                'mean': np.mean(flatten_vals, axis=0), 'std': np.std(flatten_vals, axis=0)
            }
        data_graphs = []
        for data in frame_datas:
            vals = (data - normalization['mean']) / normalization['std']
            # vals = data
            vals = torch.from_numpy(vals).contiguous().float()
            data_graph = Data(x=vals.transpose(0, 1), edge_index=edge_index)
            data_graph.target = vals.transpose(0, 1)
            data_graph.node_meta = node_meta
            data_graphs.append(data_graph)
        return data_graphs, normalization

    train_datalist, train_normalization = load_data_graphs(train_files, normalization=None)
    valid_datalist, _ = load_data_graphs(valid_files, normalization=train_normalization)
    test_datalist, _ = load_data_graphs(test_files, normalization=train_normalization)
    # valid_datalist, _ = load_data_graphs(train_files, normalization=train_normalization)
    # test_datalist, _ = load_data_graphs(train_files, normalization=train_normalization)
    return train_datalist, valid_datalist, test_datalist, train_normalization, mesh_matrices
