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


SERIES_FEATURES_LIST = ['WIND.SPD', 'TEMP', 'DEW.POINT']
SERIES_FEATURES_NAME2ID = {}
for fi, f in enumerate(SERIES_FEATURES_LIST):
    SERIES_FEATURES_NAME2ID[f] = fi


def load_noaa(dataset, target_features=('TEMP',), given_input_features=(),
              graph=None, with_node_meta=False, with_ts=False, node_meta_suffix=''):
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

    if with_node_meta:
        if node_meta_suffix == '':
            node_meta = np.load(os.path.join(datapath, 'node_meta.npy'))
            node_meta = (node_meta - np.mean(node_meta, axis=0)) / np.std(node_meta, axis=0)
        else: # node_meta with suffix only used for fixed ops, no normalization here since it's explicitly used in calculation
            node_meta = np.load(os.path.join(datapath, 'node_meta_{}.npy'.format(node_meta_suffix)))
        node_meta = torch.from_numpy(node_meta).contiguous().float()

    mesh_matrices_path = os.path.join(datapath, 'mesh_matrices{}.npz'.format(graph_suffix))
    if os.path.exists(mesh_matrices_path):
        mesh_matrices = np.load(mesh_matrices_path, allow_pickle=True)
        print('loaded matrices from {}'.format(mesh_matrices_path))
    else:
        mesh_matrices = None

    split = np.load(os.path.join(datapath, 'split.npz'))
    train_files, valid_files, test_files = split['train'], split['valid'], split['test']

    old_feature_indices = np.arange(len(SERIES_FEATURES_LIST))
    target_feature_indices = np.array([SERIES_FEATURES_NAME2ID[fn] for fn in target_features]).astype(np.int)
    given_feature_indices = np.array([SERIES_FEATURES_NAME2ID[fn] for fn in given_input_features]).astype(np.int)
    new_feature_indices = np.concatenate((given_feature_indices, target_feature_indices))

    # target_id = 1 # Temp
    def load_data_graphs(files, normalization=None):
        raw_datas = [np.load(os.path.join(datapath, filename)) for filename in files]

        # padding zero to max tru length
        # true_lengths = [x['vals'].shape[0] for x in raw_datas]
        # max_true_length = np.max(true_lengths)

        if normalization is None:
            flatten_vals = np.concatenate([x['vals'].reshape(-1, x['vals'].shape[-1]) for x in raw_datas], axis=0)
            flatten_vals = flatten_vals[:, new_feature_indices]
            normalization = {
                'mean': np.mean(flatten_vals, axis=0), 'std': np.std(flatten_vals, axis=0)
            }
        data_graphs = []
        for data in raw_datas:
            vals = (data['vals'][:, :, new_feature_indices] - normalization['mean']) / normalization['std']
            vals = torch.from_numpy(vals).contiguous().float()
            data_graph = Data(x=vals.transpose(0, 1), edge_index=edge_index)
            data_graph.target = vals.transpose(0, 1)[..., -len(target_feature_indices):]
            if with_ts:
                ts_raw = data['ts']
                ts = []
                for t_raw in ts_raw:
                    # ts.append((t_raw - ts_raw[0]) / np.timedelta64(1, 's'))
                    ts_secs = (t_raw - ts_raw[0]) / np.timedelta64(1, 's')
                    ts_year_ratio = ts_secs / (365 * 24 * 60 * 60.0)
                    ts_year_ratio = ts_year_ratio - np.fix(ts_year_ratio)
                    ts.append(ts_year_ratio)
                ts = torch.from_numpy(np.array(ts)).contiguous().float()
                data_graph.ts = ts
            if edge_dist is not None:
                data_graph.edge_dist = edge_dist
            if with_node_meta:
                data_graph.node_meta = node_meta
            data_graphs.append(data_graph)
        return data_graphs, normalization

    train_datalist, train_normalization = load_data_graphs(train_files, normalization=None)
    valid_datalist, _ = load_data_graphs(valid_files, normalization=train_normalization)
    test_datalist, _ = load_data_graphs(test_files, normalization=train_normalization)
    return train_datalist, valid_datalist, test_datalist, train_normalization, mesh_matrices
