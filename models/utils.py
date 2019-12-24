from __future__ import division
from __future__ import print_function

import os
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)

import numpy as np
from torch_geometric.data import Data, Batch
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence
from torch_scatter import scatter_add


def replace_graph(data, **kwargs):
    '''
    Create a new Data filled with kwargs, use features not listed in kwargs from data
    :param data: a torch_geometric.data.Data struct. They are not cloned!
    :param kwargs: features to replace. They are not cloned.
    :return: a new Data structure filled with kwargs and other features from data
    '''
    new_data_dict = {}
    for k, v in data:
        if k not in kwargs.keys():
            new_data_dict[k] = v
    for k, v in kwargs.items():
        new_data_dict[k] = v
    new_data = Data.from_dict(new_data_dict)
    return new_data


def pop_graph(data, popkey):
    '''
    Get data with popkey poped.
    :param data:
    :param popkey:
    :return: A new Data structure with popkey poped
    '''
    new_data_dict = {k: v for k, v in data}
    new_data_dict.pop(popkey)
    return Data.from_dict(new_data_dict)


def unbatch_node_feature_mat(node_feature, batch_ids):
    assert node_feature.shape[0] == batch_ids.shape[0]
    split_sizes = scatter_add(torch.ones_like(batch_ids), batch_ids)
    split_sizes = split_sizes.cpu().data.numpy().tolist()
    return list(torch.split(node_feature, split_sizes, dim=0))


def unbatch_node_feature_mat_tonumpy(node_feature, batch_ids):
    matlist = unbatch_node_feature_mat(node_feature, batch_ids)
    matlist = [x.data.numpy() for x in matlist]
    return matlist


def unbatch_node_feature(batched_graph, feature_name, batch_ids):
    node_feature = getattr(batched_graph, feature_name)
    return unbatch_node_feature_mat(node_feature, batch_ids)


def make_mlp(input_dim, hidden_dim, output_dim, layer_num, activation='ReLU',
             final_activation=False, batchnorm=None):
    # assert layer_num >= 2
    activation_layer_func = getattr(nn, activation)
    mlp_layers = [nn.Linear(input_dim, hidden_dim),]
    # if batchnorm == 'LayerNorm':
    #     mlp_layers.append(nn.LayerNorm(hidden_dim))
    # elif batchnorm == 'BatchNorm':
    #     mlp_layers.append(nn.BatchNorm1d(hidden_dim))
    mlp_layers.append(activation_layer_func())
    if layer_num > 1:
        for li in range(1, layer_num - 1):
            mlp_layers.append(nn.Linear(hidden_dim, hidden_dim))
            # if batchnorm == 'LayerNorm':
            #     mlp_layers.append(nn.LayerNorm(hidden_dim))
            # elif batchnorm == 'BatchNorm':
            #     mlp_layers.append(nn.BatchNorm1d(hidden_dim))
            mlp_layers.append(activation_layer_func())
    mlp_layers.append(nn.Linear(hidden_dim, output_dim))
    if final_activation:
        if batchnorm == 'LayerNorm':
            mlp_layers.append(nn.LayerNorm(hidden_dim))
        elif batchnorm == 'BatchNorm':
            mlp_layers.append(nn.BatchNorm1d(hidden_dim))
        mlp_layers.append(activation_layer_func())
    return nn.Sequential(*mlp_layers)


def collate_fn_withpad(data_list):
    '''
    Modified based on PyTorch-Geometric's implementation
    :param data_list:
    :return:
    '''
    keys = [set(data.keys) for data in data_list]
    keys = list(set.union(*keys))
    assert 'batch' not in keys

    batch = Batch()

    for key in keys:
        batch[key] = []
    batch.batch = []

    cumsum = 0
    for i, data in enumerate(data_list):
        num_nodes = data.num_nodes
        batch.batch.append(torch.full((num_nodes,), i, dtype=torch.long))
        for key in data.keys:
            item = data[key]
            item = item + cumsum if data.__cumsum__(key, item) else item
            batch[key].append(item)
        cumsum += num_nodes

    for key in keys:
        item = batch[key][0]
        if torch.is_tensor(item):
            if (len(item.shape) == 3):
                tlens = [x.shape[1] for x in batch[key]]
                maxtlens = np.max(tlens)
                to_cat = []
                for x in batch[key]:
                    to_cat.append(torch.cat([x, x.new_zeros(x.shape[0], maxtlens - x.shape[1], x.shape[2])], dim=1))
                batch[key] = torch.cat(to_cat, dim=0)
                if 'tlens' not in batch.keys:
                    batch['tlens'] = item.new_tensor(tlens, dtype=torch.long)
            else:
                batch[key] = torch.cat(
                    batch[key], dim=data_list[0].__cat_dim__(key, item))
        elif isinstance(item, int) or isinstance(item, float):
            batch[key] = torch.tensor(batch[key])
        else:
            raise ValueError('Unsupported attribute type.')
    batch.batch = torch.cat(batch.batch, dim=-1)
    return batch.contiguous()


class DataLoaderWithPad(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=True, **kwargs):
        super(DataLoaderWithPad, self).__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn_withpad,
            **kwargs)


class SeriesModel(nn.Module):
    def __init__(self, input_dim, output_dim, input_frame_num, skip_first_frames_num, is_recurrent, detach_last_step=False):
        '''
        Abstract class for series prediction
        :param input_dim: dimension for input features
        :param output_dim: dimension for output features
        :param input_frame_num: number of frames required as input when predict next frame
                                PDGN requires 2 (when only use 1-step time gradient) or more
                                RNN models require 1 input frame to predict next frame
        :param skip_first_frames_num: skipped number of frames in the beginning (will use ground truth instead)
            to make sure different models are predicting the same part of ground truth
        :param is_recurrent: whether the model is a recurrent model (requiring hidden vectors)

        Given input_frame_num = m, skip_first_frames_num = n, a sequence with a length T,
        output sequence also has length T:
        1. m >= n:
            1). 0 <= ti < m: output[ti] = input[ti]
            2). m <= ti < T: output[ti] = model((ground_truth or output, sampled via scheduler)[ti-input_frame_num:ti])
        2. m < n:
            1). 0 <= ti < m: output[ti] = input[ti]
            2). m <= ti < n: output[ti] = model(ground_truth[ti-input_frame_num:ti])
            3). n <= ti    : output[ti] = model((ground_truth or output, sampled via scheduler)[ti-input_frame_num:ti])
        '''
        super(SeriesModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_frame_num = input_frame_num
        self.skip_first_frames_num = skip_first_frames_num
        self.is_recurrent = is_recurrent
        self.detach_last_step = detach_last_step

    def forward_onestep(self, data):
        raise NotImplementedError()

    def forward(self, data, train_sample_prob=1, return_features=False):
        '''
        Assume that features to predict are at the end of feature vectors
        :param data: Data from PyTorch.Geometric, data.x->[N, T, F], last output_dim dimensions corresponde to predicted features
        :param predict_stepnum: number of steps to predict without feeding any ground truth data in the middle
        :return: predicted sequence, [N, T, output_dim]
        '''
        length = data.x.shape[1]
        valid_lengths = data.tlens
        input_frame_num = self.input_frame_num
        if self.training and (train_sample_prob == 1) and (not self.is_recurrent):
            # unwarp only for (1. is training, 2. always sample, 3. not recurrent)
            # each unwarpped sample is (input_frames, next_output_frame)
            unwarped_input_graphs = []
            for t in range(length - input_frame_num - 1 + 1):
                unwarped_input_graph = replace_graph(data, x=data.x[:, t:t+input_frame_num, :],
                                                     graph_batch=data.batch)
                unwarped_input_graph = pop_graph(unwarped_input_graph, 'batch')
                unwarped_input_graphs.append(unwarped_input_graph)
            unwarped_input_graph_batch = Batch.from_data_list(unwarped_input_graphs)
            output_graph_batch = self.forward_onestep(unwarped_input_graph_batch)
            output_onesteps = unbatch_node_feature(output_graph_batch, 'x', unwarped_input_graph_batch.batch)
            output_onestep = torch.stack(output_onesteps, dim=1)
            output_seq = torch.cat((data.target[:, :input_frame_num, :], output_onestep), dim=1)
            to_return = output_seq
        else:
            output_frames = []
            feature_frames = {
                'node': [], 'edge': [],
                'gradient_weight': [], 'laplacian_weight': []
            }
            # collect input_frame_num frames (or there are not enough frames for prediction)
            for input_i in range(input_frame_num):
                output_frames.append(data.target[:, input_i, :])
            input_graph = replace_graph(data, graph_batch=data.batch)
            for t in range(input_frame_num, length):
                given_part = data.x[:, t-input_frame_num:t, :-self.output_dim]

                predict_part_list = []
                for tsub in range(t - input_frame_num, t):
                    # use ground truth within skip_first_frames_num, to keep that all models have same input
                    if tsub < self.skip_first_frames_num:
                        sample_from_gt = True
                    else:
                        if self.training:
                            random_float = np.random.uniform(0.0, 1.0)
                            sample_from_gt = (random_float < train_sample_prob)
                        else:
                            sample_from_gt = False
                    if sample_from_gt:
                        predict_part_list.append(data.target[:, tsub, :])
                    else:
                        predict_part_list.append(output_frames[tsub])
                predict_part = torch.stack(predict_part_list, dim=1)

                # backup original 'batch' to 'graph_batch'
                if self.detach_last_step:
                    input_graph = replace_graph(input_graph,
                                                x=torch.cat((given_part, predict_part), dim=-1).detach(),
                                                graph_batch=input_graph.batch)
                else:
                    input_graph = replace_graph(input_graph,
                                                x=torch.cat((given_part, predict_part), dim=-1),
                                                graph_batch=input_graph.batch)
                # "1-step warp" means no warp. But for compatibility, we need do 1-batch and remove the true 'batch' attribute
                input_graph = pop_graph(input_graph, 'batch')
                if return_features:
                    output_graph, DG_output_data = self.forward_onestep(Batch.from_data_list([input_graph, ]), return_features=True)
                else:
                    output_graph = self.forward_onestep(Batch.from_data_list([input_graph,]))
                # recover original 'batch' from 'graph_batch'
                output_graph = replace_graph(output_graph, batch=output_graph.graph_batch)
                # remove 'graph_batch'
                output_graph = pop_graph(output_graph, 'graph_batch')
                output_frames.append(output_graph.x)
                if return_features:
                    feature_frames['node'].append(DG_output_data.x)
                    if DG_output_data.edge_attr is None:
                        feature_frames['edge'].append(DG_output_data.x.new_zeros(DG_output_data.edge_index.shape[1], 1))
                    else:
                        feature_frames['edge'].append(DG_output_data.edge_attr)
                    feature_frames['gradient_weight'].append(DG_output_data.gradient_weight)
                    if DG_output_data.laplacian_weight is None:
                        feature_frames['laplacian_weight'].append(DG_output_data.x.new_zeros(DG_output_data.edge_index.shape[1], 1))
                    else:
                        feature_frames['laplacian_weight'].append(DG_output_data.laplacian_weight)
                # use output as next input
                input_graph = output_graph

            output_seq = torch.stack(output_frames, dim=1)
            if return_features:
                feature_frames['node'] = torch.stack(feature_frames['node'], dim=1)
                feature_frames['edge'] = torch.stack(feature_frames['edge'], dim=1)
                feature_frames['gradient_weight'] = torch.stack(feature_frames['gradient_weight'], dim=1)
                feature_frames['laplacian_weight'] = torch.stack(feature_frames['laplacian_weight'], dim=1)

            # for training, use all outputs from model
            if self.training:
                to_return = output_seq
            # for evaluation, skip first input_frame_num frames and first skip_first_frames_num frames
            else:
                max_gt_frame = max(input_frame_num, self.skip_first_frames_num)
                if max_gt_frame < length:
                    to_return = torch.cat((data.target[:, :max_gt_frame, :], output_seq[:, max_gt_frame:, :]), dim=1)
                else:
                    to_return = data.target[:, :, :]

        if return_features:
            return to_return, feature_frames
        else:
            return to_return
        # to_return_unbatched = unbatch_node_feature_mat(to_return, data.batch)
        # to_return_list = []
        # for to_return_item, valid_length in zip(to_return_unbatched, valid_lengths):
        #     to_return_list.append(to_return_item[:, :valid_length, :])