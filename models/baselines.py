from __future__ import division
from __future__ import print_function

import os
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)
import itertools

from torch_geometric.data import Data
import torch
import torch.nn as nn

from models.utils import unbatch_node_feature, unbatch_node_feature_mat
from models.utils import make_mlp, SeriesModel, replace_graph


class SingleVAR(SeriesModel):
    def __init__(self,
                 input_dim,
                 output_dim,
                 input_frame_num,
                 skip_first_frames_num):

        super(SingleVAR, self).__init__(input_dim, output_dim, input_frame_num,
                                        skip_first_frames_num, is_recurrent=False)
        self.var_layer = nn.Linear(self.input_frame_num * self.input_dim, self.output_dim)

    def forward_onestep(self, data):
        out = self.var_layer(data.x.flatten(1, 2))
        out = out + data.x[:, -1, -self.output_dim:]
        out_graph = replace_graph(data, x=out)
        return out_graph


class JointVAR(SeriesModel):
    def __init__(self,
                 input_dim,
                 output_dim,
                 input_frame_num,
                 skip_first_frames_num,
                 node_num):

        super(JointVAR, self).__init__(input_dim, output_dim, input_frame_num,
                                       skip_first_frames_num, is_recurrent=False)
        self.node_num = node_num
        self.var_layer = nn.Linear(self.node_num * self.input_frame_num * self.input_dim,
                                   self.node_num * self.output_dim)

    def forward_onestep(self, data):
        input_features_list = unbatch_node_feature(data, 'x', data.batch) # list
        graph_batch_list = unbatch_node_feature(data, 'graph_batch', data.batch)
        input_features = list(itertools.chain.from_iterable([unbatch_node_feature_mat(input_features_i, graph_batch_i)
                          for input_features_i, graph_batch_i in zip(input_features_list, graph_batch_list)]))
        input_features = torch.stack(input_features, dim=0)
        input_features = input_features.reshape(input_features.shape[0], -1)
        out = self.var_layer(input_features)
        out = out.reshape(input_features.shape[0], self.node_num, self.output_dim).flatten(0, 1)
        out = out + data.x[:, -1, -self.output_dim:]
        out_graph = replace_graph(data, x=out)
        return out_graph


class SingleMLP(SeriesModel):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 layer_num,
                 input_frame_num,
                 skip_first_frames_num):

        super(SingleMLP, self).__init__(input_dim, output_dim, input_frame_num,
                                        skip_first_frames_num, is_recurrent=False)
        self.hidden_dim = hidden_dim
        self.mlp = make_mlp(input_dim * self.input_frame_num, hidden_dim, output_dim,
                            layer_num, activation='ReLU', final_activation=False)

    def forward_onestep(self, data):
        out = self.mlp(data.x.flatten(1, 2))
        out = out + data.x[:, -1, -self.output_dim:]
        out_graph = replace_graph(data, x=out)
        return out_graph


class JointMLP(SeriesModel):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 layer_num,
                 input_frame_num,
                 skip_first_frames_num,
                 node_num):
        super(JointMLP, self).__init__(input_dim, output_dim, input_frame_num,
                                        skip_first_frames_num, is_recurrent=False)
        self.hidden_dim = hidden_dim
        self.node_num = node_num
        self.mlp = make_mlp(node_num * input_dim * self.input_frame_num, node_num * hidden_dim, node_num * output_dim,
                            layer_num, activation='ReLU', final_activation=False)

    def forward_onestep(self, data):
        input_features_list = unbatch_node_feature(data, 'x', data.batch)  # list
        graph_batch_list = unbatch_node_feature(data, 'graph_batch', data.batch)
        input_features = list(itertools.chain.from_iterable([unbatch_node_feature_mat(input_features_i, graph_batch_i)
                                                             for input_features_i, graph_batch_i in
                                                             zip(input_features_list, graph_batch_list)]))
        input_features = torch.stack(input_features, dim=0)
        input_features = input_features.reshape(input_features.shape[0], -1)
        out = self.mlp(input_features)
        out = out.reshape(input_features.shape[0], self.node_num, self.output_dim).flatten(0, 1)
        out = out + data.x[:, -1, -self.output_dim:]
        out_graph = replace_graph(data, x=out)
        return out_graph


class SingleRNN(SeriesModel):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, skip_first_frames_num):
        super(SingleRNN, self).__init__(input_dim, output_dim, input_frame_num=1,
                                        skip_first_frames_num=skip_first_frames_num, is_recurrent=True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder = nn.Linear(self.hidden_dim, self.output_dim)

    def forward_onestep(self, data):
        if not hasattr(data, 'node_hidden'):
            data = replace_graph(data,
                node_hidden=data.x.new_zeros(self.num_layers, data.x.shape[0], self.hidden_dim))
        node_hidden_output, node_hidden_next = self.rnn(data.x, data.node_hidden)
        node_output = self.decoder(node_hidden_output) + data.x[:, -1:, -self.output_dim:]
        node_output = node_output.squeeze(1)
        output_graph = replace_graph(data, x=node_output, node_hidden=node_hidden_next)
        return output_graph


class JointRNN(SeriesModel):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim,
                 num_layers,
                 skip_first_frames_num,
                 node_num):
        super(JointRNN, self).__init__(input_dim, output_dim, input_frame_num=1,
                                       skip_first_frames_num=skip_first_frames_num, is_recurrent=True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.node_num = node_num

        self.rnn = nn.GRU(
            input_size=self.node_num * self.input_dim,
            hidden_size=self.node_num * self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.decoder = nn.Linear(self.node_num * self.hidden_dim, self.node_num * self.output_dim)

    def forward_onestep(self, data):
        input_features_list = unbatch_node_feature(data, 'x', data.batch)  # list
        graph_batch_list = unbatch_node_feature(data, 'graph_batch', data.batch)
        input_features = list(itertools.chain.from_iterable([unbatch_node_feature_mat(input_features_i, graph_batch_i)
                                                             for input_features_i, graph_batch_i in
                                                             zip(input_features_list, graph_batch_list)]))
        input_features = torch.stack(input_features, dim=0)
        input_features = input_features.transpose(1, 2).flatten(2, 3)

        if not hasattr(data, 'node_hidden'):
            data = replace_graph(data,
                node_hidden=data.x.new_zeros(self.num_layers, input_features.shape[0], self.rnn.hidden_size))
        node_hidden_output, node_hidden_next = self.rnn(input_features, data.node_hidden)
        node_hidden_output = self.decoder(node_hidden_output)
        node_hidden_output = node_hidden_output.reshape(input_features.shape[0], self.node_num, self.output_dim).flatten(0, 1)
        node_output = node_hidden_output + data.x[:, -1, -self.output_dim:]
        output_graph = replace_graph(data, x=node_output, node_hidden=node_hidden_next)
        return output_graph