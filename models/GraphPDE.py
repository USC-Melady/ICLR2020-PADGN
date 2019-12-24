from __future__ import division
from __future__ import print_function

import os
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)
import itertools

import numpy as np
from torch_geometric.data import Data
import torch
import torch.nn as nn
from torch_scatter import scatter_add

from models.blocks import NodeBlock, EdgeBlock, RecurrentUpdateNet
from models.utils import SeriesModel, make_mlp, replace_graph
from models.PDGN import MultiLayerRecurrentGN


class GraphPDE(SeriesModel):
    def __init__(self,
                 input_dim,
                 output_dim,
                 input_frame_num,
                 skip_first_frames_num,
                 node_meta_dim=0,
                 order=2,
                 coef_set_num_per_order=1,
                 coef_net_hidden_dim=64,
                 coef_net_layer_num=1,
                 coef_net_is_recurrent=False,
                 coef_mode='abc',
                 prediction_net_hidden_dim=64,
                 prediction_net_layer_num=2,
                 prediction_net_is_recurrent=True,
                 agg_mode='sum',
                 coef_sharing=True,
                 batchnorm=None):
        '''
        GraphPDE Model
        :param input_dim: Dimension of input features
        :param output_dim: Dimension of output features
        :param hidden_dim: Dimension of all hidden layers
        :param input_frame_num: Number of frames needed to produce 1 prediction
        :param skip_first_frames_num: Number of first frames without prediction (use ground truth for them)
        :param recurrent: Whether the model is recurrent or not
        :param coef_net_type:
                1. 'GN' (GraphNets of MLP) or 'RGN' (Recurrent GraphNets)
                2. output is coef on edges, thus arch is EdgeBlock -> (NodeBlock -> EdgeBlock)*
        :param order:
                the highest order number
                (1: gradient, 2: laplacian, 3: gradient of laplacian, 4: laplaican of laplacian...)
        :param coef_sharing: whether (2k + 1) and (2k + 2) order ops share a and b
                (by default share, e.g. gradient and laplacian ops share a, b but not c by default)
        '''
        super(GraphPDE, self).__init__(input_dim, output_dim, input_frame_num,
                                   skip_first_frames_num,
                                       is_recurrent=(coef_net_is_recurrent or prediction_net_is_recurrent))

        self.node_meta_dim = node_meta_dim
        self.coef_net_hidden_dim = coef_net_hidden_dim
        self.coef_net_layer_num = coef_net_layer_num
        self.coef_net_is_recurrent = coef_net_is_recurrent
        self.coef_mode = coef_mode
        # assert self.coef_mode in ['ab', 'ac', 'abc']
        self.prediction_net_hidden_dim = prediction_net_hidden_dim
        self.prediction_net_layer_num = prediction_net_layer_num
        self.prediction_net_is_recurrent = prediction_net_is_recurrent
        self.agg_mode = agg_mode
        assert self.agg_mode in ['sum', 'RGN', 'SingleMLP']
        self.coef_sharing = coef_sharing

        self.order = order
        self.coef_set_num_per_order = coef_set_num_per_order
        self.coef_num_each_set = len(self.coef_mode)
        self.edge_coef_num_per_order = self.coef_set_num_per_order * self.coef_num_each_set * self.input_dim
        gn_update_func_layer_num = 2

        # For each order, we have 3 coefs: a, b, c
        # Each order's message: a * (dst - b * src)
        # Each order's contrib to final output (diff): c * message = c * (a * (dst - b * src))
        # in the case of 2: c * (a * (dst - src))

        # update: gradient/laplacian share the same a, b

        if self.edge_coef_num_per_order > 0:
            coef_nets = []
            for _ in range(self.order):
                coef_net = build_gn_net_edgeout(
                    first_gn_node_dim_in=(self.input_dim + self.node_meta_dim),
                    first_gn_edge_dim_in=0,
                    edge_out_dim=self.edge_coef_num_per_order,
                    hidden_dim_gn=self.coef_net_hidden_dim,
                    layer_num=self.coef_net_layer_num,
                    update_func_layer_num=gn_update_func_layer_num,
                    is_recurrent=self.coef_net_is_recurrent,
                    batchnorm=batchnorm
                )
                coef_nets.append(coef_net)
            self.coef_nets = nn.ModuleList(coef_nets)

        if self.agg_mode == 'RGN':
            self.prediction_net = build_gn_net_nodeout(
                first_gn_node_dim_in=(self.input_dim + self.input_dim * (self.order // 2) * self.coef_set_num_per_order),
                first_gn_edge_dim_in=(self.input_dim * ((self.order + 1) // 2) * self.coef_set_num_per_order),
                node_out_dim=self.output_dim,
                hidden_dim_gn=self.prediction_net_hidden_dim,
                layer_num=self.prediction_net_layer_num,
                update_func_layer_num=gn_update_func_layer_num,
                is_recurrent=self.prediction_net_is_recurrent,
                batchnorm=batchnorm
            )
        elif self.agg_mode == 'SingleMLP':
            self.prediction_net = make_mlp(self.order * self.coef_set_num_per_order * self.input_dim,
                                           self.prediction_net_hidden_dim, self.output_dim,
                                           layer_num=1, activation='ReLU', final_activation=False)

    def forward_onestep(self, data, return_features=False):
        if self.coef_net_is_recurrent and (self.edge_coef_num_per_order > 0):
            for order_i in range(self.order):
                if not hasattr(data, 'edge_hidden_coef_net_{}'.format(order_i)):
                    kwargs = {
                        'edge_hidden_coef_net_{}'.format(order_i):
                            data.x.new_zeros(self.coef_net_layer_num,
                           self.coef_nets[order_i].gn_layers[0][0].net.num_layers,
                           data.edge_index.shape[1],
                          self.coef_nets[order_i].gn_layers[0][0].net.latent_dim)
                    }
                    data = replace_graph(data, **kwargs)
                if (self.coef_net_layer_num > 1) and (not hasattr(data, 'node_hidden_coef_net_{}'.format(order_i))):
                    kwargs = {
                        'node_hidden_coef_net_{}'.format(order_i):
                            data.x.new_zeros(self.coef_net_layer_num,
                            self.coef_nets[order_i].gn_layers[1][0].net.num_layers,
                            data.x.shape[0],
                            self.coef_nets[order_i].gn_layers[1][0].net.latent_dim)
                    }
                    data = replace_graph(data, **kwargs)
        if self.prediction_net_is_recurrent and (self.agg_mode == 'RGN'):
            if not hasattr(data, 'edge_hidden_prediction_net'):
                data = replace_graph(data,
                                     edge_hidden_prediction_net=data.x.new_zeros(self.prediction_net_layer_num,
                                                             self.prediction_net.gn_layers[0][0].net.num_layers,
                                                             data.edge_index.shape[1],
                                                             self.prediction_net.gn_layers[0][0].net.latent_dim))
            if not hasattr(data, 'node_hidden_prediction_net'):
                data = replace_graph(data,
                                     node_hidden_prediction_net=data.x.new_zeros(self.prediction_net_layer_num,
                                                             self.prediction_net.gn_layers[0][1].net.num_layers,
                                                             data.x.shape[0],
                                                             self.prediction_net.gn_layers[0][1].net.latent_dim))

        data_input = replace_graph(data, x=data.x[:, -1, :])

        if self.edge_coef_num_per_order > 0:
            if self.node_meta_dim > 0:
                coef_net_input = replace_graph(data_input, x=torch.cat((data_input.x, data.node_meta), dim=-1))
            else:
                coef_net_input = replace_graph(data_input, x=data_input.x)
            coef_out_list = []

            if self.coef_net_is_recurrent:
                coef_updated_node_hidden_list = []
                coef_updated_edge_hidden_list = []
                for order_i in range(self.order):
                    coef_net_input = replace_graph(coef_net_input,
                                                   node_hidden=getattr(coef_net_input, 'node_hidden_coef_net_{}'.format(order_i)),
                                                   edge_hidden=getattr(coef_net_input, 'edge_hidden_coef_net_{}'.format(order_i)))
                    coef_out_graph = self.coef_nets[order_i](coef_net_input)
                    coef_out_list.append(coef_out_graph.edge_attr)
                    coef_updated_node_hidden_list.append(coef_out_graph.node_hidden)
                    coef_updated_edge_hidden_list.append(coef_out_graph.edge_hidden)
                kwargs = {}
                for order_i in range(self.order):
                    kwargs['node_hidden_coef_net_{}'.format(order_i)] = coef_updated_node_hidden_list[order_i]
                    kwargs['edge_hidden_coef_net_{}'.format(order_i)] = coef_updated_edge_hidden_list[order_i]
                coef_out_graph = replace_graph(coef_net_input, **kwargs)
            else:
                coef_out_graph = None
                for order_i in range(self.order):
                    coef_net_input_i = replace_graph(coef_net_input)
                    coef_out_graph = self.coef_nets[order_i](coef_net_input_i)
                    coef_out_list.append(coef_out_graph.edge_attr)

            coef_out = torch.stack(coef_out_list, dim=0) # Order x (E x (setnum * coefnum * F))
            coef_out = coef_out.reshape(coef_out.shape[0], coef_out.shape[1],
                                        self.coef_set_num_per_order, self.coef_num_each_set, self.input_dim)
        else:
            coef_out = None
            coef_out_graph = replace_graph(data_input)

        op_calc = self._calculate_op(data_input, coef_out)

        if self.agg_mode == 'sum':
            model_prediction_out = torch.sum(torch.sum(op_calc['op_contribs'], dim=2), dim=0)
            output_graph = replace_graph(coef_out_graph,
                                         x=((data_input.x + model_prediction_out)[..., -self.output_dim:]))
        elif self.agg_mode == 'SingleMLP': # O N S F
            net_input = op_calc['op_agg_msgs'].permute(1, 0, 2, 3).flatten(1, -1)
            model_prediction_out = self.prediction_net(net_input)
            output_graph = replace_graph(coef_out_graph,
                                         x=((data_input.x + model_prediction_out)[..., -self.output_dim:]))
        else:
            new_x_collect = [data_input.x,]
            new_ea_collect = []
            for order_i in range(self.order):
                if order_i % 2 == 0:
                    new_ea_collect.append(op_calc['op_msgs'][order_i].flatten(-2, -1))
                else:
                    new_x_collect.append(op_calc['op_agg_msgs'][order_i].flatten(-2, -1))
            new_x_input = torch.cat(new_x_collect, dim=-1)
            if len(new_ea_collect) > 0:
                new_ea_input = torch.cat(new_ea_collect, dim=-1)
            else:
                new_ea_input = None
            prediction_input_graph = replace_graph(coef_out_graph,
                                                   x=new_x_input, edge_attr=new_ea_input)
            if self.prediction_net_is_recurrent:
                prediction_input_graph = replace_graph(prediction_input_graph,
                                                       node_hidden=prediction_input_graph.node_hidden_prediction_net,
                                                       edge_hidden=prediction_input_graph.edge_hidden_prediction_net)
                model_prediction_out_graph = self.prediction_net(prediction_input_graph)
                model_prediction_out_graph = replace_graph(model_prediction_out_graph,
                                                           node_hidden_prediction_net=model_prediction_out_graph.node_hidden,
                                                           edge_hidden_prediction_net=model_prediction_out_graph.edge_hidden)
            else:
                model_prediction_out_graph = self.prediction_net(prediction_input_graph)
            output_graph = replace_graph(model_prediction_out_graph,
                                             x=(data_input.x[..., -self.output_dim:] + model_prediction_out_graph.x))

        if return_features:
            DG_output_data = replace_graph(coef_out_graph,
                                           x=data_input.x,
                                           edge_attr=None,
                                           gradient_weight=coef_out.transpose(0, 1).flatten(1, -1),
                                         laplacian_weight=None)
            return output_graph, DG_output_data
        else:
            return output_graph


    def _calculate_op(self, data_input, coef_out):
        # coef_out: None or [Order x E x setnum x coefnum x F]
        op_msgs = []
        op_agg_msgs = []
        op_contribs = []

        last_op_agg_msg = data_input.x.unsqueeze(1) # N x 1 x F
        edge_index = data_input.edge_index # 2 x E

        for order_i in range(self.order):
            op_msg_src, op_msg_dst = last_op_agg_msg[edge_index[0]], last_op_agg_msg[edge_index[1]] # E x 1 x F (or E x S x F)
            coef_dict = {}
            for ci, cname in enumerate(self.coef_mode):
                if (order_i % 2 == 1) and (self.coef_sharing) and (cname in ('a', 'b')):
                    coef_dict[cname] = coef_out[order_i - 1, :, :, ci, :] # E x S x F
                else:
                    coef_dict[cname] = coef_out[order_i, :, :, ci, :] # E x S x F
            if 'a' in coef_dict.keys():
                coef_a = coef_dict['a']
            else:
                coef_a = 1
            if 'b' in coef_dict.keys():
                coef_b = coef_dict['b']
            else:
                coef_b = 1
            if 'c' in coef_dict.keys():
                coef_c = coef_dict['c']
            else:
                coef_c = 1

            if self.coef_sharing:
                if order_i % 2 == 0:
                    op_msg = coef_a * (op_msg_dst - coef_b * op_msg_src)
                else:
                    op_msg = op_msgs[-1]
            else:
                op_msg = coef_a * (op_msg_dst - coef_b * op_msg_src)
            op_msgs.append(op_msg)

            if order_i % 2 == 0:
                op_agg_msg = scatter_add(op_msg, edge_index[1], dim=0) # N x S x F
            else:
                op_agg_msg = scatter_add(op_msg, edge_index[1], dim=0)  # N x S x F
                last_op_agg_msg = op_agg_msg
            op_agg_msgs.append(op_agg_msg)

            op_contrib_msg = coef_c * op_msg # E x S x F
            op_contrib = scatter_add(op_contrib_msg, edge_index[1], dim=0) # N x S x F
            op_contribs.append(op_contrib)

        return {
            'op_msgs': torch.stack(op_msgs, dim=0), # O x E x S x F
            'op_agg_msgs': torch.stack(op_agg_msgs, dim=0), # O x N x S x F
            'op_contribs': torch.stack(op_contribs, dim=0) # O x N x S x F
        }


def build_gn_net_edgeout(first_gn_node_dim_in, first_gn_edge_dim_in,
                          edge_out_dim,
                          hidden_dim_gn, layer_num, update_func_layer_num, is_recurrent, batchnorm):
    gn_net_blocks = []
    for li in range(layer_num):
        t_node_dim_in, t_node_dim_out = hidden_dim_gn, hidden_dim_gn
        t_edge_dim_in, t_edge_dim_out = hidden_dim_gn, hidden_dim_gn
        final_activation = True
        if li == 0:
            t_edge_dim_in = first_gn_edge_dim_in
        if li <= 1:
            t_node_dim_in = first_gn_node_dim_in
        if li == layer_num - 1:
            t_edge_dim_out = edge_out_dim
            final_activation = False

        if li > 0:
            _node_input_dim = 1 * t_node_dim_in + 2 * t_edge_dim_in  # bi-direction
            if is_recurrent:
                gn_node_func = RecurrentUpdateNet(
                    in_features=_node_input_dim,
                    latent_dim=hidden_dim_gn,
                    out_features=t_node_dim_out,
                    num_layers=update_func_layer_num,
                    final_activation=True,
                    batchnorm=batchnorm
                )
            else:
                gn_node_func = make_mlp(_node_input_dim, hidden_dim_gn, t_node_dim_out,
                                        update_func_layer_num, activation='ReLU', final_activation=True,batchnorm=batchnorm)
            gn_node_block = NodeBlock(_node_input_dim, t_node_dim_out,
                                      use_nodes=True, use_sent_edges=True, use_received_edges=True,
                                      use_globals=False,
                                      custom_func=gn_node_func, recurrent=is_recurrent)
        else:
            gn_node_block = None

        if li == 0:
            _edge_input_dim = 2 * t_node_dim_in + 1 * t_edge_dim_in  # sender/receiver nodes
        else:
            _edge_input_dim = 2 * t_node_dim_out + 1 * t_edge_dim_in  # sender/receiver nodes
        if is_recurrent:
            gn_edge_func = RecurrentUpdateNet(
                in_features=_edge_input_dim,
                latent_dim=hidden_dim_gn,
                out_features=t_edge_dim_out,
                num_layers=update_func_layer_num,
                final_activation=final_activation,
                batchnorm=batchnorm
            )
        else:
            gn_edge_func = make_mlp(_edge_input_dim, hidden_dim_gn, t_edge_dim_out, update_func_layer_num,
                                    activation='ReLU', final_activation=final_activation, batchnorm=batchnorm)
        gn_edge_block = EdgeBlock(_edge_input_dim, t_edge_dim_out,
                                  use_edges=(t_edge_dim_in > 0), use_sender_nodes=True, use_receiver_nodes=True,
                                  use_globals=False, custom_func=gn_edge_func, recurrent=is_recurrent)

        li_block_list = [gn_edge_block, ]
        if gn_node_block is not None:
            li_block_list.insert(0, gn_node_block)
        if is_recurrent:
            gn_net_blocks.append(nn.Sequential(
                *li_block_list
            ))
        else:
            for gb in li_block_list:
                gn_net_blocks.append(gb)
    if is_recurrent:
        gn_net = MultiLayerRecurrentGN(gn_net_blocks)
    else:
        gn_net = nn.Sequential(*gn_net_blocks)
    return gn_net

def build_gn_net_nodeout(first_gn_node_dim_in, first_gn_edge_dim_in,
                          node_out_dim,
                          hidden_dim_gn, layer_num, update_func_layer_num, is_recurrent, batchnorm):
    gn_net_blocks = []
    for li in range(layer_num):
        t_node_dim_in, t_node_dim_out = hidden_dim_gn, hidden_dim_gn
        t_edge_dim_in, t_edge_dim_out = hidden_dim_gn, hidden_dim_gn
        final_activation = True
        if li == 0:
            t_edge_dim_in = first_gn_edge_dim_in
            t_node_dim_in = first_gn_node_dim_in
        if li == layer_num - 1:
            t_node_dim_out = node_out_dim
            final_activation = False

        _edge_input_dim = 2 * t_node_dim_in + 1 * t_edge_dim_in  # sender/receiver nodes
        if is_recurrent:
            gn_edge_func = RecurrentUpdateNet(
                in_features=_edge_input_dim,
                latent_dim=hidden_dim_gn,
                out_features=t_edge_dim_out,
                num_layers=update_func_layer_num,
                final_activation=True,
                batchnorm=batchnorm
            )
        else:
            gn_edge_func = make_mlp(_edge_input_dim, hidden_dim_gn, t_edge_dim_out, update_func_layer_num,
                                    activation='ReLU', final_activation=True, batchnorm=batchnorm)
        gn_edge_block = EdgeBlock(_edge_input_dim, t_edge_dim_out,
                                  use_edges=(t_edge_dim_in > 0), use_sender_nodes=True, use_receiver_nodes=True,
                                  use_globals=False, custom_func=gn_edge_func, recurrent=is_recurrent)

        _node_input_dim = 1 * t_node_dim_in + 2 * t_edge_dim_out  # bi-direction
        if is_recurrent:
            gn_node_func = RecurrentUpdateNet(
                in_features=_node_input_dim,
                latent_dim=hidden_dim_gn,
                out_features=t_node_dim_out,
                num_layers=update_func_layer_num,
                final_activation=final_activation,
                batchnorm=batchnorm
            )
        else:
            gn_node_func = make_mlp(_node_input_dim, hidden_dim_gn, t_node_dim_out,
                                    update_func_layer_num, activation='ReLU', final_activation=final_activation, batchnorm=batchnorm)
        gn_node_block = NodeBlock(_node_input_dim, t_node_dim_out,
                                  use_nodes=True, use_sent_edges=True, use_received_edges=True,
                                  use_globals=False,
                                  custom_func=gn_node_func, recurrent=is_recurrent)

        li_block_list = [gn_edge_block, gn_node_block]
        if is_recurrent:
            gn_net_blocks.append(nn.Sequential(
                *li_block_list
            ))
        else:
            for gb in li_block_list:
                gn_net_blocks.append(gb)
    if is_recurrent:
        gn_net = MultiLayerRecurrentGN(gn_net_blocks)
    else:
        gn_net = nn.Sequential(*gn_net_blocks)
    return gn_net