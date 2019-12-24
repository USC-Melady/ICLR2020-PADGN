from __future__ import division
from __future__ import print_function

import os
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)

import numpy as np
from numpy import linalg as LA
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
import torch
import torch.nn as nn
from torch_scatter import scatter_add

from models.blocks import NodeBlock, NodeBlockInd, EdgeBlock, RecurrentUpdateNet
from models.utils import SeriesModel, make_mlp, replace_graph


class GradientLayer(nn.Module):
    def __init__(self, net=None, kernel_param='12', kernel_feature='both'):
        super(GradientLayer, self).__init__()
        self.net = net
        self.kernel_param = kernel_param
        self.kernel_feature = kernel_feature

    def _forward_one_net(self, net, x, edge_index, target='out'):
        x_src, x_dst = x[edge_index[0]], x[edge_index[1]]
        if net is None:
            out = x_dst - x_src
            # net_out = torch.ones(edge_index.shape[1], 1).to(x.device)
            net_out = x.new_ones(edge_index.shape[1], 2)
        else:
            if self.kernel_feature == 'both':
                net_input = torch.cat((x_dst, x_src), dim=-1)
            elif self.kernel_feature == 'src':
                net_input = x_src
            elif self.kernel_feature == 'dst':
                net_input = x_dst
            net_out = net(net_input)
            # out = x_dst - net_out * x_src
            # out = net_out * (x_dst - x_src)
            net_out = net_out.reshape(-1, 2)

            net_out_ones = torch.ones_like(net_out)
            if self.kernel_param == '1':
                net_out = torch.cat((net_out[:, 0:1], net_out_ones[:, 1:2]), dim=-1)
            elif self.kernel_param == '2':
                net_out = torch.cat((net_out_ones[:, 0:1], net_out[:, 1:2]), dim=-1)

            out = net_out[:, 0:1] * (x_dst - net_out[:, 1:2] * x_src)
        if target == 'out':
            return out
        elif target == 'net_out':
            return net_out
        else:
            raise NotImplementedError()

    def forward(self, x, edge_index):
        if isinstance(self.net, nn.ModuleList):
            out_list = [self._forward_one_net(net, x, edge_index, 'out') for net in self.net]
            return torch.cat(out_list, dim=-1)
        else:
            return self._forward_one_net(self.net, x, edge_index, 'out')

    def get_net_out(self, x, edge_index):
        if isinstance(self.net, nn.ModuleList):
            net_out_list = [self._forward_one_net(net, x, edge_index, 'net_out') for net in self.net]
            return torch.cat(net_out_list, dim=-1)
        else:
            return self._forward_one_net(self.net, x, edge_index, 'net_out')


class LaplacianLayer(MessagePassing):
    def __init__(self, net=None, kernel_param='12', kernel_feature='both'):
        super(LaplacianLayer, self).__init__(aggr='add', flow='source_to_target')
        self.net = net
        self.kernel_param = kernel_param
        self.kernel_feature = kernel_feature

    def _message_one_net(self, net, x_i, x_j):
        if net is None:
            return x_i - x_j
        else:
            if self.kernel_feature == 'both':
                net_input = torch.cat((x_i, x_j), dim=-1)
            elif self.kernel_feature == 'src':
                net_input = x_i
            elif self.kernel_feature == 'dst':
                net_input = x_j
            net_out = net(net_input)
            net_out = net_out.reshape(-1, 2)

            net_out_ones = torch.ones_like(net_out)
            if self.kernel_param == '1':
                net_out = torch.cat((net_out[:, 0:1], net_out_ones[:, 1:2]), dim=-1)
            elif self.kernel_param == '2':
                net_out = torch.cat((net_out_ones[:, 0:1], net_out[:, 1:2]), dim=-1)

            return net_out[:, 0:1] * (x_i - net_out[:, 1:2] * x_j)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        if isinstance(self.net, nn.ModuleList):
            message_list = [self._message_one_net(net, x_i, x_j) for net in self.net]
            return torch.cat(message_list, dim=-1)
        else:
            return self._message_one_net(self.net, x_i, x_j)

    def update(self, aggr_out):
        return aggr_out

    def _get_one_net_out(self, net, x, edge_index):
        if net is None:
            return x.new_ones(edge_index.shape[1], 2)
        else:
            x_src, x_dst = x[edge_index[0]], x[edge_index[1]]

            if self.kernel_feature == 'both':
                net_input = torch.cat((x_src, x_dst), dim=-1)
            elif self.kernel_feature == 'src':
                net_input = x_src
            elif self.kernel_feature == 'dst':
                net_input = x_dst
            net_out = net(net_input)
            net_out = net_out.reshape(-1, 2)

            net_out_ones = torch.ones_like(net_out)
            if self.kernel_param == '1':
                net_out = torch.cat((net_out[:, 0:1], net_out_ones[:, 1:2]), dim=-1)
            elif self.kernel_param == '2':
                net_out = torch.cat((net_out_ones[:, 0:1], net_out[:, 1:2]), dim=-1)
            return net_out

    def get_net_out(self, x, edge_index):
        if isinstance(self.net, nn.ModuleList):
            net_out_list = [self._get_one_net_out(net, x, edge_index) for net in self.net]
            return torch.cat(net_out_list, dim=-1)
        else:
            return self._get_one_net_out(self.net, x, edge_index)


class MultiLayerRecurrentGN(nn.Module):
    def __init__(self, gn_layers):
        super(MultiLayerRecurrentGN, self).__init__()
        self.gn_layers = nn.ModuleList(gn_layers)
        self.gn_layer_num = len(gn_layers)

    def forward(self, graph):
        # 1st dim is layer rank
        node_hidden_list = graph.node_hidden
        edge_hidden_list = graph.edge_hidden
        updated_node_hidden_list = []
        updated_edge_hidden_list = []
        assert len(node_hidden_list) == self.gn_layer_num

        graph_li = replace_graph(graph)

        for li in range(self.gn_layer_num):
            graph_li = replace_graph(graph_li, node_hidden=node_hidden_list[li], edge_hidden=edge_hidden_list[li])
            graph_li = self.gn_layers[li](graph_li)
            updated_node_hidden_list.append(graph_li.node_hidden)
            updated_edge_hidden_list.append(graph_li.edge_hidden)


        graph = replace_graph(graph_li,
                              node_hidden=torch.stack(updated_node_hidden_list, dim=0),
                              edge_hidden=torch.stack(updated_edge_hidden_list, dim=0))
        return graph


class PDGN(SeriesModel):
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_dim_pde,
                 hidden_dim_gn,
                 input_frame_num,
                 skip_first_frames_num,
                 mode,
                 recurrent,
                 layer_num,
                 gn_layer_num=1,
                 edge_final_dim=None,
                 nophysics_mode=None,
                 use_dist=False,
                 pde_params_out_dim=None,
                 use_time_grad=True,
                 use_edge_grad=True,
                 use_laplacian=True,
                 use_pde_params=True,
                 learnable_edge_grad=False,
                 learnable_edge_grad_kernel_num=1,
                 learnable_laplacian=False,
                 learnable_laplacian_kernel_num=1,
                 grad_kernel_param_loc='12',
                 grad_kernel_feature='both',
                 laplacian_kernel_param_loc='12',
                 laplacian_kernel_feature='both',
                 node_meta_dim=0,
                 predict_model='GN'):

        super(PDGN, self).__init__(input_dim, output_dim, input_frame_num,
                                   skip_first_frames_num, is_recurrent=recurrent)

        #### For learnable parameters
        self.node_dim = input_dim  # for easy understanding
        self.edge_dim = input_dim  # for easy understanding
        self.hidden_dim_pde = hidden_dim_pde
        self.hidden_dim_gn = hidden_dim_gn
        self.layer_num = layer_num
        self.gn_layer_num = gn_layer_num

        #### For PDE
        self.mode = mode
        self.nophysics_mode = nophysics_mode

        self.use_dist = use_dist
        if self.use_dist:
            self.edge_dist_dim = 1
        else:
            self.edge_dist_dim = 0

        if pde_params_out_dim is None:
            if mode == 'diff':
                pde_params_out_dim = self.node_dim
            elif mode == 'adv':
                pde_params_out_dim = self.edge_dim

        # use time grad
        self.use_time_grad = use_time_grad
        self.time_grad_dim = self.node_dim if self.use_time_grad else 0
        # use edge grad
        self.use_edge_grad = use_edge_grad
        self.edge_grad_dim = self.edge_dim if self.use_edge_grad else 0
        # use laplacian
        self.use_laplacian = use_laplacian
        self.laplacian_dim = self.node_dim if self.use_laplacian else 0
        # use pde params
        self.use_pde_params = use_pde_params
        self.pde_params_dim = pde_params_out_dim if self.use_pde_params else 0

        self.learnable_edge_grad = learnable_edge_grad
        self.learnable_edge_grad_kernel_num = learnable_edge_grad_kernel_num
        self.learnable_laplacian = learnable_laplacian
        self.learnable_laplacian_kernel_num = learnable_laplacian_kernel_num

        if self.learnable_edge_grad:
            if grad_kernel_feature == 'both':
                grad_net_input_mult = 2
            else:
                grad_net_input_mult = 1
            grad_net_list = [
                make_mlp(grad_net_input_mult * (self.node_dim + node_meta_dim), self.hidden_dim_pde, self.node_dim * 2, self.layer_num, activation='SELU')
                for _ in range(self.learnable_edge_grad_kernel_num)
            ]
            self.gradient_layer = GradientLayer(
                net=nn.ModuleList(grad_net_list), kernel_param=grad_kernel_param_loc, kernel_feature=grad_kernel_feature
            )
            self.edge_grad_dim = self.edge_grad_dim * self.learnable_edge_grad_kernel_num
        else:
            self.gradient_layer = GradientLayer()
        if self.learnable_laplacian:
            if laplacian_kernel_feature == 'both':
                laplacian_net_input_mult = 2
            else:
                laplacian_net_input_mult = 1
            laplacian_net_list = [
                make_mlp(laplacian_net_input_mult * (self.node_dim + node_meta_dim), self.hidden_dim_pde, self.node_dim * 2, self.layer_num, activation='SELU')
                for _ in range(self.learnable_laplacian_kernel_num)
            ]
            self.laplacian_layer = LaplacianLayer(
                net=nn.ModuleList(laplacian_net_list), kernel_param=laplacian_kernel_param_loc, kernel_feature=laplacian_kernel_feature
            )
            self.laplacian_dim = self.laplacian_dim * self.learnable_laplacian_kernel_num
        else:
            self.laplacian_layer = LaplacianLayer()

        def _get_pde_specific_parts(mode):
            if mode == 'diff':
                if self.use_pde_params:
                    pde_mlp = make_mlp(2 * self.node_dim, self.hidden_dim_pde,
                                       self.pde_params_dim, self.layer_num, activation='ReLU')
                    pde_net = NodeBlockInd(2 * self.node_dim,
                                            self.pde_params_dim,
                                            custom_func=pde_mlp)  # This is actually a simple SingleMLP.
                else:
                    pde_mlp, pde_net = None, None
                first_gn_node_dim_in, first_gn_edge_dim_in = \
                    self.node_dim + self.laplacian_dim + self.time_grad_dim + self.pde_params_dim, self.edge_grad_dim
            elif mode == 'adv':
                if self.use_pde_params:
                    pde_mlp = make_mlp(2 * self.node_dim + self.edge_dim, self.hidden_dim_pde, self.pde_params_dim,
                                            self.layer_num, activation='ReLU')
                    pde_net = EdgeBlock(2 * self.node_dim + self.edge_dim, self.pde_params_dim,
                                             use_edges=True, use_sender_nodes=True, use_receiver_nodes=True,
                                             use_globals=False,
                                             custom_func=pde_mlp)  # This is actually a simple SingleMLP.
                else:
                    pde_mlp, pde_net = None, None
                first_gn_node_dim_in, first_gn_edge_dim_in = \
                    self.node_dim + self.laplacian_dim + self.time_grad_dim, self.edge_grad_dim + self.pde_params_dim
            else:
                raise NotImplementedError('{} not implemented!'.format(mode))

            return pde_mlp, pde_net, first_gn_node_dim_in, first_gn_edge_dim_in

        self.pde_mlp, self.pde_net, first_gn_node_dim_in, first_gn_edge_dim_in = _get_pde_specific_parts(self.mode)

        self.predict_model = predict_model
        if self.predict_model == 'GN':
            #### Prediction module
            gn_net_blocks = []
            for li in range(self.gn_layer_num):
                t_node_dim_in, t_node_dim_out = self.hidden_dim_gn, self.hidden_dim_gn
                t_edge_dim_in, t_edge_dim_out = self.hidden_dim_gn, self.hidden_dim_gn
                final_activation = True
                if li == 0:
                    if self.nophysics_mode == 'nopad':
                        t_node_dim_in, t_edge_dim_in = 1 * self.node_dim, self.edge_dist_dim
                    else:
                        t_node_dim_in, t_edge_dim_in = first_gn_node_dim_in, first_gn_edge_dim_in + self.edge_dist_dim
                if li == self.gn_layer_num - 1:
                    if edge_final_dim is None:
                        gn_edge_mlp_outdim = self.edge_dim
                    else:
                        gn_edge_mlp_outdim = self.hidden_dim_gn
                    t_node_dim_out, t_edge_dim_out = self.output_dim, gn_edge_mlp_outdim
                    final_activation = False

                _edge_input_dim = 2 * t_node_dim_in + 1 * t_edge_dim_in  # sender/receiver nodes
                if edge_final_dim is None:
                    gn_edge_mlp_final_activation = final_activation
                else:
                    gn_edge_mlp_final_activation = True
                if self.is_recurrent:
                    gn_edge_mlp = RecurrentUpdateNet(
                        in_features=_edge_input_dim,
                        latent_dim=self.hidden_dim_gn,
                        out_features=t_edge_dim_out,
                        num_layers=2,
                        final_activation=gn_edge_mlp_final_activation
                    )
                else:
                    gn_edge_mlp = make_mlp(_edge_input_dim, self.hidden_dim_gn, t_edge_dim_out,
                                           self.layer_num, activation='ReLU', final_activation=gn_edge_mlp_final_activation)
                gn_edge_block = EdgeBlock(_edge_input_dim, t_edge_dim_out,
                                          use_edges=(t_edge_dim_in > 0), use_sender_nodes=True, use_receiver_nodes=True,
                                          use_globals=False,
                                          custom_func=gn_edge_mlp, recurrent=self.is_recurrent)
                # Node: (curr, laplacian, du_dt, D) Edge: (gradient)
                _node_input_dim = 1 * t_node_dim_in + 2 * t_edge_dim_out  # bi-direction
                if self.is_recurrent:
                    gn_node_mlp = RecurrentUpdateNet(
                        in_features=_node_input_dim,
                        latent_dim=self.hidden_dim_gn,
                        out_features=t_node_dim_out,
                        num_layers=2,
                        final_activation=final_activation
                    )
                else:
                    gn_node_mlp = make_mlp(_node_input_dim, self.hidden_dim_gn, t_node_dim_out,
                                           self.layer_num, activation='ReLU', final_activation=final_activation)
                gn_node_block = NodeBlock(_node_input_dim, t_node_dim_out,
                                          use_nodes=True, use_sent_edges=True, use_received_edges=True,
                                          use_globals=False,
                                          custom_func=gn_node_mlp, recurrent=self.is_recurrent)
                if self.is_recurrent:
                    gn_net_blocks.append(nn.Sequential(
                        gn_edge_block, gn_node_block
                    ))
                else:
                    gn_net_blocks.append(gn_edge_block)
                    gn_net_blocks.append(gn_node_block)
            if self.is_recurrent:
                self.gn_net = MultiLayerRecurrentGN(gn_net_blocks)
            else:
                self.gn_net = nn.Sequential(*gn_net_blocks)
        else:
            assert self.predict_model == 'sum'

    def derivative_cell(self, data, length=2):
        """
        Derivative Cell (DC)
        Input:
        graph_seq:
            - len(graph_seq)==length is True.
            - Elements of graph_seq is torch.Tensor
            - All elements should have "x" and "edge_index" attributes.

        Return:
            - If length=2, only first-order temporal derivative is returned.
            - If length=3, first-, second-order temporal derivatives are returned.
            - If length>3, TODO
            - Spatial derivatives, Laplacian and Gradient are returned.
            - Zero-order derivative or current graph, i.e. graph_seq[-1], is also returned.
        """

        assert data.x.shape[1] == length

        G_prev = data.x[:, -2, :]
        G_curr = data.x[:, -1, :]
        edge_index = data.edge_index

        ret = {
            'curr': G_curr,
            'edge_index': edge_index
        }
        if self.nophysics_mode == 'nopad':
            pass
        else:
            if hasattr(data, 'node_meta'):
                gradient_input = torch.cat((G_curr, data.node_meta), dim=-1)
            else:
                gradient_input = G_curr
            ret['gradient'] = self.gradient_layer(gradient_input, edge_index)[..., :G_curr.shape[-1]]
            ret['gradient_weight'] = self.gradient_layer.get_net_out(gradient_input, edge_index).detach()
            ret['laplacian'] = self.laplacian_layer(gradient_input, edge_index)[..., :G_curr.shape[-1]]
            ret['laplacian_weight'] = self.laplacian_layer.get_net_out(gradient_input, edge_index).detach()

            # debug
            Gsrc, Gdst = G_curr[edge_index[0]], G_curr[edge_index[1]]
            debug_gradient = ret['gradient']
            debug_gradient_2 = ret['gradient_weight'][:, 0:1] * (Gdst - ret['gradient_weight'][:, 1:2] * Gsrc)
            debug_laplacian = ret['laplacian']
            debug_laplacian_2 = scatter_add(ret['laplacian_weight'][:, 0:1] * (Gsrc - ret['laplacian_weight'][:, 1:2] * Gdst), edge_index[0], dim=0)
            debug_laplacian_3 = scatter_add(ret['laplacian_weight'][:, 0:1] * (Gdst - ret['laplacian_weight'][:, 1:2] * Gsrc), edge_index[1], dim=0)

            ret["du_dt"] = G_curr - G_prev  # (N, F)
            if length == 3:
                G_prev_prev = data.x[:, -3, :]
                ret["du2_dt2"] = (G_curr - G_prev) + (G_prev_prev - G_prev)
            else:
                pass
            if self.nophysics_mode == 'zeropad':
                for k in ['gradient', 'laplacian', 'du_dt', 'du2_dt2']:
                    if k in ret.keys():
                        ret[k] = torch.zeros_like(ret[k])
        return ret

    def build_DG(self, data, DC_output, PDE_params):
        """
        Module for generating Derivative Graph.
        It builds a new graph having derivatives and PDE parameters as node-, edge-attributes.
        For instance, if a given PDE is Diffusion Eqn,
        this module will concat node-wise attributes with PDE_params (diffusion-coefficient).

        Input:
            DC_output:
                - Output of derivative_cell()
                - dictionary and key: du_dt, gradient, laplacian, curr
            PDE_params:
                - Output of NN_PDE()
                - Depending on "mode", it may be node-wise or edge-wise features.
            mode: (self)
                - This should be same as "mode" in NN_PDE
        Output:
            output_graph:
                - output_graph.x : curr, laplacian, du_dt
                - output_graph.edge_attr : gradient
                - Additionally, PDE_params will be concatenated properly.
        """

        curr = DC_output["curr"]  # (N, F)

        if self.nophysics_mode == 'nopad':
            if self.use_dist:
                output_graph = replace_graph(data, x=curr, edge_attr=data.edge_dist)
            else:
                output_graph = replace_graph(data, x=curr, edge_attr=None)
        else:
            du_dt = DC_output["du_dt"]  # (N, F)
            gradient = DC_output["gradient"]  # (E, F)
            laplacian = DC_output["laplacian"]  # (N, F)
            if self.mode == "diff":
                node_attr_to_cat = [curr,]
                if self.use_laplacian:
                    node_attr_to_cat.append(laplacian)
                if self.use_time_grad:
                    node_attr_to_cat.append(du_dt)
                if self.use_pde_params:
                    node_attr_to_cat.append(PDE_params)

                edge_attr_to_cat = []
                if self.use_dist:
                    edge_attr_to_cat.append(data.edge_dist)
                if self.use_edge_grad:
                    edge_attr_to_cat.append(gradient)
            elif self.mode == "adv":
                node_attr_to_cat = [curr, ]
                if self.use_laplacian:
                    node_attr_to_cat.append(laplacian)
                if self.use_time_grad:
                    node_attr_to_cat.append(du_dt)

                edge_attr_to_cat = []
                if self.use_dist:
                    edge_attr_to_cat.append(data.edge_dist)
                if self.use_edge_grad:
                    edge_attr_to_cat.append(gradient)
                if self.use_pde_params:
                    edge_attr_to_cat.append(PDE_params)
            else:
                # TODO
                raise NotImplementedError()
            node_attr = torch.cat(node_attr_to_cat, dim=-1)
            if len(edge_attr_to_cat) > 0:
                edge_attr = torch.cat(edge_attr_to_cat, dim=-1)
            else:
                edge_attr = None
            output_graph = replace_graph(data, x=node_attr, edge_attr=edge_attr,
                                         gradient_weight=DC_output['gradient_weight'],
                                         laplacian_weight=DC_output['laplacian_weight'])
        return output_graph

    def NN_PDE(self, DC_output):
        """
        Module for inferring unknown parameters in PDEs.
        For instance, if a given PDE is Diffusion Eqn, this module infers a diffusive coefficient, D, for all nodes.
        If a given PDE is Convection Eqn, this module infers a exteranl vector field, v, for all directions (edges).
        TODO:
            For other equations?

        Input:
            DC_output:
                - output of derivative_cell()
                - It is a dictionary.
        Output:
            Inferred Node-wise parameters or Edge-wise paramters.
        """

        #### Feature engineering
        # du_dt is commonly used.
        # ∇u, Δu, and du2_dt2 are selected based on a given PDE.

        if self.nophysics_mode == 'nopad':
            return None
        if not self.use_pde_params:
            return None

        input_dict = DC_output

        du_dt = input_dict["du_dt"]  # (N, F)
        gradient = input_dict["gradient"]  # (E, F)
        laplacian = input_dict["laplacian"]  # (N, F)
        edge_index = input_dict['edge_index']

        if self.mode == "diff":
            _graph = Data(x=torch.cat([du_dt, laplacian], dim=-1), edge_index=edge_index)

            # output_graph.x is the inferred diffusion-coefficient. (N, F)
            output_graph = self.pde_net(_graph)
            output = output_graph.x  # (N, F)

        elif self.mode == "adv":
            _graph = Data(x=du_dt, edge_attr=gradient, edge_index=edge_index)

            # output_graph.edge_attr is the inferred velocity field. (E, F)
            output_graph = self.pde_net(_graph)
            output = output_graph.edge_attr  # (E, F)

        else:
            # TODO
            pass

        if self.nophysics_mode == 'zeropad':
            output = torch.zeros_like(output)

        return output

    def forward_onestep(self, data, return_features=False):
        if self.predict_model == 'GN':
            if self.is_recurrent:
                if not hasattr(data, 'node_hidden'):
                    data = replace_graph(data,
                        node_hidden=data.x.new_zeros(self.gn_layer_num, self.gn_net.gn_layers[0][1].net.num_layers, data.x.shape[0],
                        self.gn_net.gn_layers[0][1].net.latent_dim))
                if not hasattr(data, 'edge_hidden'):
                    data = replace_graph(data,
                        edge_hidden=data.x.new_zeros(self.gn_layer_num, self.gn_net.gn_layers[0][0].net.num_layers, data.edge_index.shape[1],
                        self.gn_net.gn_layers[0][0].net.latent_dim))

        # One-step prediction
        # Read data (B,T,N,F) and return (B,1,N,output_dim).

        length = data.x.shape[1]  # T

        # 1. Derivative Cell
        DC_output = self.derivative_cell(data, length=length)  # dictionary

        # 2. NN_PDE
        PDE_params = self.NN_PDE(DC_output)  # (N,F) or (E,F)

        # 3. Derivative Graph
        DG_output = self.build_DG(data, DC_output, PDE_params)  # torch_geometric.Data

        DG_output_data = DG_output.clone().apply(lambda x: x.detach())

        # 4. Prediction
        if self.predict_model == 'GN':
            output_graph = self.gn_net(DG_output)  # torch_geometric.Data
        else:
            gradient = DC_output['gradient']
            laplacian = DC_output['laplacian']
            gradient_out = torch.zeros_like(laplacian)
            gradient_out = scatter_add(gradient, DC_output['edge_index'][1, :], dim=0, out=gradient_out)
            dx = gradient_out + laplacian
            output_graph = replace_graph(DG_output, x=dx)

        # 5. Outputs
        output_graph.x = output_graph.x + data.x[:, -1, -self.output_dim:]

        if return_features:
            return output_graph, DG_output_data
        else:
            return output_graph
