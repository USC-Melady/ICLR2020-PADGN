from __future__ import division
from __future__ import print_function

import os
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)
import itertools
import math

import numpy as np
from torch_geometric.data import Data
import torch
import torch.nn as nn
from torch_scatter import scatter_add
from scipy import sparse

from models.utils import SeriesModel, replace_graph, make_mlp
from models.GraphPDE import build_gn_net_nodeout


class LinearRegOp(SeriesModel):
    def __init__(self,
                 input_dim,
                 output_dim,
                 input_frame_num,
                 skip_first_frames_num,
                 node_meta_dim=0,
                 order=2,
                 node_num=250,
                 optype='standard',
                 mesh_matrices=None,
                 prediction_model='linear',
                 prediction_net_hidden_dim=64,
                 prediction_net_layer_num=2,
                 prediction_net_is_recurrent=True,
                 batchnorm=None):
        '''
        LinearRegOp Model: linear regressor on specified orders of operators
        :param input_dim: Dimension of input features
        :param output_dim: Dimension of output features
        :param input_frame_num: Number of frames needed to produce 1 prediction
        :param skip_first_frames_num: Number of first frames without prediction (use ground truth for them)
        :param node_meta_dim:
        :param order:
                the highest order number
                (1: gradient, 2: laplacian, 3: gradient of laplacian, 4: laplaican of laplacian...)
        :param node_num: number of nodes (each node has its own parameter in regression)
        :param optype: 'standard' or 'trimesh'
            'standard': gradient and laplacian defined on discrete graphs
            'trimesh': having coefs related to triangular mesh in addition to 'standard'
        '''
        detach_last_step = (optype == 'trimesh') # Triangular mesh based ops are not stable for backward propagation of multiple steps
        # detach_last_step = True
        super(LinearRegOp, self).__init__(input_dim, output_dim, input_frame_num,
                                   skip_first_frames_num, is_recurrent=False, detach_last_step=detach_last_step)
        print('Detach last step', self.detach_last_step)
        self.node_meta_dim = node_meta_dim
        self.order = order
        self.optype = optype
        self.node_num = node_num
        self.prediction_model = prediction_model

        if optype == 'trimesh':
            self.mesh_tensors = {}
            for k, v in mesh_matrices.items():
                if v.shape == ():
                    self.mesh_tensors[k] = sparse2tensor(v.item().tocoo()).float()
                else:
                    self.mesh_tensors[k] = torch.from_numpy(v).float()
                # # calculate averaged XN, YN on each node
                # XN, YN, F2V = self.mesh_tensors['XN'], self.mesh_tensors['YN'], self.mesh_tensors['F2V']
                # if len(XN.shape) == 1: # all nodes on the same plain
                #     XN_vert, YN_vert = XN, YN
                # else:
                #     # XN.shape = [Facenum, 3]
                #     XN_vert = spmatmul(XN.permute(1, 0).unsqueeze(0), F2V).squeeze(0).permute(1, 0) # N x 3
                #     YN_vert = spmatmul(YN.permute(1, 0).unsqueeze(0), F2V).squeeze(0).permute(1, 0) # N x 3
                # self.mesh_tensors['XN_vert'] = XN_vert
                # self.mesh_tensors['YN_vert'] = YN_vert

        if self.prediction_model == 'linear':
            if order % 2 == 0:
                self.parameterized_order_num = (order // 2) * 3
            else:
                self.parameterized_order_num = ((order - 1) // 2) * 3 + 2
            self.reg_param = nn.Parameter(torch.Tensor(node_num, self.parameterized_order_num, self.input_dim))
            nn.init.kaiming_uniform_(self.reg_param, a=math.sqrt(5))
        elif self.prediction_model == 'SingleMLP':
            self.prediction_net_hidden_dim = prediction_net_hidden_dim
            self.prediction_net = make_mlp(self.input_dim * 4, self.prediction_net_hidden_dim, self.output_dim,
                                           layer_num=1, activation='ReLU', final_activation=False)
        elif self.prediction_model in ['GN', 'RGN']:
            self.prediction_net_hidden_dim = prediction_net_hidden_dim
            self.prediction_net_layer_num = prediction_net_layer_num
            self.prediction_net_is_recurrent = prediction_net_is_recurrent
            gn_update_func_layer_num = 2
            if self.optype == 'standard':
                first_gn_node_dim_in = self.input_dim * 2
                first_gn_edge_dim_in = self.input_dim
            elif self.optype == 'trimesh':
                first_gn_node_dim_in = self.input_dim * 4
                first_gn_edge_dim_in = 0
            if self.prediction_model == 'RGN':
                self.prediction_net = build_gn_net_nodeout(
                    first_gn_node_dim_in=first_gn_node_dim_in,
                    first_gn_edge_dim_in=first_gn_edge_dim_in,
                    node_out_dim=self.output_dim,
                    hidden_dim_gn=self.prediction_net_hidden_dim,
                    layer_num=self.prediction_net_layer_num,
                    update_func_layer_num=gn_update_func_layer_num,
                    is_recurrent=self.prediction_net_is_recurrent,
                    batchnorm=batchnorm
                )
        else:
            raise NotImplementedError()

    def forward_onestep(self, data, return_features=False):
        if self.prediction_model == 'linear':
            return self.forward_onestep_linear(data, return_features)
        elif self.prediction_model == 'SingleMLP':
            return self.forward_onestep_singlemlp(data, return_features)
        elif self.prediction_model in ['GN', 'RGN']:
            return self.forward_onestep_gn_rgn(data, return_features)

    def forward_onestep_linear(self, data, return_features=False):
        data_input  = replace_graph(data, x=data.x[:, -1, :])
        if self.node_meta_dim > 0:
            node_meta = data.node_meta # should be N x 2 (x, y)
        else:
            node_meta = None

        op_calc = self._calculate_op(data_input, node_meta, self.optype) # N x self.parameterized_order_num x input_dim
        op_calc = op_calc['op_agg_msgs']
        batch_op_calc = op_calc.view(-1, self.node_num, self.parameterized_order_num, self.input_dim)
        delta_input = torch.sum(batch_op_calc * self.reg_param.unsqueeze(0), dim=2) # B x N x F
        delta_input = delta_input.view(-1, self.input_dim)
        next_x = data_input.x + delta_input
        output_graph = replace_graph(data_input, x=next_x)

        if return_features:
            DG_output_data = replace_graph(data_input,
                                           x=data_input.x,
                                           edge_attr=None,
                                           gradient_weight=self.reg_param.detach().data,
                                           laplacian_weight=None)
            return output_graph, DG_output_data
        else:
            return output_graph

    def forward_onestep_singlemlp(self, data, return_features=False):
        data_input = replace_graph(data, x=data.x[:, -1, :]) # N x F
        if self.node_meta_dim > 0:
            node_meta = data.node_meta # should be N x 2 (x, y)
        else:
            node_meta = None

        op_calc = self._calculate_op(data_input, node_meta, self.optype) # N x self.parameterized_order_num x input_dim
        op_calc = op_calc['op_agg_msgs'] # N x 3 x F

        net_input = torch.cat((data_input.x.unsqueeze(1), op_calc), dim=1).view(-1, 4 * self.input_dim) # N x (4 x F)
        delta_input = self.prediction_net(net_input) # N x F
        next_x = data_input.x + delta_input
        output_graph = replace_graph(data_input, x=next_x)
        if return_features:
            raise NotImplementedError()
        else:
            return output_graph

    def forward_onestep_gn_rgn(self, data, return_features=False):
        if self.prediction_net_is_recurrent and (self.prediction_model == 'RGN'):
            if not hasattr(data, 'edge_hidden_prediction_net'):
                data = replace_graph(data,
                                     edge_hidden_prediction_net=data.x.new_zeros(self.prediction_net_layer_num,
                                                                                 self.prediction_net.gn_layers[0][
                                                                                     0].net.num_layers,
                                                                                 data.edge_index.shape[1],
                                                                                 self.prediction_net.gn_layers[0][
                                                                                     0].net.latent_dim))
            if not hasattr(data, 'node_hidden_prediction_net'):
                data = replace_graph(data,
                                     node_hidden_prediction_net=data.x.new_zeros(self.prediction_net_layer_num,
                                                                                 self.prediction_net.gn_layers[0][
                                                                                     1].net.num_layers,
                                                                                 data.x.shape[0],
                                                                                 self.prediction_net.gn_layers[0][
                                                                                     1].net.latent_dim))

        data_input = replace_graph(data, x=data.x[:, -1, :])
        if self.node_meta_dim > 0:
            node_meta = data.node_meta # should be N x 2 (x, y)
        else:
            node_meta = None
        op_calc = self._calculate_op(data_input, node_meta, self.optype)

        if self.optype == 'standard':
            new_x_input = torch.cat((data_input.x.unsqueeze(1), op_calc['op_agg_msgs'][:, 2:3, :]), dim=1).flatten(-2, -1)
            new_ea_input = op_calc['op_msgs'][:, 0:1, :].flatten(-2, -1)
        elif self.optype == 'trimesh':
            new_x_input = torch.cat((data_input.x.unsqueeze(1), op_calc['op_agg_msgs']), dim=1).flatten(-2, -1)
            new_ea_input = None
        else:
            raise NotImplementedError()
        prediction_input_graph = replace_graph(data_input,
                                               x=new_x_input, edge_attr=new_ea_input)
        # print(torch.any(torch.isnan(data_input.x)), torch.any(torch.isnan(op_calc['op_agg_msgs'][:, 0, :])),
        #       torch.any(torch.isnan(op_calc['op_agg_msgs'][:, 1, :])),
        #       torch.any(torch.isnan(op_calc['op_agg_msgs'][:, 2, :])))
        # print(prediction_input_graph.x.shape, prediction_input_graph.edge_attr.shape)
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
            raise NotImplementedError()
        else:
            return output_graph

    def _calculate_op(self, data_input, node_coords, optype):
        if optype == 'standard':
            op_msgs = []
            op_agg_msgs = []
            last_op_agg_msg = data_input.x  # N x F
            edge_index = data_input.edge_index  # 2 x E
            for order_i in range(self.order):
                op_msg_src, op_msg_dst = last_op_agg_msg[edge_index[0]], last_op_agg_msg[
                    edge_index[1]]  # E x F
                op_msg = op_msg_dst - op_msg_src # E x F
                op_msgs.append(op_msg)
                if node_coords is None:
                    # op_agg_msg = scatter_add(op_msg, edge_index[1], dim=0)
                    # last_op_agg_msg = op_agg_msg
                    raise NotImplementedError()
                else:
                    if order_i == 0:
                        edge_vec = node_coords[edge_index[1, :]] - node_coords[edge_index[0, :]] # E x 2
                        edge_vec = edge_vec / torch.norm(edge_vec, dim=1, keepdim=True)
                        edge_vec[torch.isnan(edge_vec)] = 0
                        node_msg_g_vec = edge_vec.unsqueeze(1) * op_msg.unsqueeze(2) # E x F x 2
                        node_agg_msg_g = scatter_add(node_msg_g_vec, edge_index[1], dim=0) # N x F x 2
                    elif order_i == 1:
                        node_agg_msg_g = scatter_add(op_msg, edge_index[1], dim=0).unsqueeze(2) # N x F x 1
                    else:
                        raise NotImplementedError('Has not implemented order {:d}'.format(order_i))
                    op_agg_msgs.append(node_agg_msg_g)
            op_msgs = torch.stack(op_msgs, dim=-1).transpose(1, 2) # E x 1 x F
            op_agg_msgs = torch.cat(op_agg_msgs, dim=-1).transpose(1, 2) # N x 3 x F
        elif optype == 'trimesh':
            for k in self.mesh_tensors.keys():
                self.mesh_tensors[k] = self.mesh_tensors[k].to(data_input.x.device)
            for cpu_key in ['G', 'XN', 'YN', 'F2V']:
                self.mesh_tensors[cpu_key] = self.mesh_tensors[cpu_key].to('cpu')
            G, L, XN, YN, F2V = self.mesh_tensors['G'], self.mesh_tensors['L'], \
                self.mesh_tensors['XN'], self.mesh_tensors['YN'], self.mesh_tensors['F2V']
            batch_x = data_input.x.view(-1, self.node_num, self.input_dim).permute(0, 2, 1) # B x F x N
            laplacian = spmatmul(batch_x, L)  # B x F x N
            # calculate spmatmul part on CPU (otherwise sometimes it gives nan)
            batch_x = batch_x.to('cpu')
            # G = G.to('cpu')
            # XN, YN, F2V = XN.to('cpu'), YN.to('cpu'), F2V.to('cpu')
            grad_face = spmatmul(batch_x, G) # B x F x 3FN
            grad_face = grad_face.view(*((batch_x.shape)[:2]), 3, -1).permute(0, 1, 3, 2) # B x F x FN x 3
            # grad_face[torch.isnan(grad_face)] = 0
            # print('grad_face', torch.any(torch.isnan(grad_face)))
            grad_face_x = torch.sum(torch.mul(grad_face, XN), dim=-1)
            # print('grad_face_x', torch.any(torch.isnan(grad_face_x)))
            grad_face_y = torch.sum(torch.mul(grad_face, YN), dim=-1)
            grad_vert_x = spmatmul(grad_face_x, F2V) # B x F x N
            # print('grad_vert_x', torch.any(torch.isnan(grad_vert_x)))
            grad_vert_y = spmatmul(grad_face_y, F2V) # B x F x N
            grad_vert_x, grad_vert_y = grad_vert_x.to(data_input.x.device), grad_vert_y.to(data_input.x.device)
            op_agg_msgs = torch.stack((grad_vert_x, grad_vert_y, laplacian), dim=2) # B x F x 3 x N
            op_agg_msgs = op_agg_msgs.permute(0, 3, 2, 1).contiguous().view(-1, 3, self.input_dim) # N x 3 x F
            op_msgs = None
            # print('gradx', np.where(np.isinf(grad_vert_x.data)),
            #       'grady', np.where(np.isinf(grad_vert_y.data)),
            #       'lap', np.where(np.isinf(laplacian.data)))
            # print('op', np.where(np.isinf(op_agg_msgs)))
        else:
            raise NotImplementedError()
        return {'op_msgs': op_msgs, 'op_agg_msgs': op_agg_msgs}


def sparse2tensor(m):
    """
    Convert sparse matrix (scipy.sparse) to tensor (torch.sparse)
    """
    assert(isinstance(m, sparse.coo.coo_matrix))
    i = torch.LongTensor([m.row, m.col])
    v = torch.FloatTensor(m.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(m.shape))


def spmatmul(den, sp):
    """
    den: Dense tensor of shape batch_size x in_chan x #V
    sp : Sparse tensor of shape newlen x #V
    """
    batch_size, in_chan, nv = list(den.size())
    new_len = sp.size()[0]
    den = den.permute(2, 1, 0).contiguous().view(nv, -1)
    res = torch.spmm(sp, den).view(new_len, in_chan, batch_size).contiguous().permute(2, 1, 0)
    return res

