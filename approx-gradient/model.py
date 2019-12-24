"""
Graph Network
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min, scatter_mul
from torch_geometric.data import Data

from blocks import EdgeBlock, NodeBlock, GlobalBlock
from modules import GNConv
from utils import decompose_graph


"""
Net : Graph Network - Using local neighbors, directly return edge features
Net2 : Graph Network - Using local neighbors and standard ops, return node features and use standard ops to compute gradients
Net3 : (grad f)_ij = w_ij*f_j - f_i
Net4 : (grad f)_ij = w1_ij*(f_j - w2_ij*f_i)
"""

class Net(nn.Module):

    def __init__(self,
                 node_attr_size,
                 edge_attr_size,
                 out_size,
                 edge_hidden_size=64,
                 node_hidden_size=64,
                 global_hidden_size=64,
                 device='cpu'):
        super(Net, self).__init__()
        
        self.node_input_dim = node_attr_size
        self.edge_input_dim = edge_attr_size
        self.out_dim = out_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.global_h_dim = global_hidden_size
        self.device = device
        
        # Check the dimension. Since the latent representations are concatenated, it is doubled.
        self.eb_custom_func = nn.Sequential(nn.Linear(self.edge_input_dim + self.node_input_dim*2 + self.global_h_dim,
                                                      self.edge_h_dim),
                                            nn.ReLU(),
                                           )
        self.nb_custom_func = nn.Sequential(nn.Linear(self.node_input_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.node_h_dim),
                                            nn.ReLU(),
                                           )
        self.gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.global_h_dim),
                                            nn.ReLU(),
                                           )

        self.eb_module = EdgeBlock(self.edge_input_dim + self.node_input_dim*2 + self.global_h_dim,
                                   self.edge_h_dim,
                                   use_edges=True, 
                                   use_sender_nodes=True, 
                                   use_receiver_nodes=True, 
                                   use_globals=True,
                                   custom_func=self.eb_custom_func)

        self.nb_module = NodeBlock(self.node_input_dim + self.edge_h_dim + self.global_h_dim,
                                   self.node_h_dim,
                                   use_nodes=True,
                                   use_sent_edges=False,
                                   use_received_edges=True,
                                   use_globals=True,
                                   sent_edges_reducer=scatter_add, 
                                   received_edges_reducer=scatter_add,
                                   custom_func=self.nb_custom_func)

        self.gb_module = GlobalBlock(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                     self.global_h_dim,
                                     edge_reducer=scatter_mean,
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)
        
        self.gn1 = GNConv(self.eb_module, 
                          self.nb_module, 
                          self.gb_module, 
                          use_edge_block=True, 
                          use_node_block=True, 
                          use_global_block=True)
        
        #### Edge-update
        self.edge_dec = nn.Sequential(nn.Linear(self.edge_h_dim + self.node_h_dim*2 + self.global_h_dim,
                                                self.edge_h_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.edge_h_dim, self.out_dim)
                                     )
        
        self.edge_dec_block = EdgeBlock(self.edge_h_dim + self.node_h_dim*2 + self.global_h_dim,
                                        self.out_dim,
                                        use_edges=True, 
                                        use_sender_nodes=True, 
                                        use_receiver_nodes=True, 
                                        use_globals=True,
                                        custom_func=self.edge_dec)
        
        self.gn2 = GNConv(self.edge_dec_block,
                          None,
                          None,
                          use_edge_block=True,
                          use_node_block=False,
                          use_global_block=False)
        
    def forward(self, data):

        middle_graph = self.gn1(data)
        output_graph = self.gn2(middle_graph)
                
        return output_graph


    
class Net2(nn.Module):

    def __init__(self,
                 node_attr_size,
                 edge_attr_size,
                 out_size,
                 edge_hidden_size=64,
                 node_hidden_size=64,
                 global_hidden_size=64,
                 device='cpu'):
        super(Net2, self).__init__()
        
        self.node_input_dim = node_attr_size
        self.edge_input_dim = edge_attr_size
        self.out_dim = out_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.global_h_dim = global_hidden_size
        self.device = device
        
        # Check the dimension. Since the latent representations are concatenated, it is doubled.
        self.eb_custom_func = nn.Sequential(nn.Linear(self.edge_input_dim + self.node_input_dim*2 + self.global_h_dim,
                                                      self.edge_h_dim),
                                            nn.ReLU(),
                                           )
        self.nb_custom_func = nn.Sequential(nn.Linear(self.node_input_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.node_h_dim),
                                            nn.ReLU(),
                                           )
        self.gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.global_h_dim),
                                            nn.ReLU(),
                                           )

        self.eb_module = EdgeBlock(self.edge_input_dim + self.node_input_dim*2 + self.global_h_dim,
                                   self.edge_h_dim,
                                   use_edges=True, 
                                   use_sender_nodes=True, 
                                   use_receiver_nodes=True, 
                                   use_globals=True,
                                   custom_func=self.eb_custom_func)

        self.nb_module = NodeBlock(self.node_input_dim + self.edge_h_dim + self.global_h_dim,
                                   self.node_h_dim,
                                   use_nodes=True,
                                   use_sent_edges=False,
                                   use_received_edges=True,
                                   use_globals=True,
                                   sent_edges_reducer=scatter_add, 
                                   received_edges_reducer=scatter_add,
                                   custom_func=self.nb_custom_func)

        self.gb_module = GlobalBlock(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                     self.global_h_dim,
                                     edge_reducer=scatter_mean,
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)
        
        self.gn1 = GNConv(self.eb_module, 
                          self.nb_module, 
                          self.gb_module, 
                          use_edge_block=True, 
                          use_node_block=True, 
                          use_global_block=True)
        
        #### Node-update
        self.node_dec = nn.Sequential(nn.Linear(self.edge_h_dim*2 + self.node_h_dim + self.global_h_dim,
                                                self.node_h_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.node_h_dim, self.out_dim)
                                     )
        
        self.node_dec_block = NodeBlock(self.edge_h_dim*2 + self.node_h_dim + self.global_h_dim,
                                        self.out_dim,
                                        use_nodes=True, 
                                        use_sent_edges=True, 
                                        use_received_edges=True, 
                                        use_globals=True,
                                        sent_edges_reducer=scatter_mean, 
                                        received_edges_reducer=scatter_mean,
                                        custom_func=self.node_dec)
        
        self.gn2 = GNConv(None,
                          self.node_dec_block,
                          None,
                          use_edge_block=False,
                          use_node_block=True,
                          use_global_block=False)
        
    def forward(self, data):

        middle_graph = self.gn1(data)
        output_graph = self.gn2(middle_graph)
                
        return output_graph
    
    
    
class Net3(nn.Module):

    def __init__(self,
                 node_attr_size,
                 edge_attr_size,
                 out_size,
                 edge_hidden_size=64,
                 node_hidden_size=64,
                 global_hidden_size=64,
                 device='cpu'):
        super(Net3, self).__init__()
        
        self.node_input_dim = node_attr_size
        self.edge_input_dim = edge_attr_size
        self.out_dim = out_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.global_h_dim = global_hidden_size
        self.device = device
        
        #### For w_ij
        self.eb_custom_func = nn.Sequential(nn.Linear(self.edge_input_dim + self.node_input_dim*2 + self.global_h_dim,
                                                      self.edge_h_dim),
                                            nn.ReLU(),
                                           )
        self.nb_custom_func = nn.Sequential(nn.Linear(self.node_input_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.node_h_dim),
                                            nn.ReLU(),
                                           )
        self.gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                                      self.global_h_dim),
                                            nn.ReLU(),
                                           )

        self.eb_module = EdgeBlock(self.edge_input_dim + self.node_input_dim*2 + self.global_h_dim,
                                   self.edge_h_dim,
                                   use_edges=True, 
                                   use_sender_nodes=True, 
                                   use_receiver_nodes=True, 
                                   use_globals=True,
                                   custom_func=self.eb_custom_func)

        self.nb_module = NodeBlock(self.node_input_dim + self.edge_h_dim + self.global_h_dim,
                                   self.node_h_dim,
                                   use_nodes=True,
                                   use_sent_edges=False,
                                   use_received_edges=True,
                                   use_globals=True,
                                   sent_edges_reducer=scatter_add, 
                                   received_edges_reducer=scatter_add,
                                   custom_func=self.nb_custom_func)

        self.gb_module = GlobalBlock(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                     self.global_h_dim,
                                     edge_reducer=scatter_mean,
                                     node_reducer=scatter_mean,
                                     custom_func=self.gb_custom_func,
                                     device=device)
        
        self.gn1 = GNConv(self.eb_module,
                          self.nb_module,
                          self.gb_module,
                          use_edge_block=True, 
                          use_node_block=True, 
                          use_global_block=True)
        
        #### Edge-update
        self.edge_dec = nn.Sequential(nn.Linear(self.edge_h_dim + self.node_h_dim*2 + self.global_h_dim,
                                                self.edge_h_dim),
                                      nn.ReLU(),
                                      nn.Linear(self.edge_h_dim, self.out_dim)
                                     )
        
        self.edge_dec_block = EdgeBlock(self.edge_h_dim + self.node_h_dim*2 + self.global_h_dim,
                                        self.out_dim,
                                        use_edges=True, 
                                        use_sender_nodes=True, 
                                        use_receiver_nodes=True, 
                                        use_globals=True,
                                        custom_func=self.edge_dec)
        
        self.gn2 = GNConv(self.edge_dec_block,
                          None,
                          None,
                          use_edge_block=True,
                          use_node_block=False,
                          use_global_block=False)
        
        
    def forward(self, data):
        _x, _edge_index, _, _ = decompose_graph(data)
        
        middle_graph = self.gn1(data)
        output_graph = self.gn2(middle_graph)

        _sind, _rind = _edge_index
        ret = output_graph.edge_attr*_x[_rind,2:3] - _x[_sind,2:3]    # 3rd column is for values
                
        return ret
    

class Net4(nn.Module):

    def __init__(self,
                 node_attr_size,
                 edge_attr_size,
                 out_size,
                 edge_hidden_size=64,
                 node_hidden_size=64,
                 global_hidden_size=64,
                 device='cpu'):
        super(Net4, self).__init__()
        
        self.node_input_dim = node_attr_size
        self.edge_input_dim = edge_attr_size
        self.out_dim = out_size
        self.edge_h_dim = edge_hidden_size
        self.node_h_dim = node_hidden_size
        self.global_h_dim = global_hidden_size
        self.device = device
        
        #### For w1_ij
        self.w1_eb_custom_func = nn.Sequential(nn.Linear(self.edge_input_dim + self.node_input_dim*2 + self.global_h_dim,
                                                        self.edge_h_dim),
                                               nn.ReLU(),
                                              )
        self.w1_nb_custom_func = nn.Sequential(nn.Linear(self.node_input_dim + self.edge_h_dim + self.global_h_dim,
                                                         self.node_h_dim),
                                               nn.ReLU(),
                                              )
        self.w1_gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                                         self.global_h_dim),
                                               nn.ReLU(),
                                              )

        self.w1_eb_module = EdgeBlock(self.edge_input_dim + self.node_input_dim*2 + self.global_h_dim,
                                      self.edge_h_dim,
                                      use_edges=True, 
                                      use_sender_nodes=True, 
                                      use_receiver_nodes=True, 
                                      use_globals=True,
                                      custom_func=self.w1_eb_custom_func)

        self.w1_nb_module = NodeBlock(self.node_input_dim + self.edge_h_dim + self.global_h_dim,
                                      self.node_h_dim,
                                      use_nodes=True,
                                      use_sent_edges=False,
                                      use_received_edges=True,
                                      use_globals=True,
                                      sent_edges_reducer=scatter_add, 
                                      received_edges_reducer=scatter_add,
                                      custom_func=self.w1_nb_custom_func)

        self.w1_gb_module = GlobalBlock(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                        self.global_h_dim,
                                        edge_reducer=scatter_mean,
                                        node_reducer=scatter_mean,
                                        custom_func=self.w1_gb_custom_func,
                                        device=device)
        
        self.w1_gn1 = GNConv(self.w1_eb_module, 
                             self.w1_nb_module, 
                             self.w1_gb_module, 
                             use_edge_block=True, 
                             use_node_block=True, 
                             use_global_block=True)
        
        #### Edge-update
        self.w1_edge_dec = nn.Sequential(nn.Linear(self.edge_h_dim + self.node_h_dim*2 + self.global_h_dim,
                                                   self.edge_h_dim),
                                         nn.ReLU(),
                                         nn.Linear(self.edge_h_dim, self.out_dim)
                                        )
        
        self.w1_edge_dec_block = EdgeBlock(self.edge_h_dim + self.node_h_dim*2 + self.global_h_dim,
                                           self.out_dim,
                                           use_edges=True, 
                                           use_sender_nodes=True, 
                                           use_receiver_nodes=True, 
                                           use_globals=True,
                                           custom_func=self.w1_edge_dec)
        
        self.w1_gn2 = GNConv(self.w1_edge_dec_block,
                             None,
                             None,
                             use_edge_block=True,
                             use_node_block=False,
                             use_global_block=False)
        
        #### For w2_ij
        self.w2_eb_custom_func = nn.Sequential(nn.Linear(self.edge_input_dim + self.node_input_dim*2 + self.global_h_dim,
                                                        self.edge_h_dim),
                                               nn.ReLU(),
                                              )
        self.w2_nb_custom_func = nn.Sequential(nn.Linear(self.node_input_dim + self.edge_h_dim + self.global_h_dim,
                                                         self.node_h_dim),
                                               nn.ReLU(),
                                              )
        self.w2_gb_custom_func = nn.Sequential(nn.Linear(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                                         self.global_h_dim),
                                               nn.ReLU(),
                                              )

        self.w2_eb_module = EdgeBlock(self.edge_input_dim + self.node_input_dim*2 + self.global_h_dim,
                                      self.edge_h_dim,
                                      use_edges=True, 
                                      use_sender_nodes=True, 
                                      use_receiver_nodes=True, 
                                      use_globals=True,
                                      custom_func=self.w2_eb_custom_func)

        self.w2_nb_module = NodeBlock(self.node_input_dim + self.edge_h_dim + self.global_h_dim,
                                      self.node_h_dim,
                                      use_nodes=True,
                                      use_sent_edges=False,
                                      use_received_edges=True,
                                      use_globals=True,
                                      sent_edges_reducer=scatter_add, 
                                      received_edges_reducer=scatter_add,
                                      custom_func=self.w2_nb_custom_func)

        self.w2_gb_module = GlobalBlock(self.node_h_dim + self.edge_h_dim + self.global_h_dim,
                                        self.global_h_dim,
                                        edge_reducer=scatter_mean,
                                        node_reducer=scatter_mean,
                                        custom_func=self.w2_gb_custom_func,
                                        device=device)
        
        self.w2_gn1 = GNConv(self.w2_eb_module, 
                             self.w2_nb_module, 
                             self.w2_gb_module, 
                             use_edge_block=True, 
                             use_node_block=True, 
                             use_global_block=True)
        
        #### Edge-update
        self.w2_edge_dec = nn.Sequential(nn.Linear(self.edge_h_dim + self.node_h_dim*2 + self.global_h_dim,
                                                   self.edge_h_dim),
                                         nn.ReLU(),
                                         nn.Linear(self.edge_h_dim, self.out_dim)
                                        )
        
        self.w2_edge_dec_block = EdgeBlock(self.edge_h_dim + self.node_h_dim*2 + self.global_h_dim,
                                           self.out_dim,
                                           use_edges=True, 
                                           use_sender_nodes=True, 
                                           use_receiver_nodes=True, 
                                           use_globals=True,
                                           custom_func=self.w2_edge_dec)
        
        self.w2_gn2 = GNConv(self.w2_edge_dec_block,
                             None,
                             None,
                             use_edge_block=True,
                             use_node_block=False,
                             use_global_block=False)
        
    def forward(self, data):
        _x, _edge_index, _, _ = decompose_graph(data)
        
        w1_middle_graph = self.w1_gn1(data)
        w1_output_graph = self.w1_gn2(w1_middle_graph)
        
        w2_middle_graph = self.w2_gn1(data)
        w2_output_graph = self.w2_gn2(w2_middle_graph)

        _sind, _rind = _edge_index
        ret = w1_output_graph.edge_attr*(_x[_rind,2:3] - w2_output_graph.edge_attr*_x[_sind,2:3])    # 3rd column is for values
        
        
        return ret