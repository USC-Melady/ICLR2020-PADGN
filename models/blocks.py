import os
import sys
# dirty hack: include top level folder to path
sys.path.insert(0,
    os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
)

import torch
import torch.nn as nn

from torch_scatter import scatter_add, scatter_max, scatter_mean, scatter_min, scatter_mul

from utils.utils import decompose_graph
from models.utils import make_mlp

LATENT_SIZE = 32


class RecurrentUpdateNet(nn.Module):
    def __init__(self, in_features, latent_dim, out_features, num_layers, activation='ReLU',
                 final_activation=False, batchnorm=None):
        super(RecurrentUpdateNet, self).__init__()
        self.in_features = in_features
        self.latent_dim = latent_dim
        self.out_features = out_features
        self.num_layers = num_layers

        self.rnn = nn.GRU(
                    input_size=in_features,
                    hidden_size=latent_dim,
                    num_layers=num_layers,
                    batch_first=True
                )
        if final_activation:
            self.decoder = [nn.Linear(latent_dim, out_features),]
            if batchnorm == 'LayerNorm':
                self.decoder.append(nn.LayerNorm(out_features))
            elif batchnorm == 'BatchNorm':
                self.decoder.append(nn.BatchNorm1d(out_features))
            self.decoder.append(getattr(nn, activation)())
            print(self.decoder)
            self.decoder = nn.Sequential(*self.decoder)
        else:
            self.decoder = nn.Linear(latent_dim, out_features)

    def forward(self, input_feature, hidden_feature):
        '''

        :param input_feature: [N, in_features]
        :param hidden_feature: [num_layers, N, latent_dim]
        :return:
        '''
        input_feature = input_feature.unsqueeze(1)
        out, hidden_feature = self.rnn(input_feature, hidden_feature)
        out = self.decoder(out.squeeze(1))
        return out, hidden_feature


class GlobalBlock(nn.Module):
    """Global block, f_g.
    
    A block that updates the global features of each graph based on
    the previous global features, the aggregated features of the
    edges of the graph, and the aggregated features of the nodes of the graph.
    """

    def __init__(self,
                 in_features,
                 out_features,
                 use_edges=True,
                 use_nodes=True,
                 use_globals=True,
                 edge_reducer=scatter_mean,
                 node_reducer=scatter_mean,
                 custom_func=None,
                 recurrent=False):
        
        super(GlobalBlock, self).__init__()
        
        if not (use_nodes or use_edges or use_globals):
            raise ValueError("At least one of use_edges, "
                             "use_nodes or use_globals must be True.")
    
        self._use_edges = use_edges    # not need to differentiate sent/received edges.
        self._use_nodes = use_nodes
        self._use_globals = use_globals
        self._edge_reducer = edge_reducer
        self._node_reducer = node_reducer
        self.recurrent = recurrent
        
        # f_g is a function R^in_features -> R^out_features
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            if self.recurrent:
                self.net = RecurrentUpdateNet(
                    in_features=in_features,
                    latent_dim=LATENT_SIZE,
                    out_features=out_features,
                    num_layers=2
                )
            else:
                self.net = nn.Sequential(nn.Linear(in_features, LATENT_SIZE),
                                         nn.ReLU(),
                                         nn.Linear(LATENT_SIZE, out_features),
                                        )
    
    def forward(self, graph):
        # Decompose graph
        node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        num_edges = graph.num_edges
        num_nodes = graph.num_nodes
        
        globals_to_collect = []
        
        if self._use_globals:
            globals_to_collect.append(global_attr)    # global_attr.shape=(1, d_g)
            
        if self._use_edges:
            # no need to differentiate sent/received edges.
            try:
                agg_edges = self._edge_reducer(edge_attr, edge_attr.new_zeros(num_edges, dtype=torch.long), dim=0)
            except:
                raise ValueError("reducer should be one of scatter_* [add, mul, max, min, mean]")
            globals_to_collect.append(agg_edges)
        
        if self._use_nodes:
            try:
                agg_nodes = self._node_reducer(node_attr, node_attr.new_zeros(num_nodes, dtype=torch.long), dim=0)
            except:
                raise ValueError("reducer should be one of scatter_* [add, mul, max, min, mean]")
            globals_to_collect.append(agg_nodes)
        
        collected_globals = torch.cat(globals_to_collect, dim=-1)

        if self.recurrent:
            graph.global_attr, graph.global_hidden = self.net(collected_globals, graph.global_hidden)
        else:
            graph.global_attr = self.net(collected_globals)    # Update
        
        return graph


class EdgeBlock(nn.Module):
    """Edge block, f_e.
    Update the features of each edge based on the previous edge features,
    the features of the adjacent nodes, and the global features.
    """
    
    def __init__(self,
                 in_features,
                 out_features,
                 use_edges=True,
                 use_sender_nodes=True,
                 use_receiver_nodes=True,
                 use_globals=True,
                 custom_func=None,
                 recurrent=False):
        
        super(EdgeBlock, self).__init__()
        
        if not (use_edges or use_sender_nodes or use_receiver_nodes or use_globals):
            raise ValueError("At least one of use_edges, use_sender_nodes, "
                             "use_receiver_nodes or use_globals must be True.")
        
        self._use_edges = use_edges
        self._use_sender_nodes = use_sender_nodes
        self._use_receiver_nodes = use_receiver_nodes
        self._use_globals = use_globals
        self.recurrent = recurrent
    
        # f_e() is a function: R^in_features -> R^out_features
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            if self.recurrent:
                self.net = RecurrentUpdateNet(
                    in_features=in_features,
                    latent_dim=LATENT_SIZE,
                    out_features=out_features,
                    num_layers=2
                )
            else:
                self.net = nn.Sequential(nn.Linear(in_features, LATENT_SIZE),
                                         nn.ReLU(),
                                         nn.Linear(LATENT_SIZE, out_features),
                                        )

    def forward(self, graph):
        # Decompose graph
        node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        num_edges = graph.num_edges
        
        edges_to_collect = []
        
        if self._use_edges:
            edges_to_collect.append(edge_attr)
            
        if self._use_sender_nodes:
            senders_attr = node_attr[senders_idx, :]
            edges_to_collect.append(senders_attr)
            
        if self._use_receiver_nodes:
            receivers_attr = node_attr[receivers_idx, :]
            edges_to_collect.append(receivers_attr)
        
        if self._use_globals:
            expanded_global_attr = global_attr.expand(num_edges, global_attr.shape[1])
            edges_to_collect.append(expanded_global_attr)
            
        collected_edges = torch.cat(edges_to_collect, dim=-1)

        if self.recurrent:
            graph.edge_attr, graph.edge_hidden = self.net(collected_edges, graph.edge_hidden)
        else:
            graph.edge_attr = self.net(collected_edges)    # Update
        
        return graph
#         return self.net(collected_edges)    # (N^e, d_e+(2*d_v)+d_g) -> (N^e, out_features)
    
    
class NodeBlock(nn.Module):
    """Node block, f_v.
    Update the features of each node based on the previous node features,
    the aggregated features of the received edges,
    the aggregated features of the sent edges, and the global features.
    """
    
    def __init__(self,
                 in_features,
                 out_features,
                 use_nodes=True,
                 use_sent_edges=False,
                 use_received_edges=True,
                 use_globals=True,
                 sent_edges_reducer=scatter_add,
                 received_edges_reducer=scatter_add,
                 custom_func=None,
                 recurrent=False):
        """Initialization of the NodeBlock module.
        
        Args:
            in_features: Input dimension.
                If node, 2*edge(sent, received), and global are used, d_v+(2*d_e)+d_g.
                h'_i = f_v(h_i, AGG(h_ij), AGG(h_ji), u)
            out_features: Output dimension.
                h'_i will have the dimension.
            use_nodes: Whether to condition on node attributes.
            use_sent_edges: Whether to condition on sent edges attributes.
            use_received_edges: Whether to condition on received edges attributes.
            use_globals: Whether to condition on the global attributes.
            reducer: Aggregator. scatter_* [add, mul, max, min, mean]
        """
        
        super(NodeBlock, self).__init__()

        if not (use_nodes or use_sent_edges or use_received_edges or use_globals):
            raise ValueError("At least one of use_received_edges, use_sent_edges, "
                             "use_nodes or use_globals must be True.")
        
        self._use_nodes = use_nodes
        self._use_sent_edges = use_sent_edges
        self._use_received_edges = use_received_edges
        self._use_globals = use_globals
        self._sent_edges_reducer = sent_edges_reducer
        self._received_edges_reducer = received_edges_reducer
        self.recurrent = recurrent
        
        # f_v() is a function: R^in_features -> R^out_features
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            if self.recurrent:
                self.net = RecurrentUpdateNet(
                    in_features=in_features,
                    latent_dim=LATENT_SIZE,
                    out_features=out_features,
                    num_layers=2
                )
            else:
                self.net = nn.Sequential(nn.Linear(in_features, LATENT_SIZE),
                                         nn.ReLU(),
                                         nn.Linear(LATENT_SIZE, out_features),
                                        )

    def forward(self, graph):
        # Decompose graph
        node_attr, edge_index, edge_attr, global_attr = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        num_nodes = graph.num_nodes
        
        nodes_to_collect = []
        
        if self._use_nodes:
            nodes_to_collect.append(node_attr)
            
        if self._use_sent_edges:
            try:
                agg_sent_edges = self._sent_edges_reducer(edge_attr, senders_idx, dim=0, dim_size=num_nodes)
            except:
                raise ValueError("reducer should be one of scatter_* [add, mul, max, min, mean]")
            
            nodes_to_collect.append(agg_sent_edges)
            
        if self._use_received_edges:
            try:
                agg_received_edges = self._received_edges_reducer(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)
            except:
                raise ValueError("reducer should be one of scatter_* [add, mul, max, min, mean]")
            
            nodes_to_collect.append(agg_received_edges)
        
        if self._use_globals:
            expanded_global_attr = global_attr.expand(num_nodes, global_attr.shape[1])
            nodes_to_collect.append(expanded_global_attr)
        
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)

        if self.recurrent:
            graph.x, graph.node_hidden = self.net(collected_nodes, graph.node_hidden)
        else:
            graph.x = self.net(collected_nodes)    # Update
        
        return graph
#         return self.net(collected_nodes)    # (N^v, d_v+d_e+d_e+d_g) -> (N^v, out_features)
        

        
class NodeBlockInd(NodeBlock):
    """Node-level feature transformation.
    Each node is considered independently. (No edge is considered.)
    
    Args:
        in_features: input dimension of node representations.
        out_features: output dimension of node representations.
            (node embedding size)
            
    (N^v, d_v) -> (N^v, out_features)
    NodeBlockInd(graph) -> updated graph
    """
    
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features=32,
                 custom_func=None,
                 recurrent=False):
        
        super(NodeBlockInd, self).__init__(in_features,
                                           out_features,
                                           use_nodes=True,
                                           use_sent_edges=False,
                                           use_received_edges=False,
                                           use_globals=False,
                                           sent_edges_reducer=None,
                                           received_edges_reducer=None,
                                           custom_func=custom_func,
                                           recurrent=recurrent)

        # Customized function
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            self.hidden_features = hidden_features
            if self.recurrent:
                self.net = RecurrentUpdateNet(
                    in_features=in_features,
                    latent_dim=self.hidden_features,
                    out_features=out_features,
                    num_layers=2
                )
            else:
                self.net = nn.Sequential(nn.Linear(in_features, self.hidden_features),
                                         nn.ReLU(),
                                         nn.Linear(self.hidden_features, out_features),
                                        )


class EdgeBlockInd(EdgeBlock):
    """Edge-level feature transformation.
    Each edge is considered independently. (No node is considered.)
    
    Args:
        in_features: input dimension of edge representations.
        out_features: output dimension of edge representations.
            (edge embedding size)
    
    (N^e, d_e) -> (N^e, out_features)
    EdgeBlockInd(graph) -> updated graph
    """
    
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features=32,
                 custom_func=None,
                 recurrent=False):
        
        super(EdgeBlockInd, self).__init__(in_features,
                                           out_features,
                                           use_edges=True,
                                           use_sender_nodes=False,
                                           use_receiver_nodes=False,
                                           use_globals=False,
                                           custom_func=custom_func,
                                           recurrent=recurrent)
        
        # Customized function
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            self.hidden_features = hidden_features
            if self.recurrent:
                self.net = RecurrentUpdateNet(
                    in_features=in_features,
                    latent_dim=self.hidden_features,
                    out_features=out_features,
                    num_layers=2
                )
            else:
                self.net = nn.Sequential(nn.Linear(in_features, self.hidden_features),
                                         nn.ReLU(),
                                         nn.Linear(self.hidden_features, out_features),
                                        )
    

class GlobalBlockInd(GlobalBlock):
    """Global-level feature transformation.
    No edge/node is considered.
    
    Args:
        in_features: input dimension of global representations.
        out_features: output dimension of global representations.
            (global embedding size)
    
    (1, d_g) -> (1, out_features)
    GlobalBlockInd(graph) -> updated graph
    """
    
    def __init__(self,
                 in_features,
                 out_features,
                 hidden_features=32,
                 custom_func=None,
                 recurrent=False):
        
        super(GlobalBlockInd, self).__init__(in_features,
                                             out_features,
                                             use_edges=False,
                                             use_nodes=False,
                                             use_globals=True,
                                             edge_reducer=None,
                                             node_reducer=None,
                                             custom_func=custom_func,
                                             recurrent=recurrent)

        # Customized function
        if custom_func:
            # Customized function can be used for self.net instead of deafult function.
            # It is highly recommended to use nn.Sequential() type.
            self.net = custom_func
        else:
            self.hidden_features = hidden_features
            if self.recurrent:
                self.net = RecurrentUpdateNet(
                    in_features=in_features,
                    latent_dim=self.hidden_features,
                    out_features=out_features,
                    num_layers=2
                )
            else:
                self.net = nn.Sequential(nn.Linear(in_features, self.hidden_features),
                                         nn.ReLU(),
                                         nn.Linear(self.hidden_features, out_features),
                                        )

        
        
        