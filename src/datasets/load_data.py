import subprocess
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.utils.convert import to_networkx

from .ordered_data import OrderedData
from utils.data_utils import construct_node_feature, add_skip_connection, add_edge_attr, one_hot
from utils.dag_utils import return_order_info

def parse_pyg_mlpgate(x, edge_index, tt_dis, min_tt_dis, tt_pair_index, y, rc_pair_index, is_rc, has_redundant_fault, \
    use_edge_attr=False, reconv_skip_connection=False, no_node_cop=False, node_reconv=False, un_directed=False, num_gate_types=9, dim_edge_feature=32, logic_implication=False, mask=False):
    # get gate type embedding
    x_torch = construct_node_feature(x, no_node_cop, node_reconv, num_gate_types)
    
    # deepcopy tt_pair_index
    tt_pair_index = torch.tensor(tt_pair_index, dtype=torch.long)
    tt_pair_index = tt_pair_index.t().contiguous()
    
    # deepcopy rc_pair_index
    rc_pair_index = torch.tensor(rc_pair_index, dtype=torch.long)
    rc_pair_index = rc_pair_index.t().contiguous()
    
    # tensorize tt_dis、is_rc and min_tt_dis
    tt_dis = torch.tensor(tt_dis)
    is_rc = torch.tensor(is_rc, dtype=torch.float32).unsqueeze(1)
    min_tt_dis = torch.tensor(min_tt_dis)

    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_attr = add_edge_attr(len(edge_index), dim_edge_feature, 1)

    edge_index = edge_index.t().contiguous()
    
    # dpi dpo
    forward_level, forward_index, backward_level, backward_index = return_order_info(edge_index, x_torch.size(0))
    
    has_redundant_fault = torch.tensor(has_redundant_fault)
    if use_edge_attr:
        raise 'Unsupport edge attr'
    else:
        graph = OrderedData(x=x_torch, edge_index=edge_index, 
                            rc_pair_index=rc_pair_index, is_rc=is_rc,
                            has_redundant_fault=has_redundant_fault,
                            tt_pair_index=tt_pair_index, tt_dis=tt_dis, min_tt_dis=min_tt_dis, 
                            forward_level=forward_level, forward_index=forward_index, 
                            backward_level=backward_level, backward_index=backward_index)
        graph.use_edge_attr = False
    
    graph.gate = torch.tensor(x[:, 1:2], dtype=torch.float)
    graph.prob = torch.tensor(y).reshape((len(x), 1))

    if un_directed:
        graph = ToUndirected()(graph)
    return graph
