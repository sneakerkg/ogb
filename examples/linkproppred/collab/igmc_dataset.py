import random
from collections import namedtuple

import numpy as np
import torch as th
import dgl


from igmc_utils import subgraph_extraction_labeling

IGMCDataTuple = namedtuple('IGMCDataTuple', ['g', 'x', 'r', 'neg_g', 'neg_x', 'neg_r'])

class IGMCDataset(th.utils.data.Dataset):
    def __init__(self, full_g_num_nodes, links, adj, 
                hop=1, sample_ratio=1.0, max_nodes_per_hop=200, max_node_label=1*2+1, 
                mode='train'):
        self.full_g_num_nodes = full_g_num_nodes
        self.links = links
        self.adj = adj
        self.hop = hop
        self.sample_ratio = sample_ratio
        self.max_nodes_per_hop = max_nodes_per_hop
        self.max_node_label = max_node_label

        # self.device = device
        self.mode = mode
    
    def __len__(self):
        return len(self.links[0])

    def __getitem__(self, idx):
        # for u, v, g_label in zip(links[0], links[1], g_labels):
        u, v = self.links[0][idx], self.links[1][idx]
        subgraph = subgraph_extraction_labeling(
            1.0, (u, v), self.adj, 
            self.hop, self.sample_ratio, self.max_node_label, self.max_nodes_per_hop)
        if self.mode == 'train':
            # sample negative edge subgraphs
            neg_edge = np.random.randint(0, self.full_g_num_nodes, 2)
            neg_subgraph = subgraph_extraction_labeling(
                0.0, (neg_edge[0], neg_edge[1]), self.adj, 
                self.hop, self.sample_ratio, self.max_node_label, self.max_nodes_per_hop)
        else:
            neg_subgraph = None
        try:
            return create_dgl_graph(subgraph, neg_subgraph)
        except:
            to_try = np.random.randint(0, len(self.links[0]))
            return self.__getitem__(to_try)

def create_dgl_graph(subgraph, neg_subgraph):
    if subgraph['src'].shape[0] == 0:
        g = dgl.DGLGraph()
        g.add_nodes(2)
    else:
        g = dgl.DGLGraph((subgraph['src'], subgraph['dst']))
    #g = dgl.transform.add_self_loop(g)
    if neg_subgraph is None:
        return IGMCDataTuple(g=g, x=subgraph['x'], r=subgraph['r'], neg_g=None, neg_x=None, neg_r=None)
    else:
        if neg_subgraph['src'].shape[0] == 0:
            neg_g = dgl.DGLGraph()
            neg_g.add_nodes(2)
        else:
            neg_g = dgl.DGLGraph((neg_subgraph['src'], neg_subgraph['dst']))
        # Add self loop
        #neg_g = dgl.transform.add_self_loop(neg_g)
        return IGMCDataTuple(g=g, x=subgraph['x'], r=subgraph['r'], neg_g=neg_g, neg_x=neg_subgraph['x'], neg_r=neg_subgraph['r'])

def collate_igmc(data):
    g_list, x, r, neg_g_list, neg_x, neg_r = map(list, zip(*data))
    
    if neg_g_list[0] is None:
        g = dgl.batch(g_list)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)
        x = th.tensor(np.concatenate(x, axis=0), dtype=th.float)
        r = th.tensor(np.concatenate(r, axis=0), dtype=th.float)
        label = th.tensor(np.ones(len(g_list)), dtype=th.float)
        g.ndata['x'] = x
        g.edata['affine'] = r
    else:
        g = dgl.batch(g_list + neg_g_list)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)
        x = th.tensor(np.concatenate(x, axis=0), dtype=th.float)
        neg_x = th.tensor(np.concatenate(neg_x, axis=0), dtype=th.float)
        g.ndata['x'] = th.cat([x, neg_x], axis=0)
        r = th.tensor(np.concatenate(r, axis=0), dtype=th.float)
        neg_r = th.tensor(np.concatenate(neg_r, axis=0), dtype=th.float)
        g.edata['affine'] = th.cat([r, neg_r], axis=0)
        pos_label = th.tensor(np.ones(len(g_list)), dtype=th.float)
        neg_label = th.tensor(np.zeros(len(neg_g_list)), dtype=th.float)
        label = th.cat((pos_label, neg_label), axis=0)
    return g, label
