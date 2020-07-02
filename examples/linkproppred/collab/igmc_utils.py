import csv
import time
import random
from tqdm import tqdm
import multiprocessing as mp
from collections import OrderedDict

import numpy as np
import scipy.sparse as sp
import warnings
warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)

class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()

def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])

def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str

def links2subgraphs(
        adj, # label_values, pool,
        train_indices, val_indices, test_indices, 
        train_labels, val_labels, test_labels, 
        hop=1, sample_ratio=1.0, max_nodes_per_hop=200, max_node_label=3, 
        testing=False, parallel=True):

    def helper(adj, links, g_labels):
        g_list = []
        if not parallel: # or max_node_label is None:
            with tqdm(total=len(links[0])) as pbar:
                for u, v, g_label in zip(links[0], links[1], g_labels):
                    g = subgraph_extraction_labeling(
                        g_label, (u, v), adj, 
                        hop, sample_ratio, max_node_label, max_nodes_per_hop)
                    g_list.append(g) 
                    pbar.update(1)
        else:
            start = time.time()
            pool = mp.Pool(mp.cpu_count())
            results = pool.starmap_async(subgraph_extraction_labeling, 
                                        [(g_label, (u, v), adj, 
                                          hop, sample_ratio, max_node_label, max_nodes_per_hop) 
                                          for u, v, g_label in zip(links[0], links[1], g_labels)])
            remaining = results._number_left
            pbar = tqdm(total=remaining)
            while True:
                pbar.update(remaining - results._number_left)
                if results.ready(): break
                remaining = results._number_left
                time.sleep(1)
            g_list += results.get()
            pool.close()
            pool.join()
            pbar.close()
            end = time.time()
            print("Time eplased for subgraph extraction: {}s".format(end-start))
        return g_list

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(adj, train_indices, train_labels)
    val_graphs = helper(adj, val_indices, val_labels) if not testing else []
    test_graphs = helper(adj, test_indices, test_labels)
    return train_graphs, val_graphs, test_graphs

def subgraph_extraction_labeling(g_label, ind, adj, 
                                 hop=1, sample_ratio=1.0, max_node_label=3, max_nodes_per_hop=200):
    # extract the h-hop enclosing subgraph around link 'ind'
    dist = 0

    nodes = [ind[0], ind[1]]
    dists = [0, 0]
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])

    for dist in range(1, hop+1):
        fringe = neighbors(fringe, adj, True)
        fringe = fringe - visited
        visited = visited.union(fringe)
        if sample_ratio < 1.0:
            fringe = random.sample(fringe, int(sample_ratio*len(fringe)))
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = random.sample(fringe, max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes + list(fringe)
        dists = dists + [dist] * len(fringe)
    
    subgraph = adj[nodes, :][:, nodes]
    subgraph[0, 0] = 0

    # reindex u and v, v nodes start after u
    u, v, r = sp.find(subgraph)
    # Here is homo graph, no need
    # v += len(u_nodes)

    # NOTE: adj is symetric, so no need to make it undirectional
    subgraph_info = {'g_label': g_label, 'src': u, 'dst': v, 'r': r}

    # get structural node labels
    # NOTE: only use subgraph here
    node_labels = [x for x in dists]
    x = one_hot(node_labels, max_node_label+1)
    subgraph_info['x'] = x
    
    return subgraph_info

def neighbors(fringe, adj, row=True):
    # TODO [zhoujf] use sample_neighbors fn
    # find all 1-hop neighbors of nodes in fringe from A
    res = set()
    for node in fringe:
        if row:
            _, nei, _ = sp.find(adj[node, :])
        else:
            nei, _, _ = sp.find(adj[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def one_hot(idx, length):
    idx = np.array(idx)
    x = np.zeros([len(idx), length])
    x[np.arange(len(idx)), idx] = 1.0
    return x
