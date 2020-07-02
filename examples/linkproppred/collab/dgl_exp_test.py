import argparse
import dgl.function as fn
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.utils import expand_as_pair
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.utils.data import DataLoader

from logger import Logger

from igmc_dataset import IGMCDataset, collate_igmc
from igmc_model import IGMC
import scipy.sparse as sp

ckpt_dir = './ckpts/cpu_igmc/'

def infer(model, data_loader):
    model.eval()
    idx = 0

    print (len(data_loader))
    res = []
    for data in data_loader:
        g, label = data[0], data[1]
        # TODO: we haven't handle edge_weight yet.
        out = model(g)
        print (out.mean())
        res += [out.squeeze().cpu()]
        idx += 1
    res = torch.cat(res, dim=0)
    print ('res', len(res))
    return res


@torch.no_grad()
def test(pos_test_pred, neg_test_pred):
    evaluator = Evaluator(name='ogbl-collab')

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (test_hits)

    print (results)
    exit(0)

    return results

def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log-steps', type=int, default=1)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hidden-feats', type=int, default=256)
    parser.add_argument('--hop', type=int, default=2)
    parser.add_argument('--num-workers', type=int, default=6)
    parser.add_argument('--sample-ratio', type=float, default=1.0)
    parser.add_argument('--max-nodes-per-hop', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval-steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    raw_dataset = DglLinkPropPredDataset(name='ogbl-collab')
    edge_split = raw_dataset.get_edge_split()
    graph = raw_dataset[0]

    # Create sparse adj matrix
    num_nodes = graph.number_of_nodes()
    pos_train_edges = edge_split['train']['edge']

    eu, ev, eids = graph.edge_ids(pos_train_edges[:, 0], pos_train_edges[:, 1], return_uv=True)
    train_weights = graph.edata['edge_weight'][eids]

    uni_edges, rev_idx = np.unique(pos_train_edges, axis=0, return_inverse=True)

    sum_edge_weight = np.zeros(uni_edges.shape[0], dtype=np.float32)
    for i in range(rev_idx.shape[0]):
        sum_edge_weight[rev_idx[i]] += train_weights[i]
        if i == 500000:
            print (i)
            
    # Make it undirect
    src = np.concatenate((uni_edges[:, 0], uni_edges[:, 1]))
    dst = np.concatenate((uni_edges[:, 1], uni_edges[:, 0]))

    sum_edge_weight = np.concatenate((sum_edge_weight, sum_edge_weight))
    adj_train = sp.csr_matrix((sum_edge_weight, (src, dst)), shape=(num_nodes, num_nodes))
    uni_undirected_edges = np.vstack((src, dst))

    pos_test_edge = edge_split['test']['edge']
    neg_test_edge = edge_split['test']['edge_neg']
    print ('pos', pos_test_edge.shape, 'neg', neg_test_edge.shape)

    test_pos_dataset = IGMCDataset(
        num_nodes, np.transpose(pos_test_edge), adj_train, 
        args.hop, args.sample_ratio, args.max_nodes_per_hop, max_node_label=args.hop, 
        mode='test')

    test_pos_loader = torch.utils.data.DataLoader(test_pos_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_igmc)

    test_neg_dataset = IGMCDataset(
        num_nodes, np.transpose(neg_test_edge), adj_train, 
        args.hop, args.sample_ratio, args.max_nodes_per_hop, max_node_label=args.hop, 
        mode='test')

    test_neg_loader = torch.utils.data.DataLoader(test_neg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_igmc)

    print (device)
    model = IGMC(args.hop+1)#.to(device)

    ckpt_path = './ckpts/cpu_igmc/ckpt_0_4999.pt'
    model.load_state_dict(torch.load(ckpt_path))

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    test_pos_results = infer(model, test_pos_loader)
    test_neg_results = infer(model, test_neg_loader)

    test(test_pos_results, test_neg_results)


    for key, result in results.items():
        train_hits, valid_hits, test_hits = result
        print(key)
        print(f'Run: {run + 1:02d}, '
                f'Epoch: {epoch:02d}, '
                f'Loss: {loss:.4f}, '
                f'Train: {100 * train_hits:.2f}%, '
                f'Valid: {100 * valid_hits:.2f}%, '
                f'Test: {100 * test_hits:.2f}%')
    print('---')

if __name__ == '__main__':
    #np.random.seed(1234)
    #torch.manual_seed(1234)
    main()
