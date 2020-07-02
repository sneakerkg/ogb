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

def train(epoch, model, train_loader, optimizer):
    model.train()
    total_loss = total_examples = 0
    idx = 0

    print (len(train_loader))

    for data in train_loader:
        optimizer.zero_grad()
        g, label = data[0], data[1]
    

        dd = 0
        device = f'cuda:{dd}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        g.ndata['x'] = g.ndata['x'].to(device)
        g.edata['affine'] = g.edata['affine'].to(device)
        label = label.to(device) 


        # TODO: we haven't handle edge_weight yet.
        out = model(g)
        loss_m = nn.BCELoss()
        loss = loss_m(out, label)
        print (idx, loss)
        idx+=1

        if (idx+1) % 1000 == 0:
            path = ckpt_dir + 'ckpt_' + str(epoch) + '_' + str(idx) + '.pt'
            torch.save(model.state_dict(), path)


        loss.backward()
        optimizer.step()

        num_examples = label.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, x, g, edge_weights, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(g, x, edge_weights)

    pos_train_edge = split_edge['train']['edge'].to(x.device)
    pos_valid_edge = split_edge['valid']['edge'].to(x.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(x.device)
    pos_test_edge = split_edge['test']['edge'].to(x.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(x.device)

    pos_train_preds = []
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size):
        edge = pos_train_edge[perm].t()
        pos_train_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_train_pred = torch.cat(pos_train_preds, dim=0)

    pos_valid_preds = []
    for perm in DataLoader(range(pos_valid_edge.size(0)), batch_size):
        edge = pos_valid_edge[perm].t()
        pos_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_valid_pred = torch.cat(pos_valid_preds, dim=0)

    neg_valid_preds = []
    for perm in DataLoader(range(neg_valid_edge.size(0)), batch_size):
        edge = neg_valid_edge[perm].t()
        neg_valid_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_valid_pred = torch.cat(neg_valid_preds, dim=0)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

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

    train_dataset = IGMCDataset(
        num_nodes, uni_undirected_edges, adj_train, 
        args.hop, args.sample_ratio, args.max_nodes_per_hop, max_node_label=args.hop, 
        mode='train')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_igmc)

    print (device)
    model = IGMC(args.hop+1).to(device)

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    dur = []
    for run in range(args.runs):
        #model.reset_parameters()
        #predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()),
            lr=args.lr)

        for epoch in range(args.epochs):
            t0 = time.time()
            loss = train(epoch, model, train_loader, optimizer)
            if epoch >= 3:
                dur.append(time.time() - t0)
                print('Training time/epoch {}'.format(np.mean(dur)))

            if not args.eval:
                continue
            
            '''
            if epoch % args.eval_steps == 0:
                results = test(model, node_feats, val_loader, test_loader, , node_feats, graph,
                               edge_weights, edge_split, evaluator, args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
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
            '''

        if args.eval:
            for key in loggers.keys():
                print(key)
                loggers[key].print_statistics(run)

    if args.eval:
        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics()

if __name__ == '__main__':
    #np.random.seed(1234)
    #torch.manual_seed(1234)
    main()
