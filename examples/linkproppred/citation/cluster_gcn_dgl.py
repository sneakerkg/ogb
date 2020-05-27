import argparse
from time import time
from functools import partial

from gpuinfo import GPUInfo

import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

import dgl
import dgl.nn as nn
import dgl.function as fn
from dgl_cluster_sampler import ClusterIterDataset, subgraph_collate_fn

from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

from logger import Logger


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(nn.GraphConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(nn.GraphConv(hidden_channels, hidden_channels))
        self.convs.append(nn.GraphConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = conv(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x


class GCNInference(torch.nn.Module):
    def __init__(self, weights):
        super(GCNInference, self).__init__()
        self.weights = weights

    def forward(self, x, adj):
        out = x
        for i, (weight, bias) in enumerate(self.weights):
            out = adj @ out @ weight + bias
            out = np.clip(out, 0, None) if i < len(self.weights) - 1 else out
        return out


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, loader, optimizer, device):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    epoch_st_time = time()

    to_device_time = 0
    ff_time = 0
    pred_loss_time = 0
    bp_time = 0


    for g_data in loader:
        optimizer.zero_grad()

        tmp_time = time()
        feat = g_data.ndata['feat'].to(device)
        to_device_time += (time() - tmp_time)

        tmp_time = time()        
        h = model(g_data, feat)
        ff_time += (time() - tmp_time)

        tmp_time = time()        
        src, dst = g_data.all_edges()
        pos_out = predictor(h[src], h[dst])

        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        dst_neg = torch.randint(0, g_data.ndata['feat'].size(0), src.size(),
                                dtype=torch.long, device=device)
        neg_out = predictor(h[src], h[dst_neg])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        pred_loss_time += (time() - tmp_time)

        tmp_time = time()        
        loss.backward()
        optimizer.step()
        bp_time += (time() - tmp_time)

        num_examples = src.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples
    
    epoch_time = time() - epoch_st_time
    print ('epoch time: ', epoch_time)
    io_time = epoch_time - to_device_time - ff_time - pred_loss_time - bp_time
    memory = [0]
    return total_loss / total_examples, epoch_time, to_device_time, ff_time, pred_loss_time, bp_time, io_time, memory


@torch.no_grad()
def test(model, predictor, data, split_edge, evaluator, batch_size, device):
    model.eval()
    predictor.eval()
    tmp_time = time()
    print('Evaluating full-batch GNN on CPU...')

    model.to(torch.device('cpu'))

    h = model(data, data.ndata['feat']).to(device)

    print('Finish model forward on CPU. Takes:', time() - tmp_time)

    model.to(device)

    def test_split(split):
        source = split_edge[split]['source_node'].to(device)
        target = split_edge[split]['target_node'].to(device)
        target_neg = split_edge[split]['target_node_neg'].to(device)

        pos_preds = []
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst = source[perm], target[perm]
            pos_preds += [predictor(h[src], h[dst]).squeeze().cpu()]
        pos_pred = torch.cat(pos_preds, dim=0)

        neg_preds = []
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = target_neg.view(-1)
        for perm in DataLoader(range(source.size(0)), batch_size):
            src, dst_neg = source[perm], target_neg[perm]
            neg_preds += [predictor(h[src], h[dst_neg]).squeeze().cpu()]
        neg_pred = torch.cat(neg_preds, dim=0).view(-1, 1000)

        return evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })['mrr_list'].mean().item()

    tmp_time = time()
    train_mrr = test_split('eval_train')
    valid_mrr = test_split('valid')
    test_mrr = test_split('test')
    print ('Finish evaluation, takes:', time() - tmp_time)

    return train_mrr, valid_mrr, test_mrr


def main():
    parser = argparse.ArgumentParser(description='OGBL-Citation (Cluster-GCN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_partitions', type=int, default=15000)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_steps', type=int, default=10)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglLinkPropPredDataset(name='ogbl-citation')
    split_edge = dataset.get_edge_split()
    
    # Manually add self-loop link since GCN will wash out the feature of isolated nodes.
    # We should not handle it manually, but in GraphConv module instead.  
    n_nodes = dataset[0].number_of_nodes()
    dataset[0].readonly(False) 
    dataset[0].add_edges(torch.arange(n_nodes).long(), torch.arange(n_nodes).long())
    dataset[0].readonly(True)
    homo_g_data = dgl.to_bidirected(dataset[0])

    g_data = dgl.as_heterograph(homo_g_data)
    for k in dataset[0].node_attr_schemes().keys():
        g_data.ndata[k] = dataset[0].ndata[k]
    
    g_data.in_degree(0)
    g_data.out_degree(0)
    g_data.find_edges(0)

    # Has to use homo graph since metis transform does not take heterograph
    cluster_dataset = ClusterIterDataset('ogbl-citation', homo_g_data, args.num_partitions, use_pp=False)
    cluster_iterator = DataLoader(cluster_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers , collate_fn=partial(subgraph_collate_fn, g_data))

    # We randomly pick some training samples that we want to evaluate on:
    torch.manual_seed(12345)
    idx = torch.randperm(split_edge['train']['source_node'].numel())[:86596]
    split_edge['eval_train'] = {
        'source_node': split_edge['train']['source_node'][idx],
        'target_node': split_edge['train']['target_node'][idx],
        'target_node_neg': split_edge['valid']['target_node_neg'],
    }

    model = GCN(g_data.ndata['feat'].size(-1), args.hidden_channels, args.hidden_channels,
                args.num_layers, args.dropout).to(device)
    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1,
                              args.num_layers, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-citation')
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        epoch_time, to_device_time, ff_time, pred_loss_time, bp_time, io_time, memory = 0, 0, 0, 0, 0, 0, 0
        for epoch in range(1, 1 + args.epochs):
            loss, c_epoch_time, c_to_device_time, c_ff_time, c_pred_loss_time, c_bp_time, c_io_time, c_memory = train(model, predictor, cluster_iterator, optimizer, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}')
            epoch_time += c_epoch_time
            to_device_time += c_to_device_time
            ff_time += c_ff_time
            pred_loss_time += c_pred_loss_time
            bp_time += c_bp_time
            io_time += c_io_time
            memory = max(memory, c_memory[0])

            if epoch % args.eval_steps == 0:
                print ('Ave')
                print ('epoch time: ', epoch_time / args.eval_steps)
                print ('to_device_time: ', to_device_time/ args.eval_steps)
                print ('ff_time: ', ff_time / args.eval_steps)
                print ('pred_loss_time: ', pred_loss_time / args.eval_steps)
                print ('bp_time: ', bp_time / args.eval_steps)
                print ('io_time: ', io_time / args.eval_steps)
                print ('max memory', memory)
                print ('\n')
                epoch_time, to_device_time, ff_time, pred_loss_time, bp_time, io_time, memory = 0, 0, 0, 0, 0, 0, 0

                result = test(model, predictor, g_data, split_edge, evaluator,
                              64 * 4 * args.batch_size, device)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_mrr, valid_mrr, test_mrr = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {train_mrr:.4f}, '
                          f'Valid: {valid_mrr:.4f}, '
                          f'Test: {test_mrr:.4f}')

        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
