"""IGMC modules"""

import math 
import torch as th 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from dgl.nn.pytorch import RelGraphConv, GraphConv

import dgl
import dgl.nn as dgl_nn

from dgl.base import DGLError
import dgl.function as fn


# pylint: disable=W0235
class GraphConv_Custom(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 norm='both',
                 weight=True,
                 bias=True,
                 activation=None):
        super(GraphConv_Custom, self).__init__()
        if norm not in ('none', 'both', 'left', 'right', 'affine'):
            raise DGLError('Invalid norm value. Must be either "none", "both" or "right".'
                           ' But got "{}".'.format(norm))
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._norm = norm

        if weight:
            self.weight = nn.Parameter(th.Tensor(in_feats, out_feats))
        else:
            self.register_parameter('weight', None)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        if self.weight is not None:
            init.xavier_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, graph, feat, weight=None):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: "math:`(\text{in_feats}, \text{out_feats})`.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature
        weight : torch.Tensor, optional
            Optional external weight tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()

        ori_feat = feat

        if self._norm == 'both':
            degs = graph.out_degrees().to(feat.device).float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            feat = feat * norm
        
        if self._norm == 'left':
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            feat = norm * feat

        if weight is not None:
            if self.weight is not None:
                raise DGLError('External weight is provided while at the same time the'
                               ' module has defined its own weight parameter. Please'
                               ' create the module with flag weight=False.')
        else:
            weight = self.weight

        # aggregate first then mult W
        graph.srcdata['h'] = feat
        
        if self._norm == 'affine':
            graph.edata['affine'] = graph.edata['affine'].unsqueeze(-1)
            #print (graph.srcdata['h'].shape, graph.edata['affine'].shape)
            graph.update_all(fn.u_mul_e('h', 'affine', 'm'),
                            fn.sum(msg='m', out='h'))
        else:
            graph.update_all(fn.copy_src(src='h', out='m'),
                            fn.sum(msg='m', out='h'))

        #rst = torch.cat([ori_feat, graph.dstdata['h']], dim=-1)
        rst = graph.dstdata['h']

        if weight is not None:
            rst = th.matmul(rst, weight)

        if self._norm in ['both', 'right']:
            degs = graph.in_degrees().to(feat.device).float().clamp(min=1)
            if self._norm == 'both':
                norm = th.pow(degs, -0.5)
            else:
                norm = 1.0 / degs
            shp = norm.shape + (1,) * (feat.dim() - 1)
            norm = th.reshape(norm, shp)
            rst = rst * norm

        if self.bias is not None:
            rst = rst + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst


    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)



def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class IGMC(nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    
    def __init__(self, in_feats, gconv=GraphConv_Custom, latent_dim=[32, 32, 32, 32], 
                regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, n_side_features=0, 
                multiply_by=1):
        super(IGMC, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by
        self.convs = th.nn.ModuleList()
        self.convs.append(gconv(in_feats, latent_dim[0], norm='affine', bias=True))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1]))
        
        self.lin1 = nn.Linear(2 * sum(latent_dim), 128, bias=True)
        if side_features:
            self.lin1 = nn.Linear(2 * sum(latent_dim) + n_side_features, 128)
        self.lin2 = nn.Linear(128, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            size = conv._in_feats  * conv._out_feats 
            uniform(size, conv.weight)
            uniform(size, conv.bias)
            print (conv.weight.abs().mean(), conv.bias.abs().mean())
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, block):
        # block = edge_drop(block, self.edge_dropout, self.force_undirected, self.training)

        concat_states = []
        x = block.ndata['x']
        for conv in self.convs:
            x = th.tanh(conv(block, x))
            concat_states.append(x)
        concat_states = th.cat(concat_states, 1)        
        query = block.ndata['x'][:, 0] == 1
        query_fea = concat_states[query].reshape([-1, 2, concat_states.shape[-1]])
        x = th.cat([query_fea[:, 0, :], query_fea[:, 1, :]], 1)
        # if self.side_features:
        #     x = th.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            return F.sigmoid(x)

    def __repr__(self):
        return self.__class__.__name__