import torch

from torch_geometric.nn import knn_graph, knn, CGConv

class GNNAttention(torch.nn.Module):
    '''Uses 2 graph layers. One for self attention and one for cross attention. Self-attention based on k-NN of coordinates. Cross-attention based on k-NN in feature space'''

    def __init__(self, dim, k):
        '''dim is the feature dimensions, k is the number of neighbours to consider'''
        super().__init__()

        self.k = k
        self.conv1 = CGConv(dim, aggr='max', batch_norm=True).cuda()
        self.conv2 = CGConv(dim, aggr='max', batch_norm=True).cuda()

    def forward(self, xyz0, xyz1, f0, f1):
        b, npts, d = f0.shape
        batch_idx = torch.arange(b).repeat_interleave(npts).to(xyz0.device)
        f0 = f0.reshape(-1, d)
        f1 = f1.reshape(-1, d)

        #creates edge graph for coordinates
        edge_idx_c0 = knn_graph(xyz0.reshape(-1,3), k=self.k, batch=batch_idx)
        edge_idx_c1 = knn_graph(xyz1.reshape(-1,3), k=self.k, batch=batch_idx)

        #self-attention (layer 1)
        f0 = self.conv1(f0, edge_idx_c0)
        f1 = self.conv1(f1, edge_idx_c1)

        #cross-attention (layer 2)
        edge_idx_f = knn(f1, f0, k=self.k, batch_x=batch_idx, batch_y=batch_idx, cosine=True)
        edge_idx_f[1] += b * npts
        f = self.conv2(torch.cat([f0,f1], dim=0), edge_idx_f)
        f0, f1 = f[:(b*npts)], f[(b*npts):]

        #convert f0, f1 to dense representation again
        f0 = f0.reshape(b, npts, d)
        f1 = f1.reshape(b, npts, d)

        return f0, f1