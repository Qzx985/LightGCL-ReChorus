import torch
import torch.nn as nn
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F
from models.BaseModel import GeneralModel

class LightGCL(GeneralModel):
    reader = 'BaseReader'
    runner = 'BaseRunner'

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
        parser.add_argument('--lambda1', default=0.05, type=float, help='weight of cl loss')
        parser.add_argument('--d', default=64, type=int, help='embedding size')
        parser.add_argument('--q', default=5, type=int, help='rank')
        parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')
        parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
        parser.add_argument('--lambda2', default=1e-5, type=float, help='l2 reg weight')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        super().__init__(args, corpus)
        self.emb_size = args.d
        self.n_layers = args.gnn_layer
        self.q = args.q
        self.temp = args.temp
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(corpus.n_users, self.emb_size)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(corpus.n_items, self.emb_size)))
        
        self.adj_norm = self.build_adjmat(corpus.n_users, corpus.n_items, corpus.train_clicked_set)
        self.sparse_adj_norm = self.convert_sparse_matrix_to_tensor(self.adj_norm).to(torch.device(self.device))

    def build_adjmat(self, user_count, item_count, train_mat):
        R = sp.dok_matrix((user_count, item_count), dtype=np.float32)
        for user in train_mat:
            for item in train_mat[user]:
                R[user, item] = 1.0
        R = R.tocoo()

        row_sum = np.array(R.sum(1)).flatten() + 1e-10
        col_sum = np.array(R.sum(0)).flatten() + 1e-10

        R_coo = R.tocoo()
        data = R_coo.data
        row = R_coo.row
        col = R_coo.col

        for i in range(len(data)):
            data[i] = data[i] / pow(row_sum[row[i]] * col_sum[col[i]], 0.5)

        adj_norm = sp.csr_matrix((data, (row, col)), shape=(user_count, item_count))
        return adj_norm

    def convert_sparse_matrix_to_tensor(self, X):
        coo = X.tocoo()
        indices = torch.LongTensor(np.array([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        return torch.sparse_coo_tensor(indices, values, coo.shape)

    def compute_svd(self, norm_adj, q):
        svd_u, s, svd_v = torch.svd_lowrank(norm_adj, q=q)
        u_s = svd_u @ torch.diag(s)
        v_s = svd_v @ torch.diag(s)
        del s
        return u_s, v_s, svd_u.T, svd_v.T

    def forward(self, feed_dict):
        uids, iids = feed_dict['user_id'], feed_dict['item_id']
        u_s, v_s, ut, vt = self.compute_svd(self.sparse_adj_norm, q=self.q)
        
        E_u_list = [self.E_u_0]
        E_i_list = [self.E_i_0]
        G_u_list, G_i_list = [], []

        for layer in range(1, self.n_layers+1):
            Z_u = torch.spmm(self.sparse_adj_norm, E_i_list[layer-1])
            Z_i = torch.spmm(self.sparse_adj_norm.T, E_u_list[layer-1])

            G_u_list.append(u_s @ (vt @ E_i_list[layer-1]))
            G_i_list.append(v_s @ (ut @ E_u_list[layer-1]))

            E_u_list.append(Z_u)
            E_i_list.append(Z_i)

        G_u = sum(G_u_list)
        G_i = sum(G_i_list)
        E_u = sum(E_u_list)
        E_i = sum(E_i_list)

        u_e = E_u[uids, :]
        i_e = E_i[iids, :]
        g_u_e = G_u[uids, :]
        g_i_e = G_i[iids, :]
        prediction = (u_e[:, None, :] * i_e).sum(dim=-1)

        return {
            'prediction': prediction,
            'E_u_norm': E_u,
            'E_i_norm': E_i,
            'G_u_e': g_u_e,
            'G_i_e': g_i_e.view(-1, self.emb_size),
            'E_u_e': u_e,
            'E_i_e': i_e.view(-1, self.emb_size),
        }

    def loss(self, out_dict):
        # bpr loss
        prediction = out_dict['prediction']
        pos_pred, neg_pred = prediction[:, 0], prediction[:, 1:]
        neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
        loss_bpr = -(((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1)).clamp(min=1e-8, max=1-1e-8).log().mean()
        
        # cl loss
        E_u_norm, E_i_norm, G_u_e, G_i_e, E_u_e, E_i_e = out_dict['E_u_norm'], out_dict['E_i_norm'], out_dict['G_u_e'], out_dict['G_i_e'], out_dict['E_u_e'], out_dict['E_i_e']
        neg_score = torch.log(torch.exp(G_u_e @ E_u_norm.T / self.temp).sum(1) + 1e-8).mean()
        neg_score += torch.log(torch.exp(G_i_e @ E_i_norm.T / self.temp).sum(1) + 1e-8).mean()
        pos_score = (torch.clamp((G_u_e * E_u_e).sum(1) / self.temp,-5.0,5.0)).mean() + (torch.clamp((G_i_e * E_i_e).sum(1) / self.temp,-5.0,5.0)).mean()
        loss_cl = -pos_score + neg_score

        loss_reg = 0
        for param in self.parameters():
            loss_reg += param.norm(2).square()

        # total loss
        return loss_bpr + self.lambda1 * loss_cl + self.lambda2 * loss_reg