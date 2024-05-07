from torch import nn
import torch.nn.functional as F
import torch
from data_process import ppicell_vec
from collections import defaultdict

class DiseaseModel_mlp(nn.Module):
    def __init__(self, args):
        super(DiseaseModel_mlp, self).__init__()
        self.atom_dim = 34
        self.hid_dim = args.hidden_size
       # self.ffn = nn.Linear(args.hidden_size, args.latent_size)
        self.ffn = nn.Linear(args.hidden_size, self.atom_dim)
        self.ffn2 = nn.Linear(self.atom_dim, args.latent_size)

        self.W_gnn_trans = nn.Linear(self.atom_dim, self.hid_dim)

        self.dropout = 0.3
        self.training = True
        self.do = nn.Dropout(self.dropout)
        self.W_gnn = nn.ModuleList([nn.Linear(self.atom_dim, self.atom_dim),
                                    nn.Linear(self.atom_dim, self.atom_dim),
                                    nn.Linear(self.atom_dim, self.atom_dim)
                                    ])
        self.compound_attn = nn.ParameterList(
            [nn.Parameter(torch.randn(size=(2 * self.atom_dim, 1))) for _ in range(len(self.W_gnn))])

        self.echi_ffn = nn.Linear(args.latent_size + args.cells_size, args.num_tasks)
        self.mlp = Decoder_mlp(args)


        for param in self.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_normal_(param)


    def gnn(self, xs, A):
        for i in range(len(self.W_gnn)):
            h = torch.relu(self.W_gnn[i](xs))
            size = h.size()[0]
            N = h.size()[1]
            a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1,
                                                                                                          2 * self.atom_dim)
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(A > 0, e, zero_vec)  # 保证softmax 不为 0
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h)
            xs = xs + h_prime
        xs = self.do(F.relu(self.W_gnn_trans(xs)))  # (50,200,300)还是节点表示
        return xs

    # NOTE: assume covid targets first, then hiv targets, then random targets
    def VEC_forward(self, echi_batchsize, xs, A):  #
        drug_feature = self.gnn(xs, A)
        drug_feature = self.ffn(drug_feature)
        drug_feature = drug_feature + xs
        drug_feature = self.ffn2(drug_feature)
        drug_feature, _ = torch.max(drug_feature, dim=1)
        cell_embeddings1 = ppicell_vec.main(echi_batchsize).cuda()
        vec = torch.cat([drug_feature, cell_embeddings1], dim=-1)
        vec = torch.sigmoid(vec)
        return vec

    def forward(self, echi_batchsize, xs, A):  # 经过mlp得到药物的治疗效果
        vec = self.VEC_forward(echi_batchsize, xs, A)  # 分子表示
        score = self.mlp(vec)
        return score  # 经过线性层得到最终的分数

    def combo_forward(self, echi_batchsize, xs1, adj1, xs2, adj2):  # 药物组合的协同分数
        vecs1 = self.VEC_forward(echi_batchsize, xs1, adj1)
        vecs2 = self.VEC_forward(echi_batchsize, xs2, adj2)
        combo_vecs = vecs1 + vecs2 - vecs1 * vecs2  # 当作药物效应
        score1 = self.mlp(vecs1)
        score2 = self.mlp(vecs2)
        score = self.mlp(combo_vecs)
        bliss = score1 + score2 - score1*score2
        score = score-bliss
        return score
        # 是在这里分别得到药物1，药物2，药物组合的得分，

class Decoder_mlp(torch.nn.Module):
    def __init__(self, args):
        super(Decoder_mlp, self).__init__()

        self.input_size = args.latent_size + args.cells_size
        # self.input_size = args.latent_size
        self.hidden_dim = args.hidden_size

        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 1)
        self.drop_out = nn.Dropout(0.3)
        self.act = nn.ReLU()


    def forward(self, h):
        h = self.fc1(h)
        h =self.act(h)
        h = self.drop_out(h)
        h = self.fc2(h)
        h =self.act(h)
        h = self.drop_out(h)
        h = self.fc3(h)
        h = self.act(h)
        h = self.drop_out(h)
        h = self.fc4(h)

        return h  # sigmoid归一化，返回相似度得分



