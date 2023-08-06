import torch
import torch.nn as nn
import torch.nn.functional as F
# from layers import GraphConvolution
from dgl import function as fn
# from dgl.nn import GraphConv as GraphConvolution


class Layer(nn.Module):
    def __init__(self, g, in_dim, dropout_rate):
        super(Layer, self).__init__()
        self.g = g
        self.dropout = nn.Dropout(dropout_rate)
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def edge_applying(self, edges):
        h2 = torch.cat([edges.dst['h'], edges.src['h']], dim=1)
        a = torch.tanh(self.gate(h2)).squeeze()
        ee = edges.dst['d'] * edges.src['d']
        # ee=ee.unsqueeze(1)
        e=ee*a
        e = self.dropout(e)
        return {'e': e, 'm': a}

    #h表示为特征向量——feature
    def forward(self, h):
        self.g.ndata['h'] = h
        self.g.apply_edges(self.edge_applying)
        self.g.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'z'))

        return self.g.ndata['z']


#Feature Teacher
class Teacher_F(nn.Module):
    def __init__(self, num_nodes, in_size, hidden_size, out_size, num_layers, dropout):
        super(Teacher_F, self).__init__()
        if num_layers == 1:
            hidden_size = out_size
        # self.fun = nn.LeakyReLU(0.3)
        self.tau=1
        self.fun=nn.ReLU()
        self.imp_feat = nn.Parameter(torch.empty(size=(num_nodes, in_size)))
        nn.init.xavier_normal_(self.imp_feat.data, gain=1.414)
        self.dropout = nn.Dropout(dropout)  # dropout函数

        self.fm1 = nn.Linear(in_size, hidden_size, bias=True)
        self.fm2 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fm3 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fm4 = nn.Linear(hidden_size, out_size, bias=True)
        nn.init.xavier_normal_(self.fm1.weight, gain=1.414)
        nn.init.xavier_normal_(self.fm2.weight, gain=1.414)
        nn.init.xavier_normal_(self.fm3.weight, gain=1.414)


    def forward(self, feature):
        feature = torch.where(torch.isnan(feature), self.imp_feat, feature)
        middle_representations = []
        h1=feature
        h = self.dropout(feature)
        h = self.fm1(h)
        middle_representations.append(h)
        h=self.fun(h)
        h = self.fm2(h)
        # middle_representations.append(h)
        h=self.fun(h)
        h = self.fm3(h)
        middle_representations.append(h)
        h=self.fun(h)
        h = self.fm4(h)

        return h, h1,middle_representations

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z2))


        return torch.log(refl_sim.diag())

    def loss(self,imp_feat,edge_index,edge_label):
        R_stu_1 = imp_feat[edge_index[0]]
        R_fea_1 = imp_feat[edge_index[1]]
        fea_stu_1 = self.semi_loss(R_stu_1, R_fea_1)
        mean_edge=torch.mul(fea_stu_1.unsqueeze(0),edge_label.unsqueeze(0)).mean()
        return mean_edge


#Student
class GCN(nn.Module):
    def __init__(self, graph1,graph2, graph3, nfeat, nhid, nclass, dropout, nhid_feat, eps,tau=0.5):
        super(GCN, self).__init__()

        self.g1 = graph1
        self.g2 = graph2
        self.g3 = graph3
        # self.fun = nn.ReLU()
        self.fun=nn.ReLU()
        self.eps=eps
        self.dropout = nn.Dropout(dropout)  # dropout函数
        self.tau=tau
       
        # For layer1
        self.layer1_1 = Layer(self.g1, nhid, dropout)
        self.layer1_2 = Layer(self.g2, nhid, dropout)
        self.layer1_3 = Layer(self.g3, nhid, dropout)
        self.hw1_1 = nn.Parameter(torch.FloatTensor(nhid, nhid))
        self.hw1_2 = nn.Parameter(torch.FloatTensor(nhid, nhid))
        self.hw1_3 = nn.Parameter(torch.FloatTensor(nhid, nhid))

       


        nn.init.xavier_normal_(self.hw1_1, gain=1.414)
        nn.init.xavier_normal_(self.hw1_2, gain=1.414)
        nn.init.xavier_normal_(self.hw1_3, gain=1.414)

       

        # For layer2
        self.layer2_1 = Layer(self.g1, nhid*3 , dropout)
        self.layer2_2 = Layer(self.g2, nhid*3 , dropout)
        self.layer2_3 = Layer(self.g3, nhid *3, dropout)

        self.hw2_1 = nn.Parameter(torch.FloatTensor(nhid*3, nhid))
        self.hw2_2 = nn.Parameter(torch.FloatTensor(nhid*3, nhid))
        self.hw2_3 = nn.Parameter(torch.FloatTensor(nhid *3, nhid))
        nn.init.xavier_normal_(self.hw2_1, gain=1.414)
        nn.init.xavier_normal_(self.hw2_2, gain=1.414)
        nn.init.xavier_normal_(self.hw2_3, gain=1.414)

        self.t1 = nn.Linear(nfeat, nhid)
        # self.t2 = nn.Linear(nhid*3, nhid)
        self.t2 = nn.Linear(7 * nhid +nfeat, nhid)
        self.t3 = nn.Linear(nhid, nclass)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)
        nn.init.xavier_normal_(self.t3.weight, gain=1.414)
        # nn.init.xavier_normal_(self.t4.weight, gain=1.414)

    def forward(self, x):
        #imp[0]
        middle_representations=[]
        imp = torch.zeros([x.shape[0], x.shape[1]]).to(x.device)
        x = torch.where(torch.isnan(x), imp, x)
        h = x
        raw0 = h
        h = self.dropout(h)
        h = self.t1(h)
        middle_representations.append(h)
        h=self.fun(h)
        raw1 = h

        # for the layer1
        h1_1 = self.fun(torch.mm(self.eps * raw1 + self.layer1_1(h), self.hw1_1))
        h1_2 = self.fun(torch.mm(self.eps * raw1 + self.layer1_2(h), self.hw1_2))
        h1_3 = self.fun(torch.mm(self.eps * raw1 + self.layer1_3(h), self.hw1_3))

        # aggregation
        h =torch.cat((h1_1, h1_2, h1_3), dim=1)
        raw2 = h

        # for the layer2
        h2_1 = self.fun(torch.mm(self.eps * raw2 + self.layer2_1(h), self.hw2_1))
        h2_2 = self.fun(torch.mm(self.eps * raw2 + self.layer2_2(h), self.hw2_2))
        h2_3 = self.fun(torch.mm(self.eps * raw2 + self.layer2_3(h), self.hw2_3))

        h = torch.cat((h2_1, h2_2, h2_3,raw1,raw2,raw0), dim=1)

        node_feature = h
        scores_model = self.t2(node_feature)
        middle_representations.append(scores_model)
        scores_model=self.fun(scores_model)
        scores_model = self.t3(scores_model)

        return scores_model, middle_representations

    #contrast loss
    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,train_idx,
             mean: bool = True):
        R_stu_1 = z1[0][train_idx]
        R_fea_1 = z2[0][train_idx]
        fea_stu_1 = self.semi_loss(R_stu_1, R_fea_1)
        fea_stu_1 = fea_stu_1.mean() if mean else fea_stu_1.sum()
        R_stu_2 = z1[1][train_idx]
        R_fea_2 = z2[1][train_idx]
        fea_stu_2 = self.semi_loss(R_stu_2, R_fea_2)
        fea_stu_2 = fea_stu_2.mean() if mean else fea_stu_2.sum()



        loss_mid_fea = fea_stu_1 + fea_stu_2

        return loss_mid_fea

