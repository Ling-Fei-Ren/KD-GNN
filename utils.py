import os
import random
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve,f1_score
import torch
import dgl
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
# from tensorflow.keras.utils import to_categorical
import pickle
from dgl.data import FraudAmazonDataset
from dgl.data import FraudYelpDataset
from scipy.io import loadmat
import copy
from copy import deepcopy

def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels.cpu().numpy())
        preds[probs[:,1].detach().cpu().numpy() > thres] = 1
        mf1 = f1_score(labels.cpu().numpy(), preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

def setup_seed(seed, cuda):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if cuda is True:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def normalize_row(mx):
	"""Row-normalize sparse matrix"""
	rowsum = np.array(mx.sum(1))
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	r_mat_inv = sp.diags(r_inv)
	mx = r_mat_inv.dot(mx)
	return mx

def normalize_sparse(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



def edge_sample(hom, idx_train, y_train,device):

    row=hom.all_edges()[0]
    col = hom.all_edges()[1]
    edge_index=np.isin(row, idx_train) & np.isin(col, idx_train)

    row1=row[edge_index]
    col1=col[edge_index]

    edge_index=np.array((np.array(row1),np.array(col1)))
    edge_label=1-(y_train[row1]^y_train[col1])


    edge_label = torch.where(edge_label==0,torch.FloatTensor([-1]).to(device),torch.FloatTensor([1]).to(device))


    return edge_index,edge_label

def accuracy(output, labels):
    preds = F.softmax(output, dim=1)[:, 1]
    y_test = labels.cpu().numpy()
    y_pred = preds.detach().cpu().numpy()
    auc_roc=roc_auc_score(y_test, y_pred)

    return auc_roc

def feature_noise(features, missing_rate):
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask

def apply_feature_noise(features, mask):
    noise_matrix = torch.FloatTensor(np.random.normal(0, 0.1, size=[features.shape[0], features.shape[1]])).to(features.device)
    features = torch.where(mask==True,features+noise_matrix,features)
    return features


def feature_mask(features, missing_rate):
    mask = torch.rand(size=features.size())
    mask = mask <= missing_rate
    return mask
def apply_feature_mask(features, mask):
    features[mask] = float('nan')



def load_data(dataset, repeat, device, rate,train_rate,test_rate):
    prefix='data/'
    if dataset=='yelp':
        data = FraudYelpDataset()
        graph = data[0]
        number = graph.num_nodes()
        features = graph.ndata['feature']
        labels = graph.ndata['label']
        g1 = dgl.graph(graph['net_rur'].edges(), num_nodes=number)
        g2 = dgl.graph(graph['net_rtr'].edges(), num_nodes=number)
        g3 = dgl.graph(graph['net_rsr'].edges(), num_nodes=number)
        hom = deepcopy(g1)
        hom.add_edges(g2.all_edges()[0], g2.all_edges()[1])
        hom.add_edges(g3.all_edges()[0], g3.all_edges()[1])
        hom = dgl.to_simple(hom)
        hom = dgl.remove_self_loop(hom)
        g1=g1.to(device)
        g2 = g2.to(device)
        g3 = g3.to(device)


    elif dataset=='amazon':
        data = FraudAmazonDataset()
        graph = data[0]
        number = graph.num_nodes()
        features = graph.ndata['feature']
        labels = graph.ndata['label']
        g1 = dgl.graph(graph['net_upu'].edges(), num_nodes=number)
        g2 = dgl.graph(graph['net_usu'].edges(), num_nodes=number)
        g3 = dgl.graph(graph['net_uvu'].edges(), num_nodes=number)
        hom = deepcopy(g1)
        hom.add_edges(g2.all_edges()[0], g2.all_edges()[1])
        hom.add_edges(g3.all_edges()[0], g3.all_edges()[1])
        hom = dgl.to_simple(hom)
        hom = dgl.remove_self_loop(hom)
        g1 = g1.to(device)
        g2 = g2.to(device)
        g3 = g3.to(device)


    elif dataset=='Tele-max':
        yelp = loadmat(prefix +'Tele-max/' +'tele_max.mat')
        labels =  torch.LongTensor(yelp['label'].flatten())
        features =yelp['features']
        tele_ucu = yelp['tele_ucu']
        tele_usu = yelp['tele_usu']
        tele_utu = yelp['tele_utu']
        g1 = dgl.graph((tele_ucu[0],tele_ucu[1]))
        g2 = dgl.graph((tele_usu[0],tele_usu[1]))
        g3 = dgl.graph((tele_utu[0],tele_utu[1]))
        hom=deepcopy(g1)
        hom.add_edges(g2.all_edges()[0],g2.all_edges()[1])
        hom.add_edges(g3.all_edges()[0], g3.all_edges()[1])
        hom=dgl.to_simple(hom)
        hom=dgl.remove_self_loop(hom)
        g1 = g1.to(device)
        g2 = g2.to(device)
        g3 = g3.to(device)


    elif dataset=='Tele-mini':
        yelp = loadmat(prefix +'Tele-min/' +'tele_mini.mat')
        labels = torch.LongTensor(yelp['label'].flatten())
        features = yelp['features']
        tele_ucu = yelp['tele_ucu']
        tele_usu = yelp['tele_usu']
        tele_utu = yelp['tele_utu']
        g1 = dgl.graph((tele_ucu[0], tele_ucu[1]))
        g2 = dgl.graph((tele_usu[0], tele_usu[1]))
        g3 = dgl.graph((tele_utu[0], tele_utu[1]))
        hom = deepcopy(g1)
        hom.add_edges(g2.all_edges()[0], g2.all_edges()[1])
        hom.add_edges(g3.all_edges()[0], g3.all_edges()[1])
        hom = dgl.to_simple(hom)
        hom = dgl.remove_self_loop(hom)
        g1 = g1.to(device)
        g2 = g2.to(device)
        g3 = g3.to(device)



    # features = graph.ndata['feature']
    features = torch.FloatTensor(normalize_row(features)).to(device)
    mask = feature_mask(features, rate)
    apply_feature_mask(features, mask.to(device))

    #
    label_oneHot = torch.zeros((len(labels), 2)).scatter_(1, labels.long().reshape(-1, 1), 1).to(device)

    if dataset=='yelp' or dataset=='Tele-max'or dataset=='Tele-mini':
        index = list(range(len(labels)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                train_size=train_rate,
                                                                random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=test_rate,
                                                                random_state=2, shuffle=True)
    elif dataset == 'amazon':
        index = list(range(3305, len(labels)))
        idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                                train_size=train_rate,
                                                                random_state=2, shuffle=True)
        idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                                test_size=test_rate,
                                                                random_state=2, shuffle=True)


    return g1,g2,g3,hom,features, labels.to(device), label_oneHot, idx_train, idx_valid, idx_test




    # sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)









# def load_data(dataset, repeat, device, rate):
#     path = './data/{}/'.format(dataset)
#
#     f = np.loadtxt(path + '{}.feature'.format(dataset), dtype=float)
#     l = np.loadtxt(path + '{}.label'.format(dataset), dtype=int)
#     test = np.loadtxt(path + '{}test.txt'.format(repeat), dtype=int)
#     train = np.loadtxt(path + '{}train.txt'.format(repeat), dtype=int)
#     val = np.loadtxt(path + '{}val.txt'.format(repeat), dtype=int)
#     features = sp.csr_matrix(f, dtype=np.float32)
#     features = torch.FloatTensor(np.array(features.todense())).to(device)
#     mask = feature_mask(features, rate)
#     apply_feature_mask(features, mask)
#
#     idx_test = test.tolist()
#     idx_train = train.tolist()
#     idx_val = val.tolist()
#
#     idx_train = torch.LongTensor(idx_train).to(device)
#     idx_test = torch.LongTensor(idx_test).to(device)
#     idx_val = torch.LongTensor(idx_val).to(device)
#     label = torch.LongTensor(np.array(l)).to(device)
#
#     label_oneHot = torch.FloatTensor(to_categorical(l)).to(device)
#
#     struct_edges = np.genfromtxt(path + '{}.edge'.format(dataset), dtype=np.int32)
#     sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
#     sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])),
#                          shape=(features.shape[0], features.shape[0]), dtype=np.float32)
#     sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
#     sadj = edge_delete(rate, sadj)
#
#     # ppr_input:A+I
#     ttadj = sadj + sp.eye(sadj.shape[0])
#     ttadj = torch.FloatTensor(ttadj.todense()).to(device)
#     # A
#     tadj = torch.FloatTensor(sadj.todense()).to(device)
#     # stu_input
#     sadj = normalize_sparse(sadj + sp.eye(sadj.shape[0]))
#     nsadj = torch.FloatTensor(np.array(sadj.todense())).to(device)
#
#
#     return ttadj, tadj, nsadj, features, label, label_oneHot, idx_train, idx_val, idx_test
