from __future__ import division
from __future__ import print_function
import torch.nn.functional as F
import time
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from utils import load_data, accuracy,get_best_f1,edge_sample
from models import GCN, Teacher_F
from args import args
from logit_losses import *
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix
# from ppr_matrix import topk_ppr_matrix

# Model and optimizer
class Train:
    def __init__(self, args,repeat,acc_fea,acc_str,acc_stu):
        self.args = args
        self.repeat = repeat
        self.best_teacher_fea_val, self.best_teacher_str_val, self.best_student_val = 0, 0, 0
        self.teacher_fea_state,  self.teacher_str_state, self.student_state = None, None, None
        self.load_data()
        self.acc_list_fea = acc_fea
        self.acc_list_str = acc_str
        self.acc_list = acc_stu
        self.best_f1, self.final_tf1, self.final_trec, self.final_tpre, self.final_tmf1, self.final_tauc = 0., 0., 0., 0., 0., 0.

        # Model Initialization
        self.fea_model = Teacher_F(
                                 num_nodes=self.features.shape[0],
                                 in_size=self.features.shape[1],
                                 hidden_size=self.args.hidden_fea,
                                 out_size=self.label_oneHot.shape[1],
                                 num_layers=self.args.num_fea_layers,
                                 dropout=self.args.dropout_fea

        )
        self.fea_model.to(args.device)

        self.stu_model = GCN(graph1=self.g1,
                             graph2=self.g2,
                             graph3=self.g3,
                           nfeat=self.features.shape[1],
                           nhid=self.args.hidden_stu,
                           nclass=self.label_oneHot.shape[1],
                           dropout=self.args.dropout_stu,
                           nhid_feat=self.args.hidden_fea,
                            eps=args.eps)

        self.stu_model.to(args.device)

        self.criterionStudentKD = SoftTarget(args.Ts)

        # Setup Training Optimizer
        self.optimizerTeacherFea = optim.Adam(self.fea_model.parameters(), lr=self.args.lr_fea, weight_decay=self.args.weight_decay_fea)
        self.optimizerStudent = optim.Adam(self.stu_model.parameters(), lr=self.args.lr_stu, weight_decay=self.args.weight_decay_stu)

    def load_data(self):
        # load data
        self.g1, self.g2, self.g3, self.hom,self.features, self.labels, self.label_oneHot, self.idx_train, self.idx_valid, self.idx_test = load_data(args.dataset, self.repeat,
                                                                                       self.args.device, self.args.rate,self.args.train_rate,self.args.test_rate)

        g_list = [self.g1, self.g2, self.g3]
        name = 'self_loop'
        if name == 'self_loop':
            for g in g_list:
                deg = g.in_degrees().float().to(self.args.device)
                deg = deg + torch.ones(len(deg)).to(self.args.device)
                norm = torch.pow(deg, -0.5)
                g.ndata['d'] = norm
        else:
            for g in g_list:
                deg = g.in_degrees().float().clamp(min=1).to(self.args.device)
                norm = torch.pow(deg, -0.5)
                g.ndata['d'] = norm

        self.weight = (1 - self.labels[ self.idx_train]).sum().item() / self.labels[ self.idx_train].sum().item()
        self.edge_index, self.edge_label = edge_sample(self.hom, self.idx_train, self.labels,self.args.device)

        print('Data load init finish')


    def pre_train_teacher_fea(self,epoch):
        t = time.time()
        self.fea_model.train()
        t = time.time()
        self.optimizerTeacherFea.zero_grad()
        index_sample = np.random.choice([i for i in range(self.edge_index.shape[1])],int(self.edge_index.shape[1]*0.01))
        self.edge_index_sample = self.edge_index[:, index_sample]
        self.edge_label_sample  = self.edge_label[index_sample]
        output,imp_feat,_ = self.fea_model(self.features)
        loss_train1 = F.cross_entropy(output[self.idx_train], self.labels[self.idx_train],weight=torch.tensor([1., self.weight]).to(self.args.device))
        loss_train2=self.fea_model.loss(imp_feat,self.edge_index_sample, self.edge_label_sample)
        loss_train=loss_train1+0.1*loss_train2
        acc_train = accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizerTeacherFea.step()
        if not self.args.fastmode:
            self.fea_model.eval()
            output, imp_feat,_ = self.fea_model(self.features)

        loss_val = F.cross_entropy(output[self.idx_valid], self.labels[self.idx_valid])
        acc_val = accuracy(output[self.idx_valid], self.labels[self.idx_valid])

        if acc_val > self.best_teacher_fea_val:
            self.best_teacher_fea_val = acc_val
            self.teacher_fea_state = {
                'state_dict': self.fea_model.state_dict(),
                'best_val': acc_val,
                'best_epoch': epoch+1,
                'optimizer': self.optimizerTeacherFea.state_dict(),
            }
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

    def train_student(self, epoch):
        t = time.time()
        self.stu_model.train()
        self.optimizerStudent.zero_grad()
        output, middle_emb_stu = self.stu_model(self.features)
        soft_target_fea, imp_feat,middle_emb_fea = self.fea_model(self.features)
        contrast_fea = self.stu_model.loss(middle_emb_stu, middle_emb_fea,self.idx_train)
        loss_train = F.cross_entropy(output[self.idx_train], self.labels[self.idx_train],weight=torch.tensor([1., self.weight]).to(self.args.device)) +self.args.beta1*self.criterionStudentKD(output[self.idx_train], soft_target_fea[self.idx_train] +self.args.beta2*contrast_fea)
        loss_train.backward()
        self.optimizerStudent.step()
        if not self.args.fastmode:
            self.stu_model.eval()
            output, _ = self.stu_model(self.features)
        probs = output.softmax(1)
        f1, thres = get_best_f1(self.labels[self.idx_valid], probs[self.idx_valid])
        preds = np.zeros_like(self.labels.cpu().numpy())
        preds[probs[:, 1].detach().cpu().numpy() > thres] = 1
        trec = recall_score(self.labels[self.idx_test].cpu().numpy(), preds[self.idx_test])
        tpre = precision_score(self.labels[self.idx_test].cpu().numpy(), preds[self.idx_test])
        tmf1 = f1_score(self.labels[self.idx_test].cpu().numpy(), preds[self.idx_test], average='macro')
        tauc = roc_auc_score(self.labels[self.idx_test].cpu().numpy(), probs[self.idx_test][:, 1].detach().cpu().numpy())
        

        if self.best_f1 < f1:
            self.best_f1 = f1
            self.final_trec = trec
            self.final_tpre = tpre
            self.final_tmf1 = tmf1
            self.final_tauc = tauc
        
        # print('Test: REC {:.2f} PRE {:.2f} MF1 {:.2f} AUC {:.2f}'.format(self.final_trec * 100,
        #                                                              self.final_tpre * 100, self.final_tmf1 * 100,
        #                                                              self.final_tauc * 100))


        # loss_val = F.cross_entropy(output[self.idx_valid], self.labels[self.idx_valid])
        # acc_val = accuracy(output[self.idx_valid], self.labels[self.idx_valid])
        return self.final_tauc*100,self.final_tmf1*100

        
        
    def test(self, ts='teacher_fea'):
        if ts == 'teacher_fea':
            model = self.fea_model
            model.eval()
            output, _ = model(self.features)
            loss_test = F.cross_entropy(output[self.idx_test], self.labels[self.idx_test])
            acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
            print("{ts} Test set results:".format(ts=ts),
                  "loss= {:.4f}".format(loss_test.item()),
                  "accuracy= {:.4f}".format(acc_test.item()))
            self.acc_list_fea.append(round(acc_test.item(), 4))
        elif ts == 'student':
            model = self.stu_model
            model.eval()
            output, _ = model(self.features)
            loss_test = F.cross_entropy(output[self.idx_test], self.labels[self.idx_test])
            # auc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
            output=torch.sigmoid(output)

            auc_test=roc_auc_score(self.labels[self.idx_test].cpu().numpy(), output[self.idx_test].detach().cpu().numpy()[:,1])
            tmf1 = f1_score(self.labels[self.idx_test].cpu().numpy(),output[self.idx_test].data.cpu().numpy().argmax(axis=1),average='macro')
            trec=recall_score(self.labels[self.idx_test].cpu().numpy(),output[self.idx_test].data.cpu().numpy().argmax(axis=1))


            print("{ts} Test set results:".format(ts=ts),
                  "loss= {:.4f}".format(loss_test.item()),
                  "auc_test= {:.4f}".format(auc_test.item()),
                  "tmf1= {:.4f}".format(tmf1.item()),
                  "trec= {:.4f}".format(trec.item()))
            self.acc_list.append(round(auc_test.item(), 4))

    def save_checkpoint(self, filename='./.checkpoints/'+args.dataset, ts='teacher_fea'):
        print('Save {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'teacher_fea':
            torch.save(self.teacher_fea_state, filename)
            print('Successfully saved feature teacher model\n...')
        elif ts == 'teacher_str':
            torch.save(self.teacher_str_state, filename)
            print('Successfully saved structure teacher model\n...')
        elif ts == 'student':
            torch.save(self.student_state, filename)
            print('Successfully saved student model\n...')
        
        
    def load_checkpoint(self, filename='./.checkpoints/'+ args.dataset, ts='teacher_fea'):
        print('Load {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'teacher_fea':
            load_state = torch.load(filename)
            self.fea_model.load_state_dict(load_state['state_dict'])
            self.optimizerTeacherFea.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded feature teacher model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'].item())
        elif ts == 'student':
            load_state = torch.load(filename)
            self.stu_model.load_state_dict(load_state['state_dict'])
            self.optimizerStudent.load_state_dict(load_state['optimizer'])
            print('Successfully Loaded student model\n...')
            print("Best Epoch:", load_state['best_epoch'])
            print("Best acc_val:", load_state['best_val'].item())

