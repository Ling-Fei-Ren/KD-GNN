import numpy as np
import torch
import time
import os
from train import Train
from args import args
from utils import setup_seed
import pickle

if __name__ == '__main__':
    if not os.path.exists('.checkpoints'):
        os.makedirs('.checkpoints')
    setup_seed(args.seed, torch.cuda.is_available())

    acc_fea = []
    acc_str = []
    acc_stu = []
    repeats = 1
    beta1=[0,0.5,1,1.5,2]
    beta2 = [0, 0.5, 1, 1.5, 2]

    repeat=0
    for data1 in beta1:
        for data2 in beta2:
            args.beta1=data1
            args.beta2 = data2

            train = Train(args,repeat,acc_fea,acc_str,acc_stu)
            t_total = time.time()
            for epoch in range(args.epoch_fea):
                train.pre_train_teacher_fea(epoch)
            train.save_checkpoint(ts='teacher_fea')

            train.load_checkpoint(ts='teacher_fea')
            print('\n--------------\n')

            final_tauc=0
            final_tmf1=0
            for epoch in range(args.epoch_stu):
                final_tauc,final_tmf1=train.train_student(epoch)

            print('args.beta1','args.beta2','auc',)

            print('args.beta1: {:.2f}'.format(args.beta1),
                'args.beta2: {:.2f}'.format(args.beta2),
                'final_tauc: {:.2f}'.format(final_tauc),
                'final_tmf1: {:.2f}'.format(final_tmf1))




