import os.path
import pathlib
import sys
import torch
import torch.nn as nn
import math
from datetime import datetime



class Config(object):
    def __init__(self, args):
        print('Building Config')


        # Data path
        root = '../Dataset/'
        self.dataset = args.dataset
        print(f'dataset: {self.dataset}')

        other_datasets = ['diab', 'cold', 'CHD', 'lung', 'depr', 'pneu']
        other_datasets.remove(self.dataset)

        origin_path = f'{root}Haodf/'
        self.origin_inter_file = f'{origin_path}{self.dataset}_inter.csv'
        self.origin_doctor_file = f'{origin_path}{self.dataset}_doctor.csv'
        data_path = f'{root}'
        self.train_file = f'{data_path}{self.dataset}_train.csv'
        self.valid_file = f'{data_path}{self.dataset}_valid.csv'
        self.test_file = f'{data_path}{self.dataset}_test.csv'

        self.doctor_path = '{}{}_doctor.csv'.format(data_path, self.dataset)
        self.other_doctor_path = ['{}{}_doctor.csv'.format(data_path, d) for d in other_datasets]
        self.all_doctor_path = '{}{}_doctor_all.csv'.format(data_path, self.dataset)

        self.disease_tag_need_file = f'{root}doctor_rec/disease_tag_need/disease_tag_needed_{self.dataset}.txt'

        self.text_dcotor_columns = ['jiyu', 'zhuanyeshanchang', 'gerenjianjie',
                                    'keyanchengguo', 'shehuirenzhi', 'text_all_doctor']

        self.text_inter_columns = ['wanted_help', 'chronic disease', 'surgery history',
                                   'radiotherapy and chemotherapy history', 'disease history', 'medication usage',
                                   'disease', 'allergy history', 'major illness', 'text_all_patient']

        self.output_doctor_column = ['recvalue', 'patient_num', 'hospital', 'faculty', 'hornor', 'gift_value',
                                     'zhuanyefangxiang', 'good_review_rate', 'title', 'doctor_id',
                                     'jiyu_input_ids', 'jiyu_attention_mask',
                                     'zhuanyeshanchang_input_ids', 'zhuanyeshanchang_attention_mask',
                                     'gerenjianjie_input_ids', 'gerenjianjie_attention_mask',
                                     'keyanchengguo_input_ids', 'keyanchengguo_attention_mask',
                                     'shehuirenzhi_input_ids', 'shehuirenzhi_attention_mask',
                                     'text_all_doctor_input_ids', 'text_all_doctor_attention_mask']

        self.output_inter_column = ['doctor_id',  'neg_doctor_id', 'gender', 'age', 'height', 'weight',
                                    'hospital history', 'pregnancy situation', 'duration of illness', 'disease_tag',
                                    'wanted_help_input_ids', 'wanted_help_attention_mask',
                                    'chronic disease_input_ids', 'chronic disease_attention_mask',
                                    'surgery history_input_ids', 'surgery history_attention_mask',
                                    'radiotherapy and chemotherapy history_input_ids',
                                    'radiotherapy and chemotherapy history_attention_mask',
                                    'disease history_input_ids', 'disease history_attention_mask',
                                    'medication usage_input_ids', 'medication usage_attention_mask',
                                    'disease_input_ids', 'disease_attention_mask',
                                    'allergy history_input_ids', 'allergy history_attention_mask',
                                    'major illness_input_ids', 'major illness_attention_mask',
                                    'text_all_patient_input_ids', 'text_all_patient_attention_mask']



        # Model settings
        self.model_for_step_1 = args.model_for_step_1
        print(f'model_for_step_1: {self.model_for_step_1}')

        # Training settings
        self.batch_size = args.batch_size
        print(f'batch_size: {self.batch_size}')
        self.lr_recall = args.lr_recall
        print(f'lr_recall: {self.lr_recall}')
        self.lr_rank = args.lr_rank
        print(f'lr_rank: {self.lr_rank}')
        self.epochs = args.epochs
        print(f'epochs: {self.epochs}')
        self.embed_size = args.embed_size
        print(f'embed_size: {self.embed_size}')

        self.num_neg_easy_rank = args.num_neg_easy_rank
        print(f'num_neg_easy_rank: {self.num_neg_easy_rank}')
        self.num_neg_hard_rank = args.num_neg_hard_rank
        print(f'num_neg_hard_rank: {self.num_neg_hard_rank}')
        self.num_pos_sample_recall = args.num_neg_hard_rank
        print(f'num_pos_sample_recall: {self.num_pos_sample_recall}')
        self.num_neg_easy_recall = args.num_neg_easy_recall
        print(f'num_neg_easy_recall: {self.num_neg_easy_recall}')

        self.loss_func = args.loss_func
        print(f'loss_func: {self.loss_func}')

        self.eval_limit = args.eval_limit

        # Running settings
        self.device = args.device
        print(f'device: {self.device}')

        # load info
        self.doctor_loaded = False
        self.doctor_tokenized = False
        self.num_doctor_current = -1
        self.num_doctor_all = -1
        self.num_faculty = -1
        self.num_zhuanyefangxiang = -1
        self.num_hospital = -1
        self.all_doctor_df = -1
        self.dict_disease_doctor = {}

    def contrastive_loss(self, sample1, sample2, label):
        # label = 1: sample 1, 2 from same group; label = 0: from different group
        sim = nn.CosineSimilarity()
        dis = sim(sample1, sample2)
        margin = 1.0
        loss = torch.mean((1-label) * torch.pow(dis, 2) +
                                      label * torch.pow(torch.clamp(margin - dis, min=0.0), 2))
        return loss

    def loss(self, pos, neg):
        hinge_m = 0.5
        if self.loss_func == 'bpr':
            loss_critic = -torch.mean(torch.log(torch.sigmoid(pos - neg)))
        if self.loss_func == 'hinge':
            # hinge loss
            temp = pos - neg
            loss_critic = torch.clamp(temp + hinge_m, min=0)
            loss_critic = torch.mean(loss_critic)
        elif self.loss_func == 'log':
            # log loss
            temp = pos - neg
            loss_critic = torch.log(1 + torch.exp(temp))
            loss_critic = torch.mean(loss_critic)
        elif self.loss_func == 'square_square':
            # square-square loss
            neg = torch.clamp(hinge_m - neg, min=0)
            loss_critic = pos ** 2 + neg ** 2
            loss_critic = torch.mean(loss_critic)
        elif self.loss_func == 'square_exp':
            # square_exp loss
            gama = 1
            neg = gama * torch.exp(-neg)
            loss_critic = pos ** 2 + neg
            loss_critic = torch.mean(loss_critic)

        return loss_critic

    def filter_output_list(self, list):
        return [float('{:.4f}'.format(i)) for i in list]