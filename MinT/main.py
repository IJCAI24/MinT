import os
import time as tt
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as interpolate
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn

import model
import config
import evaluate
import dataset
from torch.optim import AdamW
from torch.optim import SGD
from torch.optim import Adam

# Random seeds
np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)

parser = argparse.ArgumentParser()
# Dataset settings
#  ['diab', 'cold', 'CHD', 'lung', 'depr', 'pneu']
parser.add_argument("--dataset", type=str, default='tangniaobing',
                    help="dataset file name")

# Training settings

parser.add_argument("--batch_size", type=int, default=16,
                    help="batch size for training")
parser.add_argument("--lr_recall", type=float, default=0.01,
                    help="learning rate for doctor recall model")
parser.add_argument("--lr_rank", type=float, default=0.001,
                    help="learning rate for doctor rank model")
parser.add_argument("--epochs", type=int, default=50,
                    help="num of training epochs")
parser.add_argument("--embed_size", type=int, default=64,
                    help="embedding size for embedding vectors")
parser.add_argument("--eval_limit", type=int, default=1,
                    help="batch size in eval to save memory")
parser.add_argument("--num_neg_easy_rank", type=int, default=2,
                    help="easy neg sample number for each test record in rank")
parser.add_argument("--num_neg_hard_rank", type=int, default=2,
                    help="hard neg sample number for each test record in rank")
parser.add_argument("--num_pos_sample_recall", type=int, default=4,
                    help="pos sample number for each test record in recall")
parser.add_argument("--num_neg_easy_recall", type=int, default=1,
                    help="easy neg sample number for each pos sample in rank")
parser.add_argument("--loss_func", type=str, default='bpr',
                    help="choose loss function in {bpr, hinge, log, square_square, square_exp")

# Model settings
# ['albert-base-v2'
# 'bert-base-uncased'
# 'distilbert-base-uncased'
# 'google/mobilebert-uncased'
# 'huawei-noah/TinyBERT_General_4L_312D'
# 'google/electra-small-discriminator'
parser.add_argument("--model_for_step_1", type=str, default='bert-base-uncased',
                    help="text encoder for step 1")

# Running settings
parser.add_argument("--device", type=str, default="cuda:0",
                    help="choose device to train model")
# parser.add_argument("--device", type=str, default="cpu",
#                     help="choose device to train model")

args = parser.parse_args()

############################################# Load config #############################################
print('\nSetting Config...\n')
config = config.Config(args=args)

############################################# Prepare data #############################################
print('\nPreparing dataset...\n')
print('Preparing train dataset')
train_dataset_recall = dataset.Data(config=config, set_type='Train')
train_loader_recall = data.DataLoader(train_dataset_recall, batch_size=config.batch_size, shuffle=True)
train_dataset_rank = dataset.Data(config=config, set_type='Train')
train_loader_rank = data.DataLoader(train_dataset_rank, batch_size=config.batch_size, shuffle=True)
# print('Preparing valid dataset')
# valid_dataset = dataset.Data(config=config, set_type='Valid')
# valid_loader = data.DataLoader(valid_dataset, batch_size=config.num_doctor_current, shuffle=False)
print('Preparing test dataset')
test_dataset = dataset.Data(config=config, set_type='Test')
test_loader = data.DataLoader(test_dataset, batch_size=config.num_doctor_all, shuffle=False)

if_test_without_model = False
if if_test_without_model:
    train_loader_recall.dataset.sample_neg()
    data = train_loader_recall.__iter__()
    i = 0
    while True:
        if data.__len__()>0:
            print(data.__next__()[0].to(config.device))

############################################# Create Model #############################################
print('\nCreating Model...\n')
Encoder = model.Text_Encoder(config=config).to(config.device)
Doctor_Recall = model.Doctor_Recall(config=config, is_training=True, Encoder=Encoder).to(config.device)
Doctor_Rank = model.Doctor_Rank(config=config, is_training=True, Encoder=Encoder).to(config.device)


if_only_test = True
if if_only_test:
    with (torch.no_grad()):
        Doctor_Recall.eval()
        Doctor_Rank.eval()
        Doctor_Recall.is_training = False
        Doctor_Rank.is_training = False
        p_r, pt_r, p_rc, r_rc, h_rc, m_rc, n_rc, p_rko, r_rko, h_rko, m_rko, n_rko, p_rk, r_rk, h_rk, m_rk, n_rk = \
            evaluate.evaluate_doctor_rec(
                Doctor_Recall=Doctor_Recall, Doctor_Rank=Doctor_Rank, loader=test_loader, config=config)
    print(f'precision_recall: {p_r}\n'
          f'precision_tag_racall: {pt_r}\n'
          f'precision_recall: {p_rc}\n'
          f'recall_recall: {r_rc}\n'
          f'hr_recall: {h_rc}\n'
          f'mrr_recall: {m_rc}\n'
          f'ndcg_recall: {n_rc}\n'
          f'precision_rank_only: {p_rko}\n'
          f'recall_rank_only: {r_rko}\n'
          f'hr_rank_only: {h_rko}\n'
          f'mrr_rank_only: {m_rko}\n'
          f'ndcg_rank_only: {n_rko}\n'
          f'precision_rank_RECAL: {p_rk}\n'
          f'recall_rank_RECAL: {r_rk}\n'
          f'hr_rank_RECAL: {h_rk}\n'
          f'mrr_rank_RECAL: {m_rk}\n'
          f'ndcg_rank_RECAL: {n_rk}\n')

############################################# Training #############################################
print('\nTraining...\n')
optimizer_recall = Adam(Doctor_Recall.parameters(), lr=config.lr_recall, weight_decay=0.001)
optimizer_rank = Adam(Doctor_Rank.parameters(), lr=config.lr_rank, weight_decay=0.001)
list_loss_recall, list_loss_rank = [], []

for epoch in range(config.epochs):
    train_loader_recall.dataset.sample_neg()
    # train_loader_rank.dataset.sample_neg()
    start_time = tt.time()
    Doctor_Recall.train()
    Doctor_Recall.is_training = True
    Doctor_Rank.train()
    Doctor_Rank.is_training = True
    criteria = config.loss
    count = 0
    for data in tqdm(train_loader_recall):
        count += 1

        # region column for doctor and inter
        output_doctor_column = ['recvalue', 'patient_num', 'hospital', 'faculty', 'hornor', 'gift_value',
                                     'zhuanyefangxiang', 'good_review_rate', 'title', 'doctor_id',
                                     'jiyu_input_ids', 'jiyu_attention_mask',
                                     'zhuanyeshanchang_input_ids', 'zhuanyeshanchang_attention_mask',
                                     'gerenjianjie_input_ids', 'gerenjianjie_attention_mask',
                                     'keyanchengguo_input_ids', 'keyanchengguo_attention_mask',
                                     'shehuirenzhi_input_ids', 'shehuirenzhi_attention_mask',
                                     'text_all_doctor_input_ids', 'text_all_doctor_attention_mask']

        output_inter_column = ['doctor_id',  'neg_doctor_id', 'gender', 'age', 'height', 'weight',
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
        # endregion

        # region Process Input
        p_doctor_id = data[0].to(config.device)
        n_doctor_id = data[1].to(config.device)
        gender = data[2].to(config.device)
        age = data[3].to(config.device)
        height = data[4].to(config.device)
        weight = data[5].to(config.device)
        hospital_history = torch.stack(data[6], 0).transpose(0, 1).to(config.device)
        pregnancy_situation = data[7].to(config.device)
        duration_of_illness = data[8].to(config.device)
        wanted_help_i = torch.stack(data[10], 0).transpose(0, 1).to(config.device)
        wanted_help_m = torch.stack(data[11], 0).transpose(0, 1).to(config.device)
        chronic_disease_i = torch.stack(data[12], 0).transpose(0, 1).to(config.device)
        chronic_disease_m = torch.stack(data[13], 0).transpose(0, 1).to(config.device)
        surgery_history_i = torch.stack(data[14], 0).transpose(0, 1).to(config.device)
        surgery_history_m = torch.stack(data[15], 0).transpose(0, 1).to(config.device)
        chemotherapy_history_i = torch.stack(data[16], 0).transpose(0, 1).to(config.device)
        chemotherapy_history_m = torch.stack(data[17], 0).transpose(0, 1).to(config.device)
        disease_history_i = torch.stack(data[18], 0).transpose(0, 1).to(config.device)
        disease_history_m = torch.stack(data[19], 0).transpose(0, 1).to(config.device)
        medication_usage_i = torch.stack(data[20], 0).transpose(0, 1).to(config.device)
        medication_usage_m = torch.stack(data[21], 0).transpose(0, 1).to(config.device)
        disease_i = torch.stack(data[22], 0).transpose(0, 1).to(config.device)
        disease_m = torch.stack(data[23], 0).transpose(0, 1).to(config.device)
        allergy_history_i = torch.stack(data[24], 0).transpose(0, 1).to(config.device)
        allergy_history_m = torch.stack(data[25], 0).transpose(0, 1).to(config.device)
        major_illness_i = torch.stack(data[26], 0).transpose(0, 1).to(config.device)
        major_illness_m = torch.stack(data[27], 0).transpose(0, 1).to(config.device)
        text_all_patient_i = torch.stack(data[28], 0).transpose(0, 1).to(config.device)
        text_all_patient_m = torch.stack(data[29], 0).transpose(0, 1).to(config.device)

        pos_recvalue = data[30].to(config.device)
        pos_patient_num = data[31].to(config.device)
        pos_hospital = data[32].to(config.device)
        pos_faculty = data[33].to(config.device)
        pos_hornor = data[34].to(config.device)
        pos_gift_value = data[35].to(config.device)
        pos_zhuanyefangxiang = data[36].to(config.device)
        pos_good_review_rate = data[37].to(config.device)
        pos_title = torch.stack(data[38], 0).transpose(0, 1).to(config.device)
        pos_doctor_id = data[39].to(config.device)
        pos_jiyu_i = torch.stack(data[40], 0).transpose(0, 1).to(config.device)
        pos_jiyu_m = torch.stack(data[41], 0).transpose(0, 1).to(config.device)
        pos_zhuanyeshanchang_i = torch.stack(data[42], 0).transpose(0, 1).to(config.device)
        pos_zhuanyeshanchang_m = torch.stack(data[43], 0).transpose(0, 1).to(config.device)
        pos_gerenjianjie_i = torch.stack(data[44], 0).transpose(0, 1).to(config.device)
        pos_gerenjianjie_m = torch.stack(data[45], 0).transpose(0, 1).to(config.device)
        pos_keyanchengguo_i = torch.stack(data[46], 0).transpose(0, 1).to(config.device)
        pos_keyanchengguo_m = torch.stack(data[47], 0).transpose(0, 1).to(config.device)
        pos_shehuirenzhi_i = torch.stack(data[48], 0).transpose(0, 1).to(config.device)
        pos_shehuirenzhi_m = torch.stack(data[49], 0).transpose(0, 1).to(config.device)
        pos_text_all_doctor_i = torch.stack(data[50], 0).transpose(0, 1).to(config.device)
        pos_text_all_doctor_m = torch.stack(data[51], 0).transpose(0, 1).to(config.device)

        neg_recvalue = data[52].to(config.device)
        neg_patient_num = data[53].to(config.device)
        neg_hospital = data[54].to(config.device)
        neg_faculty = data[55].to(config.device)
        neg_hornor = data[56].to(config.device)
        neg_gift_value = data[57].to(config.device)
        neg_zhuanyefangxiang = data[58].to(config.device)
        neg_good_review_rate = data[59].to(config.device)
        neg_title = torch.stack(data[60], 0).transpose(0, 1).to(config.device)
        neg_doctor_id = data[61].to(config.device)
        neg_jiyu_i = torch.stack(data[62], 0).transpose(0, 1).to(config.device)
        neg_jiyu_m = torch.stack(data[63], 0).transpose(0, 1).to(config.device)
        neg_zhuanyeshanchang_i = torch.stack(data[64], 0).transpose(0, 1).to(config.device)
        neg_zhuanyeshanchang_m = torch.stack(data[65], 0).transpose(0, 1).to(config.device)
        neg_gerenjianjie_i = torch.stack(data[66], 0).transpose(0, 1).to(config.device)
        neg_gerenjianjie_m = torch.stack(data[67], 0).transpose(0, 1).to(config.device)
        neg_keyanchengguo_i = torch.stack(data[68], 0).transpose(0, 1).to(config.device)
        neg_keyanchengguo_m = torch.stack(data[69], 0).transpose(0, 1).to(config.device)
        neg_shehuirenzhi_i = torch.stack(data[70], 0).transpose(0, 1).to(config.device)
        neg_shehuirenzhi_m = torch.stack(data[71], 0).transpose(0, 1).to(config.device)
        neg_text_all_doctor_i = torch.stack(data[72], 0).transpose(0, 1).to(config.device)
        neg_text_all_doctor_m = torch.stack(data[73], 0).transpose(0, 1).to(config.device)
        # endregion
        # torch.autograd.set_detect_anomaly(True)

        if_train_recall = False
        if if_train_recall:
            Doctor_Recall.zero_grad()
            output_doctor_recall_pos = Doctor_Recall(text_all_patient_i, text_all_patient_m,
                                                     pos_text_all_doctor_i, pos_text_all_doctor_m)
            output_doctor_recall_neg = Doctor_Recall(text_all_patient_i, text_all_patient_m,
                                                     neg_text_all_doctor_i, neg_text_all_doctor_m)

            loss_recall = criteria(output_doctor_recall_pos, output_doctor_recall_neg)
            list_loss_recall.append(loss_recall.item())
            loss_recall.backward()
            optimizer_recall.step()
        else:
            output_doctor_recall_pos = torch.tensor([-1])
            output_doctor_recall_neg = torch.tensor([-1])

        if_train_rank = True
        if if_train_rank:
            Doctor_Rank.zero_grad()
            # region output_doctor_rank_pos = Doctor_Rank(xxx
            output_doctor_rank_pos = Doctor_Rank(p_doctor_id, gender, age, height, weight,
                                                 hospital_history, pregnancy_situation, duration_of_illness,
                                                 wanted_help_i, wanted_help_m, chronic_disease_i, chronic_disease_m,
                                                 surgery_history_i, surgery_history_m, chemotherapy_history_i,
                                                 chemotherapy_history_m,
                                                 disease_history_i, disease_history_m, medication_usage_i,
                                                 medication_usage_m, disease_i, disease_m,
                                                 allergy_history_i, allergy_history_m, major_illness_i, major_illness_m,

                                                 pos_recvalue, pos_patient_num, pos_hospital, pos_faculty,
                                                 pos_hornor, pos_gift_value, pos_zhuanyefangxiang, pos_good_review_rate,
                                                 pos_title,
                                                 pos_jiyu_i, pos_jiyu_m, pos_zhuanyeshanchang_i, pos_zhuanyeshanchang_m,
                                                 pos_gerenjianjie_i, pos_gerenjianjie_m, pos_keyanchengguo_i,
                                                 pos_keyanchengguo_m,
                                                 pos_shehuirenzhi_i, pos_shehuirenzhi_m)
            # endregion

            # region output_doctor_rank_neg = Doctor_Rank(xxx
            output_doctor_rank_neg = Doctor_Rank(n_doctor_id, gender, age, height, weight,
                                                 hospital_history, pregnancy_situation, duration_of_illness,
                                                 wanted_help_i, wanted_help_m, chronic_disease_i, chronic_disease_m,
                                                 surgery_history_i, surgery_history_m, chemotherapy_history_i,
                                                 chemotherapy_history_m,
                                                 disease_history_i, disease_history_m, medication_usage_i,
                                                 medication_usage_m, disease_i, disease_m,
                                                 allergy_history_i, allergy_history_m, major_illness_i, major_illness_m,

                                                 neg_recvalue, neg_patient_num, neg_hospital, neg_faculty,
                                                 neg_hornor, neg_gift_value, neg_zhuanyefangxiang, neg_good_review_rate,
                                                 neg_title,
                                                 neg_jiyu_i, neg_jiyu_m, neg_zhuanyeshanchang_i, neg_zhuanyeshanchang_m,
                                                 neg_gerenjianjie_i, neg_gerenjianjie_m, neg_keyanchengguo_i,
                                                 neg_keyanchengguo_m,
                                                 neg_shehuirenzhi_i, neg_shehuirenzhi_m)
            # endregion

            loss_rank = criteria(output_doctor_rank_pos, output_doctor_rank_neg)
            list_loss_rank.append(loss_rank.item())
            loss_rank.backward()
            optimizer_rank.step()
        else:
            output_doctor_rank_pos = torch.tensor([-1])
            output_doctor_rank_neg = torch.tensor([-1])
        print(f'\n output_doctor_recall_pos:'
              f' {config.filter_output_list(output_doctor_recall_pos.cpu().detach().numpy().tolist())}'
              f'\n output_doctor_recall_neg: '
              f'{config.filter_output_list(output_doctor_recall_neg.cpu().detach().numpy().tolist())}'
              f'\n output_doctor_rank_pos: '
              f'{config.filter_output_list(output_doctor_rank_pos.cpu().detach().numpy().tolist())}'
              f'\n output_doctor_rank_neg:'
              f' {config.filter_output_list(output_doctor_rank_neg.cpu().detach().numpy().tolist())}')
        print(f'\n recall_loss: {config.filter_output_list(list_loss_recall)}'
              f'\n rank_loss: {config.filter_output_list(list_loss_rank)} ')

        if_test_in_epoch = False
        if if_test_in_epoch and count % 15 == 1:
            elapsed_time = tt.time() - start_time
            print(f'count: {count}, Time: {tt.strftime("%H: %M: %S", tt.gmtime(elapsed_time))}')

            Doctor_Recall.eval()
            Doctor_Recall.is_training = False
            Doctor_Recall.reset_for_evaluate()
            Doctor_Rank.eval()
            Doctor_Rank.is_training = False
            Doctor_Rank.reset_for_evaluate()
            with (torch.no_grad()):
                p_r, p_rc, r_rc, h_rc, m_rc, n_rc, p_rko, r_rko, h_rko, m_rko, n_rko, p_rk, r_rk, h_rk, m_rk, n_rk = \
                    evaluate.evaluate_doctor_rec(
                        Doctor_Recall=Doctor_Recall, Doctor_Rank=Doctor_Rank, loader=test_loader, config=config)
            print(f'precision_recall: {p_r}\n'
                  f'hr_recall: {h_rc}\n'
                  f'precision_rank_only: {p_rko}\n'
                  f'recall_rank_only: {r_rko}\n'
                  f'hr_rank_only: {h_rko}\n'
                  f'mrr_rank_only: {m_rko}\n'
                  f'ndcg_rank_only: {n_rko}\n'
                  f'precision_rank_RECAL: {p_rk}\n'
                  f'recall_rank_RECAL: {r_rk}\n'
                  f'hr_rank_RECAL: {h_rk}\n'
                  f'mrr_rank_RECAL: {m_rk}\n'
                  f'ndcg_rank_RECAL: {n_rk}\n'
                  f'list_loss_recall: {str(list_loss_recall)}\n'
                  f'list_loss_rank: {str(list_loss_rank)}\n')
            Doctor_Recall.train()
            Doctor_Recall.is_training = True
            Doctor_Rank.train()
            Doctor_Rank.is_training = True

    elapsed_time = tt.time() - start_time
    print('Epoch: {}, Time: {}'.format(epoch, tt.strftime("%H: %M: %S", tt.gmtime(elapsed_time))))
    if_test = False
    if if_test:
        Doctor_Recall.eval()
        Doctor_Recall.is_training = False
        Doctor_Recall.reset_for_evaluate()
        Doctor_Rank.eval()
        Doctor_Rank.is_training = False
        Doctor_Rank.reset_for_evaluate()
        with (torch.no_grad()):
            p_r, p_rc, r_rc, h_rc, m_rc, n_rc, p_rko, r_rko, h_rko, m_rko, n_rko, p_rk, r_rk, h_rk, m_rk, n_rk = \
                evaluate.evaluate_doctor_rec(
                    Doctor_Recall=Doctor_Recall, Doctor_Rank=Doctor_Rank, loader=test_loader, config=config)
        print(f'precision_recall: {p_r}\n'
              f'hr_recall: {h_rc}\n'
              f'precision_rank_only: {p_rko}\n'
              f'recall_rank_only: {r_rko}\n'
              f'hr_rank_only: {h_rko}\n'
              f'mrr_rank_only: {m_rko}\n'
              f'ndcg_rank_only: {n_rko}\n'
              f'precision_rank_RECAL: {p_rk}\n'
              f'recall_rank_RECAL: {r_rk}\n'
              f'hr_rank_RECAL: {h_rk}\n'
              f'mrr_rank_RECAL: {m_rk}\n'
              f'ndcg_rank_RECAL: {n_rk}\n'
              f'list_loss_recall: {str(list_loss_recall)}\n'
              f'list_loss_rank: {str(list_loss_rank)}\n')


# Evaluation
print('\nStart evaluation...\n')

# print(f'precision: {precision}\nrecall: {recall}\nhr: {hr}\nmrr: {mrr}\nndcg: {ndcg}')
