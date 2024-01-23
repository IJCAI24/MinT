import math
import os
import gzip
import json
import sys
import time
import random

import pandas
import torch
import transformers
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.utils.data as data

import config

random.seed(2023)

generate_traditional_set = False  # If generate dataset for traditional model

def load_all_doctor(config:config.Config, tokenizer):
    if config.doctor_loaded:
        return
    dtype = {'link': str, 'title': str, 'recvalue': float, 'patient_num': float, 'hospital': int, 'faculty': int,
             'hornor': float, 'gift_value': float, 'jiyu': str, 'zhuanyefangxiang': int, 'zhuanyeshanchang': str,
             'gerenjianjie': str, 'keyanchengguo': str, 'shehuirenzhi': str, 'good_review_rate': float,
             'doctor_id': int}
    doctor_df = pd.read_csv(config.doctor_path, sep='\t', keep_default_na=False, dtype=dtype)
    doctor_df['title'] = list(map(eval, doctor_df['title']))
    num_hospital = max(doctor_df['hospital']) + 1
    num_faculty = max(doctor_df['faculty']) + 1
    num_zhuanyefangxiang = max(doctor_df['zhuanyefangxiang']) + 1
    num_doctor = len(doctor_df)

    config.num_doctor_current = num_doctor

    all_doctor_df = doctor_df

    for path in config.other_doctor_path:
        doctor_df_now = pd.read_csv(path, sep='\t', dtype=dtype, keep_default_na=False)
        doctor_df_now['title'] = list(map(eval, doctor_df_now['title']))

        list_hospital = list(doctor_df_now['hospital'])
        list_faculty = list(doctor_df_now['faculty'])
        list_zhuanyefangxiang = list(doctor_df_now['zhuanyefangxiang'])
        list_doctor_id = list(doctor_df_now['doctor_id'])
        for i in range(len(doctor_df_now)):
            list_hospital[i] += num_hospital
            list_faculty[i] += num_faculty
            list_zhuanyefangxiang[i] += num_zhuanyefangxiang
            list_doctor_id[i] += num_doctor
        doctor_df_now['hospital'] = list_hospital
        doctor_df_now['faculty'] = list_faculty
        doctor_df_now['zhuanyefangxiang'] = list_zhuanyefangxiang
        doctor_df_now['doctor_id'] = list_doctor_id

        num_hospital += max(doctor_df_now['hospital']) + 1
        num_faculty += max(doctor_df_now['faculty']) + 1
        num_zhuanyefangxiang += max(doctor_df_now['zhuanyefangxiang']) + 1
        num_doctor += len(doctor_df_now)

        all_doctor_df = pd.concat([all_doctor_df, doctor_df_now], axis=0, ignore_index=True)

    # tokenize all text info
    for column in config.text_dcotor_columns:
        column_now = all_doctor_df[column].tolist()
        column_token = tokenizer.batch_encode_plus(column_now, padding=True, truncation=True)
        # all_doctor_df.drop(columns=[column])
        all_doctor_df[column + '_input_ids'] = column_token['input_ids']
        all_doctor_df[column + '_attention_mask'] = column_token['attention_mask']

    config.num_hospital = num_hospital
    config.num_faculty = num_faculty
    config.num_zhuanyefangxiang = num_zhuanyefangxiang
    config.num_doctor_all = num_doctor
    print(f'num_doctor_all: {config.num_doctor_all}')
    config.all_doctor_df = all_doctor_df
    config.doctor_loaded = True
    # print(all_doctor_df.columns.tolist())

    all_doctor_df.to_csv(config.all_doctor_path, sep='\t')

class Data(data.Dataset):
    def __init__(self, config: config.Config, set_type):
        super(Data, self).__init__()
        self.config = config
        self.set_type = set_type
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_for_step_1)

        if not (os.path.exists(config.train_file)
                and os.path.exists(config.valid_file)
                and os.path.exists(config.test_file)):
            print('ERROR: Can\'t find dataset file')
            print(f'train_file: {config.train_file}')
            print(f'valid_file: {config.valid_file}')
            print(f'test_file: {config.test_file}')
            exit()
        self.data_df = None
        self.load_dataset()
        load_all_doctor(config, self.tokenizer)
        self.tokenize_data()
        self.original_data_df = []
        self.process_df_update_config()
        self.process_dict_disease_doctor()
        if self.set_type == 'Train':
            self.original_data_df = self.data_df.copy()
        else:
            self.make_full_test_set()
        # print(self.data_df.columns.tolist())

    def process_dict_disease_doctor(self):
        # process dict_disease_doctor
        for i in range(len(self.data_df)):
            tag = self.data_df['disease_tag'][i]
            doctor_id = self.data_df['doctor_id'][i]
            if tag not in self.config.dict_disease_doctor:
                self.config.dict_disease_doctor[tag] = []
            if doctor_id not in self.config.dict_disease_doctor[tag]:
                self.config.dict_disease_doctor[tag].append(doctor_id)

    def load_dataset(self):
        data_path = ''
        if self.set_type == 'Train':
            data_path = self.config.train_file
        elif self.set_type == 'Valid':
            data_path = self.config.valid_file
        elif self.set_type == 'Test':
            data_path = self.config.test_file
        self.data_df = pd.read_csv(data_path, header=0, sep='\t', keep_default_na=False,
                                   dtype={'doctor_id': int, 'age': float, 'height': float, 'weight': float,
                                          'duration of illness': float, 'gender': int, 'pregnancy situation': int})
        print(f'len_raw_dataframe: {len(self.data_df)}')

    def process_df_update_config(self):
        # process df
        columns_eval = ['hospital history']
        for column in columns_eval:
            self.data_df[column] = list(map(eval, self.data_df[column]))

        # update config

    def tokenize_data(self):
        for column in self.config.text_inter_columns:
            column_now = self.data_df[column].tolist()
            column_token = self.tokenizer.batch_encode_plus(column_now, padding=True, truncation=True)

            # calculate unknown
            calculate_unknown = False
            if calculate_unknown:
                unknown_word = 0
                for text in column_now:
                    token_test = self.tokenizer.tokenize(text, truncation=True)
                    # print(token_test)
                    unknown_word += sum([1 for token in token_test if token == self.tokenizer.unk_token])
                print(f'unknown word in column {column}: {unknown_word}')

            # self.data_df.drop(columns=[column])
            self.data_df[column + '_input_ids'] = column_token['input_ids']
            self.data_df[column + '_attention_mask'] = column_token['attention_mask']

    def sample_neg(self):
        del self.data_df
        self.data_df = self.original_data_df.copy()
        
        num_neg_hard = self.config.num_neg_hard_rank
        num_neg_easy = self.config.num_neg_easy_rank
        num_doctor_current = self.config.num_doctor_current
        num_doctor_all = self.config.num_doctor_all
        neg_num = num_neg_easy + num_neg_hard

        columns = self.data_df.columns.tolist()
        self.data_df = pd.DataFrame(np.repeat(self.data_df.values, neg_num, axis=0), columns=columns)
        easy_doctor_list = range(num_doctor_current, num_doctor_all)
        list_neg_id = [-1 for _ in range(len(self.data_df))]

        for i in range(int(len(self.data_df) / neg_num)):
            start_num = i * neg_num
            doctor_now = self.data_df['doctor_id'][start_num]
            neg_easy_list = random.sample(easy_doctor_list, num_neg_easy)
            neg_hard_list = random.sample(list(range(num_doctor_current)), num_neg_hard)
            while doctor_now in neg_hard_list:
                neg_hard_list = random.sample(list(range(num_doctor_current)), num_neg_hard)

            for j in range(len(neg_easy_list)):
                list_neg_id[start_num+j] = neg_easy_list[j]
            for j in range(len(neg_hard_list)):
                list_neg_id[start_num+num_neg_easy+j] = neg_hard_list[j]
        self.data_df['neg_doctor_id'] = list_neg_id

    def make_full_test_set(self):
        '''
        ['Unnamed: 0', 'link', 'title', 'recvalue', 'patient_num', 'hospital', 'faculty', 'hornor', 'gift_value', 'jiyu', 'zhuanyefangxiang', 'zhuanyeshanchang', 'gerenjianjie', 'keyanchengguo', 'shehuirenzhi', 'good_review_rate', 'doctor_id', 'jiyu_input_ids', 'jiyu_attention_mask', 'zhuanyeshanchang_input_ids', 'zhuanyeshanchang_attention_mask', 'gerenjianjie_input_ids', 'gerenjianjie_attention_mask', 'keyanchengguo_input_ids', 'keyanchengguo_attention_mask', 'shehuirenzhi_input_ids', 'shehuirenzhi_attention_mask']
        '''
        num_doctor_all = self.config.num_doctor_all
        columns = self.data_df.columns.tolist()
        self.data_df = pd.DataFrame(np.repeat(self.data_df.values, num_doctor_all, axis=0), columns=columns)
        self.data_df['neg_doctor_id'] = [i % num_doctor_all for i in range(len(self.data_df))]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        list_output = []
        pos_doctor = self.data_df['doctor_id'][idx]
        neg_doctor = self.data_df['neg_doctor_id'][idx]
        for column in self.config.output_inter_column:
            list_output.append(self.data_df[column][idx])
        for column in self.config.output_doctor_column:
            list_output.append(self.config.all_doctor_df[column][pos_doctor])
        for column in self.config.output_doctor_column:
            list_output.append(self.config.all_doctor_df[column][neg_doctor])


        return list_output
