import math
import os
import random
import re
import argparse
import pandas as pd

random.seed(2023)

# datasets = ['diab', 'cold', 'CHD', 'lung', 'depr', 'pneu']


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default='diab',
                    help="dataset file name")

args = parser.parse_args()

dataset = '---'
root = '../Dataset/'
origin_path = f'{root}Haodf/'
origin_inter_file = f'{origin_path}{dataset}_inter.csv'
origin_doctor_file = f'{origin_path}{dataset}_doctor.csv'
data_path = f'{root}'
train_file = f'{data_path}{dataset}_train.csv'
valid_file = f'{data_path}{dataset}_valid.csv'
test_file = f'{data_path}{dataset}_test.csv'

info_path = f'{data_path}{dataset}_info.txt'
doctor_path = f'{data_path}{dataset}_doctor.csv'

disease_tag_need_file = f'{root}doctor_rec/disease_tag_need/disease_tag_needed_{dataset}.txt'

wanted_dcotor_columns = ['link', 'title', 'recvalue', 'patient_num', 'major', 'faculty', 'hornor',
                         'good_review', 'medium_bad_review', 'gift_value', 'jiyu', 'zhuanyefangxiang',
                         'zhuanyeshanchang',
                         'gerenjianjie', 'keyanchengguo', 'shehuirenzhi']

wanted_inter_columns = ['link', 'hospital history', 'wanted help 1', 'wanted help 2', 'pregnancy situation',
                        'duration of illness', 'chronic disease', 'surgery history',
                        'radiotherapy and chemotherapy history',
                        'disease history', 'medication usage', 'disease', 'disease description',
                        'allergy history', 'major illness',
                        'gender', 'age', 'height', 'weight', 'disease_tag2']

save_doctor_column = ['link', 'title', 'recvalue', 'patient_num', 'major', 'faculty', 'hornor',
                      'good_review', 'medium_bad_review', 'gift_value', 'jiyu', 'zhuanyefangxiang',
                      'zhuanyeshanchang',
                      'gerenjianjie', 'keyanchengguo', 'shehuirenzhi'
                      ]
save_inter_column = [

]


def remove_bracket(s):
    s = re.sub(u"\\(.*?\\)", "", s)
    return s


def process_doctor_df(doctor_df, inter_df):
    #################################### process doctor text info ####################################

    ############ filter columns and strip all text ############
    doctor_df = doctor_df[wanted_dcotor_columns]
    for column in wanted_dcotor_columns:
        column_now = doctor_df[column].tolist()
        column_now = list(map(str.strip, column_now))
        doctor_df[column] = column_now

    ############ map doctor link into idx ############
    dict_doctor_link_num = {}
    for i in range(len(doctor_df)):
        link = doctor_df['link'][i]
        # if len(link) > 0 and link not in dict_doctor_link_num:
        if link not in dict_doctor_link_num:
            dict_doctor_link_num[link] = len(dict_doctor_link_num)

    ############ process all text info ############
    columns = ['jiyu', 'zhuanyeshanchang', 'gerenjianjie', 'keyanchengguo', 'shehuirenzhi']
    columns_trans = ['wish', 'expertise', 'individual resume', 'scientific payoffs', 'social position']
    for column in columns:
        column_now = doctor_df[column]
        for i in range(len(column_now)):
            if len(column_now[i]) == 0:
                column_now[i] = 'empty'
            column_now[i] = columns_trans[columns.index(column)] + ' is ' + column_now[i]
        doctor_df[column] = column_now

    ############ process all number info ############
    # ['recvalue', 'patient_num', 'gift_value'] is processed
    for column in ['recvalue', 'patient_num', 'good_review', 'medium_bad_review', 'gift_value']:
        column_now = doctor_df[column].tolist()
        for i in range(len(column_now)):
            if column_now[i] == '':
                column_now[i] = 0
            else:
                try:
                    column_now[i] = float(column_now[i])
                except:
                    print(f'ERROR: error occurs when process column:"{column}"-"{column_now[i]}"')
                    column_now[i] = 0
        doctor_df[column] = column_now
    # 'good_review', 'medium_bad_review'
    column_good = doctor_df['good_review'].tolist()
    column_notgood = doctor_df['medium_bad_review'].tolist()
    for i in range(len(column_good)):
        if column_good[i] == '' or column_good[i] == 0:
            column_good[i] = 0
            continue
        if column_notgood[i] == '':
            column_notgood[i] = 0
        else:
            try:
                column_good[i] = int(column_good[i])
                column_notgood[i] = int(column_notgood[i])
            except:
                print(f'ERROR: error occurs when process column:"good_review"-"{column_good[i]}"')
                column_good[i] = column_good[i] / (column_good[i] + column_notgood[i])
        doctor_df['good_review_rate'] = column_good
    doctor_df.drop(columns=['good_review', 'medium_bad_review'], inplace=True)

    # 'hornor' is processed
    column_now = doctor_df['hornor'].tolist()
    for i in range(len(column_now)):
        if column_now[i] == '':
            column_now[i] = 0
        else:
            try:
                column_now[i] = int(re.findall(r"\d+", column_now[i])[-1])
            except:
                print(f'ERROR: error occurs when process column:"hornor"-"{column_now[i]}"')
                column_now[i] = 0
        doctor_df['hornor'] = column_now
    ############ process title info ############
    # 'title

    # title_translate['教授'] = 'professor'
    # title_translate['主治医师'] = 'attending physician'
    # title_translate['副主任医师'] = 'associate chief physician'
    # title_translate['讲师'] = 'lecturer'
    # title_translate['副教授'] = 'associate professor'
    # title_translate['副研究员'] = 'associate research fellow'
    # title_translate['助教'] = 'teaching assistant'
    # title_translate['研究员'] = 'research fellow'
    # title_translate['医师'] = 'physician'
    # title_translate['心理治疗师'] = 'psychotherapist'
    # title_translate['心理咨询师'] = 'psychological counselor'
    # title_translate['技师'] = 'technician'
    # title_translate['副主任技师'] = 'associate chief technician'
    # title_translate['主管技师'] = 'supervisor technician'
    # title_translate['主管药师'] = 'supervisor pharmacist'

    dict_title_idx = {
        'professor': 1,
        'attending physician': 2,
        'associate chief physician': 3,
        'lecturer': 4,
        'associate professor': 5,
        'associate research fellow': 6,
        'teaching assistant': 7,
        'research fellow': 8,
        'physician': 9,
        'psychotherapist': 10,
        'psychological counselor': 11,
        'technician': 12,
        'associate chief technician': 13,
        'supervisor technician': 14,
        'supervisor pharmacist': 15,
        'chief physician': 16
    }
    column_now = doctor_df['title'].tolist()
    for i in range(len(column_now)):
        if column_now[i] == '':
            column_now[i] = [0, 0]
        else:
            try:
                column_now[i] = eval(column_now[i])
                if column_now[i] == ['']:
                    column_now[i] = [0, 0]
                else:
                    for j in range(len(column_now[i])):
                        column_now[i][j] = dict_title_idx[column_now[i][j]]
                    # expand to len = 2
                    if len(column_now[i]) == 1:
                        column_now[i].append(0)
            except:
                print(column_now[i])
                print(column_now[i][0])
                print(f'ERROR: error occurs when process column:"title"-"{column_now[i]}"')
                column_now[i] = [0, 0]
        doctor_df['title'] = column_now

    ############ process zhuanyefangxiang info ############
    dict_column_zhuanyefangxiang = {}
    column_now = doctor_df['zhuanyefangxiang'].tolist()
    for i in range(len(column_now)):
        column_now[i] = column_now[i].lower()
        if column_now[i] == '':
            column_now[i] = -1
        if column_now[i] not in dict_column_zhuanyefangxiang:
            dict_column_zhuanyefangxiang[column_now[i]] = len(dict_column_zhuanyefangxiang)
        column_now[i] = dict_column_zhuanyefangxiang[column_now[i]]
        doctor_df['zhuanyefangxiang'] = column_now
    print('\nStatistic for zhuanyefangxiang')
    print(f'len(dict_column_zhuanyefangxiang): {len(dict_column_zhuanyefangxiang)}')
    for key in dict_column_zhuanyefangxiang.keys():
        print(key)

    ############ process faculty info ############
    dict_column_faculty = {}
    column_now = doctor_df['faculty'].tolist()
    for i in range(len(column_now)):
        column_now[i] = column_now[i].lower().split(':')[-1].strip()
        if column_now[i] == '':
            column_now[i] = -1
        if column_now[i] not in dict_column_faculty:
            dict_column_faculty[column_now[i]] = len(dict_column_faculty)
        column_now[i] = dict_column_faculty[column_now[i]]
        doctor_df['faculty'] = column_now
    print('\nStatistic for faculty')
    print(f'len(dict_column_faculty): {len(dict_column_faculty)}')
    for key in dict_column_faculty.keys():
        print(key)

    ############ process major info ############
    dict_column_hospital = {}
    column_now = doctor_df['major'].tolist()
    for i in range(len(column_now)):
        column_now[i] = column_now[i].lower()
        if column_now[i] == '':
            column_now[i] = -1
        if column_now[i] not in dict_column_hospital:
            dict_column_hospital[column_now[i]] = len(dict_column_hospital)
        column_now[i] = dict_column_hospital[column_now[i]]
        doctor_df['major'] = column_now
    doctor_df.rename(columns={'major': 'hospital'}, inplace=True)
    print('\nStatistic for major')
    print(f'len(dict_column_hospital): {len(dict_column_hospital)}')
    for key in dict_column_hospital.keys():
        print(key)

    ############ generate all text info for doctor ############
    column_all_text = []
    for i in range(len(doctor_df)):
        columns = ['zhuanyeshanchang', 'gerenjianjie', 'keyanchengguo', 'shehuirenzhi', 'jiyu']
        text_now = ''
        for column in columns:
            text_now += doctor_df[column][i]
        column_all_text.append(text_now)
    doctor_df['text_all_doctor'] = column_all_text

    #################################### process inter info ####################################

    ############ filter columns and strip all text ############
    inter_df = inter_df[wanted_inter_columns]
    for column in wanted_inter_columns:
        column_now = inter_df[column].tolist()
        column_now = list(map(str.strip, column_now))
        inter_df[column] = column_now

    ############ process disease_tag2 info ############
    column_now = inter_df['disease_tag2']
    for i in range(len(column_now)):
        column_now[i] = column_now[i].lower()
    inter_df['disease_tag2'] = column_now
    inter_df.rename(columns={'disease_tag2': 'disease_tag'}, inplace=True)

    ############ process history hospital info ############
    column_now = inter_df['hospital history']
    for i in range(len(column_now)):
        column_now[i] = column_now[i].lower()
        list_now = []
        for hospital in dict_column_hospital.keys():
            if str(hospital) in str(column_now[i]):
                list_now.append(dict_column_hospital[hospital])
                break
        if len(list_now) == 0:
            list_now = [-1]
        column_now[i] = list_now
    inter_df['hospital history'] = column_now

    ############ process wanted_help info ############
    column_now_1 = inter_df['wanted help 1'].tolist()
    column_now_2 = inter_df['wanted help 2'].tolist()
    column_now = []
    for i in range(len(column_now_1)):
        column_now.append('wanted help is ' + column_now_1[i] + ' ' + column_now_2[i])
    inter_df['wanted_help'] = column_now
    inter_df.drop(columns=['wanted help 1', 'wanted help 2'], inplace=True)

    ############ process gender_age_weight_height info ############
    columns = ['age', 'weight', 'height']
    for column in columns:
        column_now = inter_df[column]
        for i in range(len(column_now)):
            try:
                column_now[i] = float(column_now[i])
            except:
                print(f'ERROR: error occurs when process column:"{column}"-"{column_now[i]}"')

    columns = ['gender']
    for column in columns:
        column_now = inter_df[column]
        for i in range(len(column_now)):
            try:
                column_now[i] = int(column_now[i])
            except:
                print(f'ERROR: error occurs when process column:"{column}"-"{column_now[i]}"')
    # list_age = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57,
    #             60, 63, 66, 69, 72, 75, 78, 81, 84, 87, 90, 93, 96]
    # list_age = [0, 2, 5, 13, 18, 33, 48, 64, 78, 200]
    # column_age = inter_df['age']
    # for i in range(len(column_age)):
    #     age_now = column_age[i]
    #     for j in range(len(list_age)):
    #         if list_age[j] < age_now <= list_age[j + 1]:
    #             column_age[i] = j
    #             break
    #
    # inter_df['age'] = column_age

    ############ process pregnancy info ############
    column_now = inter_df['pregnancy situation'].tolist()
    for i in range(len(column_now)):
        column_now[i] = column_now[i].lower()
        if 'pregnancy' in column_now[i] and 'not' not in column_now[i]:
            print(column_now[i])
            column_now[i] = 1
        else:
            column_now[i] = 0
        inter_df['pregnancy situation'] = column_now

    ############ process illness duration info ############
    column_now = inter_df['duration of illness'].tolist()
    for i in range(len(column_now)):
        column_now[i] = column_now[i].lower().split(' ')
        have_word = -1
        time_word = ['years', 'months', 'weeks', 'days', 'year', 'month', 'week', 'day']
        dict_num = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'none': 9,
                    'ten': 10, 'a': 1}
        tw_mean = [365, 30, 7, 1, 365, 30, 7, 1]
        mean = -1
        for word in time_word:
            if word in column_now[i]:
                have_word = column_now[i].index(word)
                mean = tw_mean[time_word.index(word)]
                break
        if have_word == -1:
            column_now[i] = 0
        else:
            try:
                num = column_now[i][have_word - 1]
                if num in dict_num:
                    column_now[i] = dict_num[num] * mean
                else:
                    column_now[i] = float(re.findall(r"\d+", num)[-1]) * mean
            except:
                column_now[i] = 0
        inter_df['duration of illness'] = column_now

    ############ process medication usage disease disease description info ############
    # and chronic disease surgery history radiotherapy and chemotherapy history disease history
    # allergy history major illness
    columns = ['medication usage', 'disease', 'disease description', 'chronic disease', 'surgery history',
               'radiotherapy and chemotherapy history', 'disease history', 'allergy history', 'major illness']
    for column in columns:
        column_now = inter_df[column]
        for i in range(len(column_now)):
            column_now[i] = remove_bracket(column_now[i]).strip()
            if column_now[i] == '':
                column_now[i] = 'empty'
            column_now[i] = column + ' is ' + column_now[i]
        inter_df[column] = column_now

    ############ merge disease and disease description ########
    column_disease = inter_df['disease']
    column_disease_description = inter_df['disease description']

    for i in range(len(column_disease)):
        column_disease[i] = column_disease[i] + column_disease_description[i]
    inter_df['disease'] = column_disease

    inter_df.drop(columns=['disease description'], inplace=True)

    ############ generate all text info for patient ############
    column_all_text = []
    for i in range(len(inter_df)):
        columns = ['wanted_help', 'disease', 'chronic disease', 'disease history',
                   'surgery history', 'radiotherapy and chemotherapy history', 'allergy history', 'major illness',
                   'medication usage']
        text_now = ''
        for column in columns:
            text_now += inter_df[column][i]
        column_all_text.append(text_now)
    inter_df['text_all_patient'] = column_all_text

    return doctor_df, inter_df, dict_doctor_link_num


def process_dataset():
    global dataset
    dataset = args.dataset
    print('Making dataset...')
    inter_df = pd.read_csv(origin_inter_file, sep='\t', header=0, keep_default_na=False, dtype=str)
    doctor_df = pd.read_csv(origin_doctor_file, sep='\t', header=0, keep_default_na=False, dtype=str)
    '''
    Inter_columns = ['link', 'hospital history', 'wanted help 1', 'wanted help 2', 'pregnancy situation',
        'duration of illness', 'chronic disease', 'surgery history', 'radiotherapy and chemotherapy history',
        'disease history', 'medication usage', 'disease', 'disease description', 'height and weight', 
        'allergy histor', 'major illness', 'consultation suggestion', 'medical record summary', 
        'preliminary diagnosis', 'doctor advice', 'gender', 'age', 'height', 'weight', 'disease_tag', 'disease_tag2']


    Doctor_columns = ['name', 'link', 'title', 'recvalue', 'patient_num', 'major', 'faculty', 'hornor',
        'good_review', 'medium_bad_review', 'gift_value', 'jiyu', 'zhuanyefangxiang', 'zhuanyeshanchang', 
        'gerenjianjie', 'wodetuandui', 'keyanchengguo', 'shehuirenzhi', 'huojiangrongyu', 'jinxiujingli', 
        'gongzuojingli', 'jiaoyujingli']
    '''

    ############ process df  ############
    doctor_df, inter_df, dict_doctor_link_num = process_doctor_df(doctor_df=doctor_df, inter_df=inter_df)

    ############ process  ############
    # filter disease_tag
    if os.path.exists(disease_tag_need_file):
        with open(disease_tag_need_file, 'r') as f:

            lines = f.readlines()
            dict_tag_need = eval(lines[1].strip())
            # conver all key to lower case
            for key in list(dict_tag_need.keys()):
                dict_tag_need[key.lower()] = dict_tag_need[key]
    else:
        print(f'ERROR: no disease_tag_need_file found: {disease_tag_need_file}')

    # list_inter_all = []
    # filter disease tag and contruct dict_doctor_inters
    dict_doctor_inters = {}
    tag_error = 0
    for i in range(len(inter_df)):
        disease_tag = inter_df['disease_tag'][i]
        if disease_tag not in dict_tag_need:
            tag_error += 1
            continue
            # print(f'ERROR: disease tag: "{disease_tag}" not in dict_tag_need')
        elif dict_tag_need[disease_tag] == 0:
            continue

        try:
            doctor_num_now = dict_doctor_link_num[inter_df['link'][i]]
        except:
            print(f'ERROR: inter link not in doctor list, link = \'{inter_df["link"][i]}\'')
            exit()

        if doctor_num_now not in dict_doctor_inters:
            dict_doctor_inters[doctor_num_now] = []
        dict_doctor_inters[doctor_num_now].append(i)

    print(f'Tag Error: {tag_error}')
    train_list, valid_list, test_list = [], [], []
    # only valid doctor could have num thus to remove num for invalid doctors
    dict_doctor_valid_idx = {}
    valid_doctor_num = 0
    for doctor_num in dict_doctor_inters.keys():
        doctor_inters_now = dict_doctor_inters[doctor_num]
        num_inters_now = len(doctor_inters_now)
        if num_inters_now < 3:
            continue
        dict_doctor_valid_idx[doctor_num] = valid_doctor_num
        valid_doctor_num += 1

        test_num = math.ceil(num_inters_now * 0.2)
        valid_num = math.ceil(num_inters_now * 0.1)
        # train_num = num_inters_now - test_num - valid_num

        test_list_now = random.sample(doctor_inters_now, test_num)
        test_list += test_list_now
        for sample in test_list_now:
            doctor_inters_now.remove(sample)
        valid_list_now = random.sample(doctor_inters_now, valid_num)
        valid_list += valid_list_now
        for sample in valid_list_now:
            doctor_inters_now.remove(sample)
        train_list += doctor_inters_now
    print(f'valid_doctor: {valid_doctor_num}')
    print(f'len_train_set: {len(train_list)}, len_valid_set: {len(valid_list)}, len_test_set: {len(test_list)} ')

    ############ process valid doctor ############
    dict_doctor_valid_idx_reverse = {}
    for key in dict_doctor_valid_idx.keys():
        dict_doctor_valid_idx_reverse[dict_doctor_valid_idx[key]] = key
    valid_doctor_list = []
    for i in range(len(dict_doctor_valid_idx_reverse)):
        valid_doctor_list.append(dict_doctor_valid_idx_reverse[i])
    doctor_df = doctor_df.loc[valid_doctor_list, :]
    doctor_df.reset_index(drop=True, inplace=True)

    doctor_df['doctor_id'] = [-1 for _ in range(len(doctor_df))]
    for i in range(len(doctor_df)):
        doctor_num_now = dict_doctor_link_num[doctor_df['link'][i]]
        if doctor_num_now in dict_doctor_valid_idx:
            doctor_df['doctor_id'][i] = dict_doctor_valid_idx[doctor_num_now]

    # process zhuanyefangxiang faculty hospital
    dict_zhuanyefangxiang = {-1: 0}
    column_zhuanyefangxiang = doctor_df['zhuanyefangxiang']
    for i in range(len(column_zhuanyefangxiang)):
        if column_zhuanyefangxiang[i] not in dict_zhuanyefangxiang:
            dict_zhuanyefangxiang[column_zhuanyefangxiang[i]] = len(dict_zhuanyefangxiang)
        column_zhuanyefangxiang[i] = dict_zhuanyefangxiang[column_zhuanyefangxiang[i]]
    doctor_df['zhuanyefangxiang'] = column_zhuanyefangxiang

    dict_faculty = {-1: 0}
    column_faculty = doctor_df['faculty']
    for i in range(len(column_faculty)):
        if column_faculty[i] not in dict_faculty:
            dict_faculty[column_faculty[i]] = len(dict_faculty)
        column_faculty[i] = dict_faculty[column_faculty[i]]
    doctor_df['faculty'] = column_faculty

    dict_hospital = {-1: 0}
    column_hospital = doctor_df['hospital']
    for i in range(len(column_hospital)):
        if column_hospital[i] not in dict_hospital:
            dict_hospital[column_hospital[i]] = len(dict_hospital)
        column_hospital[i] = dict_hospital[column_hospital[i]]
    doctor_df['hospital'] = column_hospital

    # process inter hospital
    column_hospital_inter = inter_df['hospital history']
    for i in range(len(column_hospital_inter)):
        line = []
        for hospital in column_hospital_inter[i]:
            if hospital in dict_hospital:
                line.append(dict_hospital[hospital])
        line = line[:2]
        line += [0 for _ in range(2-len(line))]
        column_hospital_inter[i] = line
    inter_df['hospital history'] = column_hospital_inter

    ############ Save doctor info ############
    doctor_df.to_csv(doctor_path, sep='\t')

    # Add doctor idx in inter df
    inter_df['doctor_id'] = [-1 for _ in range(len(inter_df))]
    for i in range(len(inter_df)):
        if inter_df['link'][i] in dict_doctor_link_num:
            doctor_num_now = dict_doctor_link_num[inter_df['link'][i]]
            if doctor_num_now in dict_doctor_valid_idx:
                inter_df['doctor_id'][i] = dict_doctor_valid_idx[doctor_num_now]
        else:
            print(f'ERROR: inter_df[\'link\'][i] not in dict_doctor_link_num, keyerror: {inter_df["link"][i]}')
    # Save train valid test dataset
    train_df = inter_df.loc[train_list, :]
    train_df.reset_index(drop=True, inplace=True)
    train_df.to_csv(train_file, sep='\t')

    valid_df = inter_df.loc[valid_list, :]
    valid_df.reset_index(drop=True, inplace=True)
    valid_df.to_csv(valid_file, sep='\t')

    test_df = inter_df.loc[test_list, :]
    test_df.reset_index(drop=True, inplace=True)
    test_df.to_csv(test_file, sep='\t')
    if -1 in train_df['doctor_id'] or -1 in valid_df['doctor_id'] or -1 in test_df['doctor_id']:
        print('ERROR: -1 in train_df[\'doctor_id\']')

    print(f'Making dataset {dataset} done!')
    print(f'len_train_set: {len(train_df)}, len_valid_set: {len(valid_df)}, len_test_set: {len(test_df)}')
    print(f'valid_doctor: {len(dict_doctor_valid_idx)}')


if __name__ == "__main__":
    print('Note: You are running dataset_process.py')
    process_dataset()