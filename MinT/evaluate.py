import sys
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import math

import time as tt

import config


def m_NDCG(actual, predicted, topk):
    res = 0
    k = min(topk, len(actual))
    idcg = idcg_k(k)
    dcg_k = sum([int(predicted[j] in set(actual)) / math.log(j + 2, 2) for j in range(topk)])
    res += dcg_k / idcg
    return res / float(len(actual))


def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def m_MRR(targets, pred):
    m = 0
    for p in pred:
        if p in targets:
            index = pred.index(p)
            m += np.reciprocal(float(index + 1))
    return m / len(pred)


def compute_precision_recall(targets, predictions, k):
    pred = predictions[:k]
    num_hit = len(set(pred).intersection(set(targets)))
    precision = float(num_hit) / len(pred)
    recall = float(num_hit) / len(targets)
    hr = min(1, num_hit)
    mrr = m_MRR(targets, pred)
    ndcg = m_NDCG(targets, pred, k)
    return precision, recall, hr, mrr, ndcg
    # return hr, ndcg


def evaluate_doctor_rec(Doctor_Recall, Doctor_Rank, loader, config: config.Config):
    time_need = [[], []]
    time_before_recall, time_after_recall, time_before_rank, time_after_rank = 0, 0, 0, 0
    eval_limit = config.eval_limit
    device = config.device
    PRECISION_RANK_only, RECALL_RANK_only, MRR_RANK_only, NDCG_RANK_only, HR_RANK_only = [], [], [], [], []
    PRECISION_RANK, RECALL_RANK, MRR_RANK, NDCG_RANK, HR_RANK = [], [], [], [], []
    P_RECALL, DT_RECALL, RECALL_RECALL, PRECISION_RECALL, HR_RECALL, MRR_RECALL, NDCG_RECALL = [], [], [], [], [], [], []
    count = 0

    for data in tqdm(loader):
        # if count == 10:
        #     break
        count += 1

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
        disease_tag = data[9]
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
        # print('neg_text_all_doctor_i_1')
        # print(neg_text_all_doctor_i[0][:10].cpu().numpy().tolist())
        # print(neg_text_all_doctor_i[1][:10].cpu().numpy().tolist())
        # print(neg_text_all_doctor_i[2][:10].cpu().numpy().tolist())
        # print(neg_text_all_doctor_i[3][:10].cpu().numpy().tolist())
        # print(neg_text_all_doctor_i[4][:10].cpu().numpy().tolist())
        # endregion

        recall_output = torch.tensor([]).to(device)
        rank_output = torch.tensor([]).to(device)
        for i in range(math.ceil(len(p_doctor_id)/eval_limit)):
            s = i * eval_limit
            e = min((i+1) * eval_limit, len(p_doctor_id))
            if count <= 1 and s % (10 * config.eval_limit) == 0:
                print(s, e)
            if_recall = True
            if if_recall:
                time_before_recall = tt.time()
                output_doctor_recall_neg = Doctor_Recall(
                    text_all_patient_i[s:e], text_all_patient_m[s:e],
                    neg_text_all_doctor_i[s:e], neg_text_all_doctor_m[s:e],
                )
                time_after_recall = tt.time()
            else:
                output_doctor_recall_neg = torch.tensor([0.0 for _ in range(e-s)]).to(config.device)

            if_rank = True
            if if_rank:
                time_before_rank = tt.time()
                output_doctor_rank_neg = Doctor_Rank(
                neg_doctor_id[s:e], gender[s:e], age[s:e], height[s:e], weight[s:e], hospital_history[s:e],
                pregnancy_situation[s:e], duration_of_illness[s:e],
                wanted_help_i[s:e], wanted_help_m[s:e], chronic_disease_i[s:e], chronic_disease_m[s:e],
                surgery_history_i[s:e], surgery_history_m[s:e], chemotherapy_history_i[s:e], chemotherapy_history_m[s:e],
                disease_history_i[s:e], disease_history_m[s:e], medication_usage_i[s:e], medication_usage_m[s:e],
                disease_i[s:e], disease_m[s:e],
                allergy_history_i[s:e], allergy_history_m[s:e], major_illness_i[s:e], major_illness_m[s:e],
                neg_recvalue[s:e], neg_patient_num[s:e], neg_hospital[s:e], neg_faculty[s:e], neg_hornor[s:e],
                neg_gift_value[s:e], neg_zhuanyefangxiang[s:e], neg_good_review_rate[s:e], neg_title[s:e],
                neg_jiyu_i[s:e], neg_jiyu_m[s:e], neg_zhuanyeshanchang_i[s:e], neg_zhuanyeshanchang_m[s:e],
                neg_gerenjianjie_i[s:e], neg_gerenjianjie_m[s:e], neg_keyanchengguo_i[s:e], neg_keyanchengguo_m[s:e],
                neg_shehuirenzhi_i[s:e], neg_shehuirenzhi_m[s:e]
                )
                time_after_rank = tt.time()
            else:
                output_doctor_rank_neg = torch.tensor([0.0 for _ in range(e-s)]).to(config.device)

            if len(output_doctor_recall_neg.shape) == 0:
                output_doctor_recall_neg = output_doctor_recall_neg.unsqueeze(dim=-1)
            if len(output_doctor_rank_neg.shape) == 0:
                output_doctor_rank_neg = output_doctor_rank_neg.unsqueeze(dim=-1)
            recall_output = torch.cat([recall_output, output_doctor_recall_neg])
            rank_output = torch.cat([rank_output, output_doctor_rank_neg])
            # print(1)
            # print(recall_output)
            # print(rank_output)
            time_need[0].append(time_after_recall - time_before_recall)
            time_need[1].append(time_after_rank - time_before_rank)
            print(np.mean(time_need[0]), np.mean(time_need[1]))

        # make the prediction for target doctor = 0 to avoid disturbance
        target_doctor = int(p_doctor_id[0].item())
        target_disease_tag = disease_tag[0]
        if target_disease_tag in config.dict_disease_doctor:
            list_tag_doctor = config.dict_disease_doctor[target_disease_tag]
        else:
            list_tag_doctor = []

        # print(recall_output[:20])
        # print('neg_text_all_doctor_i_2')
        # print(neg_text_all_doctor_i[0][:10].cpu().numpy().tolist())
        # print(neg_text_all_doctor_i[1][:10].cpu().numpy().tolist())
        # print(neg_text_all_doctor_i[2][:10].cpu().numpy().tolist())
        # print(neg_text_all_doctor_i[3][:10].cpu().numpy().tolist())
        # print(neg_text_all_doctor_i[4][:10].cpu().numpy().tolist())
        _, prediction = torch.topk(recall_output, len(recall_output))
        prediction = prediction.cpu().numpy().tolist()
        pos_doctor_position = prediction.index(target_doctor)
        list_p_recall, list_dt_recall, list_p, list_r, list_h, list_m, list_n = [], [], [], [], [], [], []
        list_p1, list_r1, list_h1, list_m1, list_n1 = [], [], [], [], []

        ##### must contain k = 100 #####
        for k in [1, 5, 10, 20, 30, 40, 50, 100]:
            p, r, h, m, n = compute_precision_recall([target_doctor], prediction, k)
            in_current_department, in_current_tag = 0, 0
            for i in range(k):
                if prediction[i] < config.num_doctor_current:
                    in_current_department += 1
                if prediction[i] in list_tag_doctor:
                    in_current_tag += 1
            list_p_recall.append(in_current_department/k)
            list_dt_recall.append(in_current_tag/k)
            list_p.append(p)
            list_r.append(r)
            list_h.append(h)
            list_m.append(m)
            list_n.append(n)

            # extract candidate doctor list filtered by doctor recall model
            if k == 100:
                output_filtered = rank_output[prediction[:k]]
                # print('output_filtered')
                # print(output_filtered)
                _, prediction_rank = torch.topk(output_filtered, len(output_filtered))
                prediction_rank = prediction_rank.cpu().numpy().tolist()
                if pos_doctor_position >= k:
                    list_p1 += [0, 0, 0, 0]
                    list_r1 += [0, 0, 0, 0]
                    list_h1 += [0, 0, 0, 0]
                    list_m1 += [0, 0, 0, 0]
                    list_n1 += [0, 0, 0, 0]
                else:
                    for k in [1, 5, 10, 20]:
                        p, r, h, m, n = compute_precision_recall([pos_doctor_position], prediction_rank, k)
                        list_p1.append(p)
                        list_r1.append(r)
                        list_h1.append(h)
                        list_m1.append(m)
                        list_n1.append(n)
        P_RECALL.append(list_p_recall)
        DT_RECALL.append(list_dt_recall)
        PRECISION_RECALL.append(list_p)
        RECALL_RECALL.append(list_r)
        HR_RECALL.append(list_h)
        MRR_RECALL.append(list_m)
        NDCG_RECALL.append(list_n)

        PRECISION_RANK.append(list_p1)
        RECALL_RANK.append(list_r1)
        HR_RANK.append(list_h1)
        MRR_RANK.append(list_m1)
        NDCG_RANK.append(list_n1)

        # evaluate for inter recall only
        _, prediction = torch.topk(rank_output, len(rank_output))
        prediction = prediction.cpu().numpy().tolist()
        list_p, list_r, list_h, list_m, list_n = [], [], [], [], []
        for k in [1, 5, 10, 20]:
            p, r, h, m, n = compute_precision_recall([target_doctor], prediction, k)
            list_p.append(p)
            list_r.append(r)
            list_h.append(h)
            list_m.append(m)
            list_n.append(n)
        PRECISION_RANK_only.append(list_p)
        RECALL_RANK_only.append(list_r)
        HR_RANK_only.append(list_h)
        MRR_RANK_only.append(list_m)
        NDCG_RANK_only.append(list_n)
    return (np.mean(P_RECALL, axis=0),
            np.mean(DT_RECALL, axis=0),
            np.mean(PRECISION_RECALL, axis=0),
            np.mean(RECALL_RECALL, axis=0),
            np.mean(HR_RECALL, axis=0),
            np.mean(MRR_RECALL, axis=0),
            np.mean(NDCG_RECALL, axis=0),
            np.mean(PRECISION_RANK_only, axis=0),
            np.mean(RECALL_RANK_only, axis=0),
            np.mean(HR_RANK_only, axis=0),
            np.mean(MRR_RANK_only, axis=0),
            np.mean(NDCG_RANK_only, axis=0),
            np.mean(PRECISION_RANK, axis=0),
            np.mean(RECALL_RANK, axis=0),
            np.mean(HR_RANK, axis=0),
            np.mean(MRR_RANK, axis=0),
            np.mean(NDCG_RANK, axis=0))

if __name__ == "__main__":
    print('Note: You are running evaluate.py')
