import torch
import torch.nn as nn
import hashlib
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import transformers

import config
# Random parameter settings
torch.manual_seed(2023)
torch.cuda.manual_seed(2023)

def tensor_hash(tensor):
    tensor = tensor.cpu().reshape(-1)
    tensor_bytes = tensor.numpy().tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

class Text_Encoder(nn.Module):
    def __init__(self, config: config.Config):
        super(Text_Encoder, self).__init__()
        self.config = config
        self.encoder = transformers.AutoModel.from_pretrained(config.model_for_step_1)
        self.fc_trans_size_1 = nn.Linear(self.encoder.config.hidden_size, 256)
        self.fc_trans_size_2 = nn.Linear(256, self.config.embed_size)
        self.fc_trans_size_3 = nn.Linear(self.encoder.config.hidden_size, 256)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.type = ''

    def mean_with_mask(self, last_hidden_state, mask):
        output = []
        for i in range(len(last_hidden_state)):
            len_valid = torch.sum(mask[i])
            line_now = last_hidden_state[i][:len_valid]
            mean_now = line_now.mean(dim=0).unsqueeze(0)
            output.append(mean_now)
        output = torch.cat(output, dim=0)
        return output

    def forward(self, token, mask):
        with torch.no_grad():
            # output = self.encoder(input_ids=token, attention_mask=mask).pooler_output
            output = self.encoder(input_ids=token, attention_mask=mask).last_hidden_state
            # output = self.mean_with_mask(output, mask)
            # output = output.mean(dim=1)
            output = output[:, 0]
        output = self.relu(self.fc_trans_size_1(output))
        output = self.relu(self.fc_trans_size_2(output))
        return output


class Doctor_Recall(nn.Module):
    def __init__(self, config: config.Config, is_training, Encoder):
        super(Doctor_Recall, self).__init__()
        self.config = config
        self.is_training = is_training
        self.Encoder = Encoder

        self.similarity_func = nn.CosineSimilarity()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # self.sim_func = 'max'
        # self.sim_func = 'mean'

        # hash memory vector for speed up evaliuation
        self.hidden_state_memory = {}

    def reset_for_evaluate(self):
        self.hidden_state_memory = {}

    def encoder_with_memory(self, i, m):
        if self.is_training:
            return self.Encoder(i, m)
        else:
            return self.Encoder(i, m)

            hash_i = tensor_hash(i)
            if hash_i not in self.hidden_state_memory:
                self.hidden_state_memory[hash_i] = self.Encoder(i, m)
            return self.hidden_state_memory[hash_i]

    def forward(self, text_all_patient_i, text_all_patient_m, text_all_doctor_i, text_all_doctor_m):
        self.Encoder.type = 'recall'
        embed_text_all_patient = self.encoder_with_memory(text_all_patient_i, text_all_patient_m)
        embed_text_all_doctor = self.encoder_with_memory(text_all_doctor_i, text_all_doctor_m)

        sim = self.similarity_func(embed_text_all_patient, embed_text_all_doctor)
        return sim.squeeze()

    def forward_old(self, text_all_i, text_all_m, jiyu_i, jiyu_m,
                zhuanyeshanchang_i, zhuanyeshanchang_m, gerenjianjie_i, gerenjianjie_m,
                keyanchengguo_i, keyanchengguo_m, shehuirenzhi_i, shehuirenzhi_m):
        self.Encoder.type = 'recall'
        embed_text_all = self.encoder_with_memory(text_all_i, text_all_m)
        # embed_jiyu = self.encoder_with_memory(jiyu_i, jiyu_m)
        embed_zhuanyeshanchang = self.encoder_with_memory(zhuanyeshanchang_i, zhuanyeshanchang_m)
        # embed_gerenjianjie = self.encoder_with_memory(gerenjianjie_i, gerenjianjie_m)
        # embed_keyanchengguo = self.encoder_with_memory(keyanchengguo_i, keyanchengguo_m)
        # embed_shehuirenzhi = self.encoder_with_memory(shehuirenzhi_i, shehuirenzhi_m)

        # embed_doctor_all = [embed_jiyu, embed_zhuanyeshanchang, embed_gerenjianjie,
        #                     embed_keyanchengguo, embed_shehuirenzhi]
        embed_doctor_all = [embed_zhuanyeshanchang]

        all_sim = []
        for embed_doctor in embed_doctor_all:
            all_sim.append(self.similarity_func(embed_text_all, embed_doctor))
        all_sim = torch.stack(all_sim, dim=1)
        # print('all_sim1')
        # print(all_sim.cpu().numpy().tolist())

        if self.sim_func =='max':
            all_sim = torch.max(all_sim, dim=1).values
        elif self.sim_func == 'mean':
            all_sim = torch.mean(all_sim, dim=1)
        # print('embed_text_all')
        # print(embed_text_all[0].cpu().numpy().tolist())
        # print(embed_zhuanyeshanchang[0].cpu().numpy().tolist())
        # print('all_sim2')
        # print(all_sim.cpu().numpy().tolist())
        return all_sim.squeeze()

class Doctor_Rank(nn.Module):
    def __init__(self, config: config.Config, is_training, Encoder):
        super(Doctor_Rank, self).__init__()
        self.config = config
        self.is_training = is_training
        self.Encoder = Encoder
        embed_size = self.config.embed_size

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.map_method = 'sim'
        # self.map_method = 'mlp'

        # inter info embed
        self.embed_inter_num_info = nn.Linear(6, embed_size)
        self.embed_hospital = nn.Embedding(self.config.num_hospital, embed_size, padding_idx=0)

        # doctor info embed
        self.embed_doctor_num_info = nn.Linear(5, embed_size)
        self.embed_doctor_id = nn.Embedding(self.config.num_doctor_all, embed_size)
        self.embed_title = nn.Embedding(17, embed_size, padding_idx=0)
        self.embed_zhuanyefangxiang = nn.Embedding(self.config.num_zhuanyefangxiang, embed_size, padding_idx=0)
        self.embed_faculty = nn.Embedding(self.config.num_faculty, embed_size, padding_idx=0)

        # inter process layers
        self.patient_history_fuse = nn.Linear(embed_size*6, embed_size)

        # doctor process layers
        self.doctor_expertise_fuse = nn.Linear(embed_size*3, embed_size)
        self.doctor_quality_fuse = nn.Linear(embed_size*2, 1)

        # attention layer
        self.attention_layer = nn.Linear(11, 1, bias=False)
        self.attention_layer.weight.data = torch.tensor([1/11 for _ in range(11)])
        # map layer
        self.similarity_func = nn.CosineSimilarity()
        self.fc_map = nn.Linear(2*embed_size, 1)

        # hash memory vector for speed up evaliuation
        self.hidden_state_memory = {}


    def reset_for_evaluate(self):
        self.hidden_state_memory = {}

    def encoder_with_memory(self, i, m):
        if self.is_training:
            return self.Encoder(i, m)
        else:
            return self.Encoder(i, m)

            hash_i = tensor_hash(i)
            if hash_i not in self.hidden_state_memory:
                self.hidden_state_memory[hash_i] = self.Encoder(i, m)
            return self.hidden_state_memory[hash_i]

    def mapping(self, t1, t2):
        if self.map_method == 'mlp':
            output = torch.cat([t1, t2], dim=1)
            output = self.fc_map(output)
            output = self.relu(output)
            return output
        elif self.map_method == 'sim':
            return self.similarity_func(t1, t2).unsqueeze(dim=1)

    def forward(self, doctor_id, gender, age, height, weight,
                hospital_history, pregnancy_situation, duration_of_illness,
                wanted_help_i, wanted_help_m, chronic_disease_i, chronic_disease_m,
                surgery_history_i, surgery_history_m, chemotherapy_history_i, chemotherapy_history_m,
                disease_history_i, disease_history_m, medication_usage_i, medication_usage_m,
                disease_i, disease_m, allergy_history_i, allergy_history_m, major_illness_i, major_illness_m,

                recvalue, patient_num, hospital, faculty,
                hornor, gift_value, zhuanyefangxiang, good_review_rate, title,
                jiyu_i, jiyu_m, zhuanyeshanchang_i, zhuanyeshanchang_m, gerenjianjie_i, gerenjianjie_m,
                keyanchengguo_i, keyanchengguo_m, shehuirenzhi_i, shehuirenzhi_m):
        self.Encoder.type = 'rank'

        ############################## embed inter infor ##############################
        age = age.unsqueeze(1)
        gender = gender.unsqueeze(1)
        height = height.unsqueeze(1)
        weight = weight.unsqueeze(1)
        pregnancy_situation = pregnancy_situation.unsqueeze(1)
        duration_of_illness = duration_of_illness.unsqueeze(1)
        inter_num_info = torch.cat([age, gender, height, weight, pregnancy_situation, duration_of_illness],
                                   dim=1).to(torch.float32)

        # embed_age = self.embed_age(age)
        # embed_gender = self.embed_gender(gender)
        # embed_pregnancy_situation = self.embed_pregnancy_situation(pregnancy_situation)
        embed_hospital_history = self.embed_hospital(hospital_history)
        embed_hospital_history = torch.mean(embed_hospital_history, dim=1)

        embed_wanted_help = self.encoder_with_memory(wanted_help_i, wanted_help_m)
        embed_chronic_disease = self.encoder_with_memory(chronic_disease_i, chronic_disease_m)
        embed_surgery_history = self.encoder_with_memory(surgery_history_i, surgery_history_m)
        embed_chemotherapy_history = self.encoder_with_memory(chemotherapy_history_i, chemotherapy_history_m)
        embed_disease_history = self.encoder_with_memory(disease_history_i, disease_history_m)
        embed_medication_usage = self.encoder_with_memory(medication_usage_i, medication_usage_m)
        embed_disease = self.encoder_with_memory(disease_i, disease_m)
        embed_allergy_history = self.encoder_with_memory(allergy_history_i, allergy_history_m)
        embed_major_illness = self.encoder_with_memory(major_illness_i, major_illness_m)


        ############################## embed doctor info ##############################
        recvalue = recvalue.unsqueeze(1)
        patient_num = patient_num.unsqueeze(1)
        hornor = hornor.unsqueeze(1)
        gift_value = gift_value.unsqueeze(1)
        good_review_rate = good_review_rate.unsqueeze(1)
        doctor_num_info = torch.cat([recvalue, patient_num, hornor, gift_value, good_review_rate], dim=1).to(torch.float32)
        embed_doctor_num_info = self.embed_doctor_num_info(doctor_num_info)

        embed_doctor_id = self.embed_doctor_id(doctor_id)
        embed_title = self.embed_title(title)
        embed_title = torch.mean(embed_title, dim=1)
        embed_zhuanyefangxiang = self.embed_zhuanyefangxiang(zhuanyefangxiang)
        embed_faculty = self.embed_faculty(faculty)
        embed_hospital = self.embed_hospital(hospital)

        # embed_jiyu = self.encoder_with_memory(jiyu_i, jiyu_m)
        embed_zhuanyeshanchang = self.encoder_with_memory(zhuanyeshanchang_i, zhuanyeshanchang_m)
        # embed_gerenjianjie = self.encoder_with_memory(gerenjianjie_i, gerenjianjie_m)
        # embed_keyanchengguo = self.encoder_with_memory(keyanchengguo_i, keyanchengguo_m)
        # embed_shehuirenzhi = self.encoder_with_memory(shehuirenzhi_i, shehuirenzhi_m)

        ############################## process inter info ##############################
        inter_num_info = self.embed_inter_num_info(inter_num_info)

        inter_history_info = torch.cat([embed_chronic_disease, embed_surgery_history, embed_chemotherapy_history,
                                embed_disease_history, embed_allergy_history, embed_major_illness], dim=1).to(torch.float32)
        # print(inter_history_info.shape)
        # exit()
        inter_history_info = self.patient_history_fuse(inter_history_info)

        inter_current_info = [embed_wanted_help, embed_medication_usage, embed_disease]

        ############################## process doctor info ##############################
        doctor_expertise_info = torch.cat([embed_zhuanyeshanchang, embed_faculty, embed_zhuanyefangxiang], dim=1)
        doctor_expertise_info = self.doctor_expertise_fuse(doctor_expertise_info)

        doctor_quality_info = torch.cat([embed_doctor_num_info, embed_title], dim=1)
        doctor_quality_info = self.doctor_quality_fuse(doctor_quality_info)
        doctor_quality_info = self.tanh(doctor_quality_info)

        ############################## process all embeddings ##############################
        embed_doctor_id = self.relu(embed_doctor_id)
        inter_num_info = self.relu(inter_num_info)
        inter_history_info = self.relu(inter_history_info)
        # embed_hospital = self.relu(embed_hospital)
        # embed_hospital_history = self.relu(embed_hospital_history)
        doctor_expertise_info = self.relu(doctor_expertise_info)
        inter_current_info[0] = self.relu(inter_current_info[0])
        inter_current_info[1] = self.relu(inter_current_info[1])
        inter_current_info[2] = self.relu(inter_current_info[2])

        ############################## mapping ##############################
        mappings = torch.cat([
            self.mapping(embed_doctor_id, inter_num_info),
            self.mapping(embed_doctor_id, inter_history_info),
            self.mapping(embed_doctor_id, inter_current_info[0]),
            self.mapping(embed_doctor_id, inter_current_info[1]),
            self.mapping(embed_doctor_id, inter_current_info[2]),

            self.mapping(doctor_expertise_info, inter_history_info),
            self.mapping(doctor_expertise_info, inter_current_info[0]),
            self.mapping(doctor_expertise_info, inter_current_info[1]),
            self.mapping(doctor_expertise_info, inter_current_info[2]),

            self.mapping(embed_hospital, embed_hospital_history),
            doctor_quality_info
        ], dim=1)

        # if not self.is_training:
        #     print('self.attention_layer.weight.data: {}'
        #           .format(self.attention_layer.weight.data/torch.sum(self.attention_layer.weight.data)))

        output = self.attention_layer(mappings)/torch.sum(self.attention_layer.weight.data)
        if_print_attention = False
        if if_print_attention:
            print('attention: {}'.format(self.attention_layer.weight.data/torch.sum(self.attention_layer.weight.data)))
        return output



if __name__ == "__main__":
    # ['albert-base-v2'
    # 'bert-base-uncased'
    # 'distilbert-base-uncased'
    # 'google/mobilebert-uncased'
    # 'huawei-noah/TinyBERT_General_4L_312D'
    # 'google/electra-small-discriminator'
    print(f'model hidden_size: {transformers.AutoModel.from_pretrained("albert-base-v2").config.hidden_size}')
    model = transformers.AutoModel.from_pretrained('bert-base-uncased')
    print(f'model hidden_size: {model.config.hidden_size}')
    model = transformers.AutoModel.from_pretrained('distilbert-base-uncased')
    print(f'model hidden_size: {model.config.hidden_size}')
    model = transformers.AutoModel.from_pretrained('google/mobilebert-uncased')
    print(f'model hidden_size: {model.config.hidden_size}')
    model = transformers.AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    print(f'model hidden_size: {model.config.hidden_size}')
    model = transformers.AutoModel.from_pretrained('google/electra-small-discriminator')
    print(f'model hidden_size: {model.config.hidden_size}')

    print('Note: You are running model.py')
    # model = transformers.AutoModel.from_pretrained('bert-base-uncased')
    # tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
    #
    # s1 = 'this is a input'
    # s2 = 'this is another input of x'
    # ids = tokenizer.batch_encode_plus([s1, s2], padding=True)
    # print(ids)
    # rep = model(input_ids=torch.tensor(ids['input_ids']),
    #             attention_mask=torch.tensor(ids['attention_mask']))
    # print(rep)



