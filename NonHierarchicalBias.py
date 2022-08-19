import pandas as pd
from pprint import pprint
from tqdm.notebook import tqdm
import numpy as np

import torch

import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import os
import matplotlib.pyplot as plt

from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', 
                        type=str, 
                        default='bert',
                        # required=True,
                       )
    parser.add_argument('--method', type=str, 
                        default = 'aul',
                        # required=True,
                        choices=['aula', 'aul', 'cps', 'sss'])
    args = parser.parse_args()

    return args

def load_tokenizer_and_model(args):
    
    '''
    Load tokenizer and model to evaluate.
    '''
    if args.model == 'bert':
        pretrained_weights = 'bert-base-cased'
    elif args.model == 'distilbert':
        pretrained_weights = 'distilbert-base-cased'
    elif args.model == "roberta":
        pretrained_weights = 'roberta-base'
    elif args.model == "albert":
        pretrained_weights = 'albert-base-v2'
    elif args.model == "deberta":
        pretrained_weights = 'microsoft/deberta-v3-small'
    elif args.model == "electra":
        pretrained_weights = 'google/electra-small-discriminator'
    elif args.model == "bart":
        pretrained_weights = 'facebook/bart-base'
    else:
        pretrained_weights = args.model
    model = AutoModelForMaskedLM.from_pretrained(pretrained_weights,
                                                 output_hidden_states=True,
                                                 output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights)

    model = model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    return tokenizer, model

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

log_softmax = torch.nn.LogSoftmax(dim=1)

def calculate_aul_batch(model, inputs, log_softmax, attention):
    '''
    Given token ids of a sequence, return the averaged log probability of
    unmasked sequence (AULA or AUL).
    '''
    output = model(**inputs)
    # logits = output.logits.squeeze(0)
    log_probs = torch.nn.functional.log_softmax(output['logits'],dim=2) # torch.Size([92, 11, 28996])
    token_ids = inputs['input_ids'].detach()
    # print(token_ids.shape)
    # token_log_probs = log_probs.gather(1, token_ids)[1:-1]
    token_log_probs = log_probs.gather(dim=2, index=token_ids.unsqueeze(2))[:,1:-1,:].squeeze(2) # torch.Size([92, 9])
    

    if attention:
        # TODO: optimization for batch 
        attentions = torch.mean(torch.cat(output.attentions, 0), 0)
        averaged_attentions = torch.mean(attentions, 0)
        averaged_token_attentions = torch.mean(averaged_attentions, 0)
        token_log_probs = token_log_probs.squeeze(1) * averaged_token_attentions[1:-1]
    
    
    sentence_log_prob = torch.mean(token_log_probs,dim=-1)
    score = sentence_log_prob.detach().cpu().numpy()

    # ranks = get_rank_for_gold_token(log_probs, token_ids)

    return score

def cal_DVR(country, location_dict, adj_list, tokenizer, args, calculate_aul_batch, is_city=True):
    
    if is_city:
        location_list = location_dict[country]
        score_matrix = np.zeros([len(location_list), len(adj_list)])
        # score_matrix = []
        for i in range(len(location_list)):
            sent_list = []
            for j in range(len(adj_list)):
                location = location_list[i]
                adj = adj_list[j]
                sentence = f"People in {location} are {adj}"
                sent_list.append(sentence)
            inputs = tokenizer(sent_list, return_tensors='pt', padding=True, truncation=True)
            attention = True if args.method == 'aula' else False
            score = calculate_aul_batch(model, inputs, log_softmax, attention)
            score_matrix[i] = score
        # score_matrix = np.stack(score_matrix, axis=0)
            

    else:
        score_matrix = np.zeros([len(adj_list)])
        sent_list = []
        for j in range(len(adj_list)):
            location = country
            adj = adj_list[j]
            sentence = f"People in {location} are {adj}"
            sent_list.append(sentence)
        inputs = tokenizer(sent_list, return_tensors='pt', padding=True, truncation=True)
        attention = True if args.method == 'aula' else False
        score = calculate_aul_batch(model, inputs, log_softmax, attention)
        score_matrix = score 
    return score_matrix

from collections import defaultdict
import geonamescache

gc = geonamescache.GeonamesCache()
# gets nested dictionary for countries
countries = gc.get_countries()
conti_con_dict =  defaultdict(list)
cities = gc.get_cities()
country_full_name_list = []
cnt = 0
cc1 = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA']
conti_con_dict = defaultdict(list)
for c in cc1:
    name = gc.get_continents()[c]['name']

    for k in countries:
        if countries[k]['continentcode'] == c:
            conti_con_dict[name].append(countries[k]['name'])
            country_full_name_list.append(countries[k]['name'])
    cnt += len(conti_con_dict[name])


con_codes = [countries[k]['iso'] for k in countries if countries[k]['name'] in country_full_name_list]


location_dict = {}
for coun in countries:
    location_dict[ countries[coun]['name'] ] = []
    for k in cities:
        if cities[k]['countrycode'] == coun:
            # print(cities[k]['name'])
            location_dict[countries[coun]['name'] ].append(cities[k]['name'])

word_str = "precocious, resourceful, inquisitive, genius, inventive, astute, adaptable, reflective, discerning, intuitive, inquiring, judicious, analytical, apt, venerable, imaginative, shrewd, thoughtful,\
     wise, smart, ingenious, clever, brilliant, logical, intelligent, \
        alluring, voluptuous, blushing, homely, plump, sensual, gorgeous, slim, bald, athletic, fashionable, stout, ugly, muscular, slender, \
    feeble, handsome, healthy, attractive, fat, weak, thin, pretty, beautiful, strong,\
         powerful, strong, confident, dominant, potent, command, assert, loud, bold, succeed, triumph, leader, dynamic, winner,\
    weak, surrender, timid, vulnerable, wispy, failure, shy, fragile, loser, \
    upright, honest, loyal, gentle, treacherous, clownish, brave, kind, hard-working, thrifty, optimistic, tolerant, earnest, straightforward, \
    narrow-minded, humble, punctual, single-minded, uncompromising, \
    Homemaker, nurse, receptionist, librarian, socialite, hairdresser, \
    nanny, bookkeeper, stylist, housekeeper, maestro, skipper, protege, \
    philosopher, captain, architect, financier, warrior, broadcaster, magician"
adj_list = word_str.split(', ')

adj_num = len(adj_list)
model_list = ['bert', 'albert', 'roberta', 'bart']
for mn in model_list:
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.model = mn
    args.method = 'aul'
    tokenizer, model = load_tokenizer_and_model(args)
    ##
    model_name = args.model #'roberta'
    print('model_name', model_name)
    for num, continent in enumerate(conti_con_dict.keys()):
    # for num, continent in enumerate(['Europe']):
        torch.cuda.empty_cache()
        contry_num = len(conti_con_dict[continent])
        # V_conti = np.zeros([contry_num, len(adj_list)])
        v_conti = np.zeros([contry_num, len(adj_list)])
        C_R_country = np.zeros([contry_num])

        for con_i in range(contry_num):
            torch.cuda.empty_cache()

            country = conti_con_dict[continent][con_i]
            # print('processing:', country)
            #cities
            city_list = location_dict[country]

            score_matrix = np.zeros([len(city_list), adj_num])

            for city_num, city in enumerate(city_list):
                if '/' in city:
                    city = city.replace('/', '')
                score =  np.load('./results/city112d/' + mn + '/' + city + '.npy' )
                score_matrix[city_num] = score
            # print('score_matrix', score_matrix)
            # #cities
            demoninator = np.linalg.norm(score_matrix, ord=2, axis=1).reshape(-1,1)
            score_matrix = score_matrix / demoninator

            # print('city number', score_matrix.shape[0])

            if score_matrix.shape[0] == 1:
         
                V_rj = cal_DVR(country, location_dict, adj_list, tokenizer, args, calculate_aul_batch, is_city=False)
                V_rj = V_rj / np.linalg.norm(V_rj, ord=2)

                c_R = 0 
                v_conti[con_i] = V_rj
                C_R_country[con_i] = 0
            
            elif score_matrix.shape[0] == 0:
                V_rj = cal_DVR(country, location_dict, adj_list, tokenizer, args, calculate_aul_batch, is_city=False)
                V_rj = V_rj / np.linalg.norm(V_rj, ord=2)

                c_R = 0 
                v_conti[con_i] = V_rj
                C_R_country[con_i] = 0
            else:

                vrj = cal_DVR(country, location_dict, adj_list, tokenizer, args, calculate_aul_batch, is_city=False)
                vrj = vrj / np.linalg.norm(vrj, ord=2)
                count = 0
                sum_c = 0
                for i in range(score_matrix.shape[0]-1):
                    for j in range(i+1, score_matrix.shape[0]):
                        sum_c += np.linalg.norm(vrj - score_matrix[j], ord=2)
                        count += 1
                
                C_R_country[con_i] = sum_c * 2 / count * (count-1)
        #continent

        if not os.path.exists('./results/' + model_name + '_adj/'):
            os.makedirs('./results/' + model_name + '_adj/')
        np.save('./results/' + model_name + '_adj/' + continent + model_name + 'c_plain.npy', C_R_country)
    torch.cuda.empty_cache()
    pre_path = './results/' + args.model +'_adj/'
    # V_afr = np.load(pre_path + 'Africa'+ model_name + 'Vrj.npy')
    v_afr = np.load(pre_path + 'Africa'+ model_name + 'vrj.npy')
    C_afr = np.load(pre_path + 'Africa'+ model_name + 'c_plain.npy')

    # V_asi = np.load(pre_path + 'Asia'+ model_name + 'Vrj.npy')
    v_asi = np.load(pre_path + 'Asia'+ model_name + 'vrj.npy')
    C_asi = np.load(pre_path + 'Asia'+ model_name + 'c_plain.npy')

    # V_eur = np.load(pre_path + 'Europe'+ model_name + 'Vrj.npy')
    v_eur = np.load(pre_path + 'Europe'+ model_name + 'vrj.npy')
    C_eur = np.load(pre_path + 'Europe'+ model_name + 'c_plain.npy')

    # V_na = np.load(pre_path + 'North America'+ model_name + 'Vrj.npy')
    v_na = np.load(pre_path + 'North America'+ model_name + 'vrj.npy')
    C_na = np.load(pre_path + 'North America'+ model_name + 'c_plain.npy')

    # V_oce = np.load(pre_path + 'Oceania'+ model_name + 'Vrj.npy')
    v_oce = np.load(pre_path + 'Oceania'+ model_name + 'vrj.npy')
    C_oce = np.load(pre_path + 'Oceania'+ model_name + 'c_plain.npy')

    # V_sa = np.load(pre_path + 'South America'+ model_name + 'Vrj.npy')
    v_sa = np.load(pre_path + 'South America'+ model_name + 'vrj.npy')
    C_sa = np.load(pre_path + 'South America'+ model_name + 'c_plain.npy')
    V_list = [v_afr, v_asi, v_eur, v_na, v_oce, v_sa]
    C_list = [C_afr, C_asi, C_eur, C_na, C_oce, C_sa]
    continent = ['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America']

    cont_C = np.zeros([6])
    cont_V = np.zeros([6, len(adj_list)])
    
    V_continet = [] # np.zeros([0, len(adj_list)])
    for num, (V,C) in enumerate(zip(V_list, C_list)):

        # continent v
        vrj_conti = cal_DVR(continent[num], location_dict, adj_list, tokenizer, args, calculate_aul_batch, is_city=False)
        vrj_conti = vrj_conti / np.linalg.norm(vrj_conti, ord=2)
        
        #country
        demoninator = np.linalg.norm(V, ord=2, axis=1).reshape(-1,1)
        V = V / demoninator
        conti = continent[num] #africa
        country_list = conti_con_dict[conti]
        for country in country_list:
            # print('country', country)#congo
            #city 
            city_list = location_dict[country] #['sd, 12]
            score_matrix = np.zeros([len(city_list), adj_num])
            for city_num, city in enumerate(city_list):
                if '/' in city:
                    city = city.replace('/', '')
                score =  np.load('./results/city112d/' + mn + '/' + city + '.npy' )
                score_matrix[city_num] = score

            demoninator = np.linalg.norm(score_matrix, ord=2, axis=1).reshape(-1,1)
            score_matrix = score_matrix / demoninator
            V = np.concatenate([V, score_matrix], axis=0)
        # vrj_conti = vrj_conti
        V = np.concatenate([V, vrj_conti.reshape(1, -1)], axis=0)
        
        print(V.shape)

        count = 0.0
        # sum_c = 0.0
        all_dist = []
        for i in range(V.shape[0]-1):
            for j in range(i+1, V.shape[0]):
                # sum_c += np.linalg.norm(V[i] - V[j], ord=2)
                all_dist.append(np.linalg.norm(V[i] - V[j], ord=2))
                count += 1
        
        # C_R_country[con_i] = sum_c * 2 / count * (count-1)
       
        # C_R_country[con_i] = wv_conti

        # cont_C[num] = sum_c * 2 / (count * (count-1))
        cont_C[num] = np.mean(all_dist)
        print(cont_C[num])
        cont_V[num] = vrj_conti
        # V_continet = np.concatenate([V_continet, V], axis=0)
        V_continet.append(V)

    V_continet  =  np.concatenate(V_continet, axis =0 )
    print(V_continet.shape)

    #overall
    C = cont_C
    V = V_continet #continent v
   
    demoninator = np.linalg.norm(V, ord=2, axis=1).reshape(-1,1)
    V = V / demoninator

    print(V.shape)

    count = 0
    # sum_c = 0
    all_dist = []
    for i in range(V.shape[0]-1):
        for j in range(i+1, V.shape[0]):
            # sum_c += np.linalg.norm(V[i] - V[j], ord=2)
            all_dist.append(np.linalg.norm(V[i] - V[j], ord=2))
            count += 1


    print('model',mn)
    for i in cont_C:
        print(i)
    # print(sum_c * 2 / (count * (count - 1)))
    print(np.mean(all_dist))