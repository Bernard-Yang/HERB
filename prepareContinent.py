import pandas as pd
from pprint import pprint
from tqdm.notebook import tqdm
import numpy as np

import torch
import os
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

import matplotlib.pyplot as plt

from collections import defaultdict
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data', type=str, required=True,
    #                     choices=['cp', 'ss'],
                        # help='Path to evaluation dataset.')
    # parser.add_argument('--output', type=str, required=True,
    #                     help='Path to result text file')
    parser.add_argument('--model', 
                        type=str, 
                        default='bert',
                        # required=True,
                       )
    parser.add_argument('--method', type=str, 
                        default = 'aul',
                        # required=True,
                        choices=['aula', 'aul', 'cps', 'sss'])
                        
    parser.add_argument('--ablation', type=bool, 
                        default = False)
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


#generate continent values for each LM
model_list = ['bert', 'roberta', 'albert', 'bart']
for mn in model_list:
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser()
    args, unknown = parser.parse_known_args()
    args.model = mn
    args.method = 'aul'
    args.ablation = True
    if args.ablation:
        adj_list = ['gawky', 'industrious', 'perceptive', 'visionary', 'imaginative',
       'shrewd', 'resourceful', 'textured', 'jaded', 'instinctive',
       'enquiring', 'diligent', 'methodology', 'ironic', 'storied',
       'inventive', 'canny', 'insightful', 'good', 'intelligent',
       'inventive', 'clumsy', 'superb', 'rational', 'smart', 'seductive',
       'curvaceous', 'wrinkling', 'geeky', 'scrawny', 'sensuous',
       'lovely', 'slimmer', 'eagle', 'basketball', 'trendy', 'slender',
       'nasty', 'skeletal', 'elongated', 'anemic', 'charming',
       'healthier', 'desirable', 'calories', 'weaker', 'thick', 'quite',
       'lovely', 'stronger', 'strong', 'stronger', 'optimistic',
       'predominant', 'powerful', 'commander', 'asserting', 'deafening',
       'daring', 'successor', 'victory', 'party', 'interaction',
       'winners', 'weaker', 'surrendered', 'hesitant', 'susceptible',
       'spiky', 'failed', 'timid', 'shaky', 'losers', 'sturdy',
       'truthful', 'loyalists', 'playful', 'perilous', 'buffoonish',
       'courageous', 'sort', 'hardworking', 'frugal', 'pessimistic',
       'intolerant', 'thoughtful', 'simple', 'self-important',
       'unassuming', 'courteous', 'monomaniacal', 'unyielding',
       'housewife', 'doctor', 'waitress', 'archivist', 'businesswoman',
       'manicurist', 'housekeeper', 'janitor', 'stylists', 'nanny',
       'virtuoso', 'captain', 'protégé', 'mathematician', 'skipper',
       'sculptor', 'billionaire', 'dragon', 'television', 'illusionist']
    tokenizer, model = load_tokenizer_and_model(args)
    ##
    model_name = args.model #'roberta'
    print('model_name', model_name)
    for num, continent in enumerate(conti_con_dict.keys()):
    # for num, continent in enumerate(['Europe']):
        torch.cuda.empty_cache()
        contry_num = len(conti_con_dict[continent])
        V_conti = np.zeros([contry_num, len(adj_list)])
        v_conti = np.zeros([contry_num, len(adj_list)])
        C_R_country = np.zeros([contry_num])

        for con_i in range(contry_num):
            torch.cuda.empty_cache()

            country = conti_con_dict[continent][con_i]
            print('processing:', country)
            #cities
            city_list = location_dict[country]
            score_matrix = np.zeros([len(city_list), 112])
            #load city value
            for city_num, city in enumerate(city_list):
                if '/' in city:
                    city = city.replace('/', '')
                score =  np.load('./results/city112d/' + mn + '/' + city + '.npy' )
                score_matrix[city_num] = score
            # print('score_matrix', score_matrix)
            # #cities
            demoninator = np.linalg.norm(score_matrix, ord=2, axis=1).reshape(-1,1)
            score_matrix = score_matrix / demoninator

            C_R = np.zeros([score_matrix.shape[0]])
            c_R = np.zeros([len(adj_list)])
            # print('city number', score_matrix.shape[0])

            if score_matrix.shape[0] == 1:
                vrj = cal_DVR(country, location_dict, adj_list, tokenizer, args, calculate_aul_batch, is_city=True)[0]
                vrj = vrj / np.linalg.norm(vrj, ord=2)

                V_rj = cal_DVR(country, location_dict, adj_list, tokenizer, args, calculate_aul_batch, is_city=False)
                V_rj = V_rj / np.linalg.norm(V_rj, ord=2)

                V_rj = V_rj + vrj
                c_R = 0 
                V_conti[con_i] = V_rj
                v_conti[con_i] = vrj
                C_R_country[con_i] = 0
            
            elif score_matrix.shape[0] == 0:
                V_rj = cal_DVR(country, location_dict, adj_list, tokenizer, args, calculate_aul_batch, is_city=False)
                V_rj = V_rj / np.linalg.norm(V_rj, ord=2)

                c_R = 0 
                V_conti[con_i] = V_rj
                v_conti[con_i] = V_rj
                C_R_country[con_i] = 0
            else:
                #city
                v_avg = np.sum(score_matrix, axis=0) / score_matrix.shape[0]

                # for i in range(score_matrix.shape[1]):
                #city wise
                for line in range(score_matrix.shape[0]-1):
                    cal = score_matrix[line, :] - score_matrix[line+1:, :]
                    cal *= cal
                    cal = np.sum(cal, axis=0) # (92,
                    cal_city = np.linalg.norm(score_matrix[line, :] - v_avg, ord=2)
                    C_R[line] = cal_city 
                c_R = cal

                # print('c_R', c_R)
                c_R = 2 * c_R / (score_matrix.shape[0] * (score_matrix.shape[0] - 1))
                e_C_R = np.zeros_like(c_R)
                for i in range(len(e_C_R)):
                    e_C_R[i] = np.exp(c_R[i]) / np.sum(np.exp(c_R))
                # e_C_R = np.log(e_C_R)
                # print('e_C_R', e_C_R)
                #V(rj)
                # v_avg = np.sum(score_matrix, axis=0) / score_matrix.shape[0]
                V_rj = e_C_R * v_avg 
                vrj = cal_DVR(country, location_dict, adj_list, tokenizer, args, calculate_aul_batch, is_city=False)
                vrj = vrj / np.linalg.norm(vrj, ord=2)
                
                V_rj += vrj
                # print('V_rj', V_rj)
                V_conti[con_i] = V_rj
                v_conti[con_i] = vrj

                softmax_d = 0.0
                for i in range(C_R.shape[0]-1):
                    # softmax_d += np.sum(np.exp(C_R[i] + C_R[i+1])) #  
                    for j in range(i+1, C_R.shape[0]):
                        softmax_d += np.sum(np.exp( (C_R[i] + C_R[j]) )) #  


                #loop cities
                wv = 0.0
                for i_c in range(score_matrix.shape[0]):
                    v1_city = score_matrix[i_c, :]
                    C_R1 = C_R[i_c]
                    for i_c_new in range(i_c+1, score_matrix.shape[0]):
                        C_R2 = C_R[i_c_new]
                        v2_city = score_matrix[i_c_new, :]
                        v = np.linalg.norm(v1_city - v2_city, ord=2)
                        w12 = np.exp((C_R1 + C_R2)  ) / softmax_d 
                        # w12 = 0.01
                        wv = wv + w12 * v 
                wv = 2 * wv / (score_matrix.shape[0] * (score_matrix.shape[0] - 1))
                # print('wv', wv)
                C_R_country[con_i] = wv
        #continent
        path = './results/' + model_name + '_adjSub/' if args.ablation else './results/' + model_name + '_adj/'
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(path + continent + model_name + 'Vrj.npy', V_conti)
        np.save(path + continent + model_name + 'vrj.npy', v_conti)
        np.save(path + continent + model_name + 'cR.npy', C_R_country)


