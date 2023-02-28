import pickle
import time

import numpy as np
from chatgpt_wrapper import ChatGPT
import sys
import csv
import copy
maxInt = sys.maxsize
MAXLEN = 2500
from constant import MIMIC_3_DIR
while True:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

import re
import pandas as pd
from datasets_a import load_full_codes
ind2c, desc = load_full_codes('{}/train_full.csv'.format(MIMIC_3_DIR))
c2ind = {}
for key in ind2c.keys():
    c2ind[ind2c[key]] = key
c2ind_pa = {}
for key in c2ind.keys():
    key = re.sub('(\.[0-9]+)', '', key)
    c2ind_pa[key] = len(c2ind_pa)
#c2ind = c2ind_pa
import json
from transformers import LongformerTokenizer, AutoModelForCTC, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("whaleloops/KEPTlongformer-PMM3")
test_csv = json.load(open('./preprocess/mimic3-50_train.json', 'r'))
yhats = []
ys = []
bot = ChatGPT()
raw_answers = {}
import os
if os.path.exists('chatgpt_weak_anno-50.pkl'):
    raw_answers = pickle.load(open('chatgpt_weak_anno-50.pkl', 'rb'))
from evaluation import all_macro, all_micro
from data_mimic import proc_text, get_headersandindex, get_subnote
for item_id, items in enumerate(test_csv):
    id = items['subject_id']+'_'+items['hadm_id']
    if id not in raw_answers:
        responses = []
        bot.new_conversation()
        yhat = np.zeros(len(c2ind), dtype=int)
        y = np.zeros(len(c2ind), dtype=int)
        text = items['TEXT']
        labels = items['LABELS'].split(';')
        text = re.sub(r'\[\*\*[^\]]*\*\*\]', '',
                      text)  # remove any mimic special token like [**2120-2-28**] or [**Hospital1 3278**]
        tmp = tokenizer.tokenize(proc_text(text))
        if len(tmp) <= MAXLEN:
            text = tokenizer.convert_tokens_to_string(tmp)
        else:
            headers_pos = get_headersandindex(text)
            if len(headers_pos) > 1:
                new_text = get_subnote(text, headers_pos)
                text = new_text
                text = proc_text(text)
            text = tokenizer.convert_tokens_to_string(tmp[0:MAXLEN])

            # else:
            #     to_sav.append((str(self.df[index]['LABELS']),text))
        templete = ''
        for lid, label in enumerate(labels):
            templete += '{}. ICD-9 CM code: {}, {}\nEvidence: [\"text from the target clinical note\"]\n'.format(lid+1, label, desc[label])
        for label in labels:
            y[c2ind[label]] = 1
        prompt = "Please fill the evidence part of the target ICD-9 CM code list from the target clinical note.\n" \
                 "Guidelines:\n" \
                 "1. Cite the evidence from the given clinical note using \"\" and do not rewrite or omit it!\n" \
                 "2. Avoid using ... !\n" \
                 "3. Treat the given ICD-9 CM code list as ground truth.\n" \
                 "4. Using <No Evidence> if cannot find any evidence from the given note.\n" \
                 "5. Do not make up any evidence.\n" \
                 "Target Clinical Note:\n" + text + '\n' \
                 "Target ICD-9 CM Code List:\n" + templete
        len_total = len(tokenizer.tokenize(prompt))
        if len_total > 3200:
            to_cut = len_total-3200
            text = tokenizer.convert_tokens_to_string(tmp[0:len(tmp)-to_cut])
            prompt = "Please fill the evidence part of the target ICD-9 CM code list from the target clinical note.\n" \
                     "Guidelines:\n" \
                     "1. Cite the evidence from the given clinical note using \"\" and do not rewrite or omit it!\n" \
                     "2. Avoid using ... !\n" \
                     "3. Treat the given ICD-9 CM code list as ground truth.\n" \
                     "4. Using <No Evidence> if cannot find any evidence from the given note.\n" \
                     "5. Do not make up any evidence.\n" \
                     "Target Clinical Note:\n" + text + '\n' \
                                                        "Target ICD-9 CM Code List:\n" + templete
        print(prompt)
        flag = True
        response = bot.ask(prompt)
        responses.append(response)
        #prompt_recall = 'Try again and assign as many as ICD-9 codes to the provided clinical note to improve the recall performance.'
        #response_recall = bot.ask(prompt_recall)
        marks = [x[0] for x in re.findall(" (((V[0-9]{2})|([E]*[0-9]{2,3}))(\.[0-9x]+){0,1})", response)]
        while "Unusable response produced" in response:
            print("<response>: {}".format(response))
            time.sleep(65*60)
            bot.refresh_session()
            response = bot.ask(prompt)

        miss_codes = list(set(labels).difference(set(marks)))
        for mark in marks:
            if mark in c2ind:
                yhat[c2ind[mark]] = 1
        yhats.append(yhat)
        ys.append(y)
        tp = np.sum(yhat&y)
        print("<response>: {}".format(response))
        print(marks)
        print(labels)
        recall = tp/(np.sum(y)+1e-4)
        precision = tp/(np.sum(yhat)+1e-4)
        print("recall: {}".format(tp/np.sum(y)))
        print("precision: {}".format(tp / np.sum(yhat)))
        print("f1: {}".format(2*recall*precision/(recall+precision)))
        raw_answers[id] = responses
        pickle.dump(raw_answers, open('chatgpt_weak_anno-50.pkl', 'wb'))
        time.sleep(20)
    else:
        continue
ys = np.stack(ys, axis=0)
yhats = np.stack(yhats, axis=0)
print("[MACRO] accuracy, precision, recall, f-measure")
print(all_macro(yhats, ys))
ymics = ys.ravel()
yhatmics = yhats.ravel()
print("[MiCRO] accuracy, precision, recall, f-measure")
print(all_micro(yhatmics, ymics))