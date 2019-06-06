# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:57:30 2019

@author: x219427
"""

import pandas as pd
import numpy as np
import json
from pprint import pprint
from pandas.io.json import json_normalize
import time, timeit
import os, sys
import keras
import tensorflow as tf

path = "C:\\Saurabh\\QA Systems"

with open(os.path.join(path, 'Data', 'train-v2.0.json')) as read_train:
    train = json.load(read_train)

with open(os.path.join(path, 'Data', 'dev-v2.0.json')) as read_test:
    test = json.load(read_test)


train_df_list = []
for i in range(len(train['data'])):
    
    for j in range(len(train['data'][i]['paragraphs'])):
        
        context = train['data'][i]['paragraphs'][j]['context']
        
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')
        context = context.lower()
        
        for k in range(len(train['data'][i]['paragraphs'][j]['qas'])):
            
            question = train['data'][i]['paragraphs'][j]['qas'][k]['question']
            question = question.lower()
            
            is_impossible = train['data'][i]['paragraphs'][j]['qas'][k]['is_impossible']
            
            if is_impossible == False:
                
                for l in range(len(train['data'][i]['paragraphs'][j]['qas'][k]['answers'])):
                    
                    answer = train['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['text']
                    answer = answer.lower()
                    
                    answer_start = train['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['answer_start']
                    answer_end = answer_start + len(answer)
            else:  
                
                answer = ""
                answer_start = ""
                answer_end = ""
            train_df_list.append((context, question, is_impossible, answer, answer_start, answer_end))
train_df = pd.DataFrame(train_df_list)


test_df_list = []
for i in range(len(test['data'])):
    
    for j in range(len(test['data'][i]['paragraphs'])):
        
        context = test['data'][i]['paragraphs'][j]['context']
        
        context = context.replace("''", '" ')
        context = context.replace("``", '" ')
        context = context.lower()
        
        for k in range(len(test['data'][i]['paragraphs'][j]['qas'])):
            
            question = test['data'][i]['paragraphs'][j]['qas'][k]['question']
            question = question.lower()
            
            is_impossible = test['data'][i]['paragraphs'][j]['qas'][k]['is_impossible']
            
            if is_impossible == False:
                
                for l in range(len(test['data'][i]['paragraphs'][j]['qas'][k]['answers'])):
                    
                    answer = test['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['text']
                    answer = answer.lower()
                    
                    answer_start = test['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['answer_start']
                    answer_end = answer_start + len(answer)
            else:  
                
                answer = ""
                answer_start = ""
                answer_end = ""
            test_df_list.append((context, question, is_impossible, answer, answer_start, answer_end))
test_df = pd.DataFrame(test_df_list)





         
