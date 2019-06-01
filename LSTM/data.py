import json
import pandas as pd
from pprint import pprint
from pandas.io.json import json_normalize
import time, timeit

import os
os.getcwd()

print('read data')

with open('F:/QA_with_NN/here_is_your_answer/dev-v2.0.json') as f:
    data = json.load(f)

train = data

train_data_list = []
context = []
question = []
answer = []
answer_start = []
is_impossible = []
example = []

for i in range(len(train['data'])):
    for j in range(len(train['data'][i]['paragraphs'])):
        context.append(train['data'][i]['paragraphs'][j]['context'])
        for k in range(len(train['data'][i]['paragraphs'][j]['qas'])):
            question.append(train['data'][i]['paragraphs'][j]['qas'][k]['question'])
            is_impossible.append(train['data'][i]['paragraphs'][j]['qas'][k]['is_impossible'])
            for l in range(len(train['data'][i]['paragraphs'][j]['qas'][k]['answers'])):
                answer.append(train['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['text'])
                answer_start.append(train['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['answer_start'])
               

for i in range(len(train['data'])):
    for j in range(len(train['data'][i]['paragraphs'])):
        context_1 = train['data'][i]['paragraphs'][j]['context']
        for k in range(len(train['data'][i]['paragraphs'][j]['qas'])):
            question_1 = train['data'][i]['paragraphs'][j]['qas'][k]['question']
            is_impossible_1 = train['data'][i]['paragraphs'][j]['qas'][k]['is_impossible']
            if is_impossible_1 == False:
                for l in range(len(train['data'][i]['paragraphs'][j]['qas'][k]['answers'])):
                    answer_1 = train['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['text']
                    answer_start_1 = train['data'][i]['paragraphs'][j]['qas'][k]['answers'][l]['answer_start']
            else:  
                answer_1 = ""
                answer_start_1= ""
            example.append((context_1, question_1, is_impossible_1, answer_1, answer_start_1))


example[1]
#  context, question, is_impossible, answer, answer_start

test = []
for i in range(len(example)):
    test.append(example[i][2])

dataframe = pd.DataFrame(example)

len(dataframe)
# running everything on 4 lines
df = dataframe.head(4)

'''
class read_file():
    def __init__(self):
        self.filename = 'xyz'
        self.columns = ['context', 'ques', 'ans']
        self.seq_len = 50
        
    def get_data(self):
        return read_file(self.filename)

'''