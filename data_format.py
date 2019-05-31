import pandas as pd
import numpy as np
import json
from pprint import pprint
from pandas.io.json import json_normalize
import time, timeit


print('read data')
with open('data.json') as f:
    data = json.load(f)

# print(len(data))
# print(type(data))

# print('what is happening??')
# df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')
# print(len(df))
# print(list(df))

# print('dig deeper')
# df = pd.DataFrame.from_dict(json_normalize(data['data']), orient='columns')
# print(len(df))
# print(list(df))
# print(type(data['data']))

# df = pd.DataFrame(data['data'])
# print(df.columns)
# print(df.head(10))
# print(df['title'])
# print(df.paragraphs.iloc[0])


#print(type(data['data'][0]))
#print(type(data['data'][1]))


# print('still not good')
# print(len(data['data']))

for _ in range(len(data['data'])):

    if _ == 0:
        # print(type(data['data'][_]))
        # print(data['data'][_]['title'])
        # print(type(data['data'][_]['paragraphs']))
        # print(len(data['data'][_]['paragraphs']))
        # for i in range(len(data['data'][_]['paragraphs'])):
        #     # print(type(data['data'][_]['paragraphs'][0]))
        #     # print(type(data['data'][_]['paragraphs'][0]['context']))
        #     # print(len(data['data'][_]['paragraphs'][i]['qas']))
        # print(type(data['data'][_]['paragraphs'][0]['qas']))
        # print('qas len')
        # print(len(data['data'][_]['paragraphs'][0]['qas'][0]))
        # print('qas type')
        # print(type(data['data'][_]['paragraphs'][0]['qas'][0]))
        # print(type(data['data'][_]['paragraphs'][0]['qas'][0]))
        # print('qas')
        # print(data['data'][_]['paragraphs'][0]['qas'][0])
        print(data['data'][_]['paragraphs'][0]['qas'][0]['question'])
        print(data['data'][_]['paragraphs'][0]['qas'][0]['id'])
        print(data['data'][_]['paragraphs'][0]['qas'][0]['answers'])
        print(data['data'][_]['paragraphs'][0]['qas'][0]['is_impossible'])
        print('type')
        print(type(data['data'][_]['paragraphs'][0]['qas'][0]['question']))
        print(type(data['data'][_]['paragraphs'][0]['qas'][0]['id']))
        print(type(data['data'][_]['paragraphs'][0]['qas'][0]['answers']))
        print(type(data['data'][_]['paragraphs'][0]['qas'][0]['is_impossible']))
        
        

print('understood?, haha')


with open("C:/Saurabh/QA Systems/Data/train-v2.0.json") as read_file:
    train = json.load(read_file)
    
#train['data'][0]['paragraphs'][1].keys()
#Out[43]: dict_keys(['qas', 'context'])
    
#type(train['data'][0]['paragraphs'][1]['qas'])
#Out[45]: list

#train['data'][0]['paragraphs'][1]['qas'][0]
#Out[44]: 
#{'question': 'After her second solo album, what other entertainment venture did Beyonce explore?',
# 'id': '56be86cf3aeaaa14008c9076',
# 'answers': [{'text': 'acting', 'answer_start': 207}],
# 'is_impossible': False}    

#train['data'][0]['paragraphs'][1]['context']
#Out[47]: 'Following the disbandment of Destiny\'s Child in June 2005, 

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



test = []
for i in range(len(example)):
    test.append(example[i][2])

dataframe = pd.DataFrame(example)


from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM, GlobalMaxPool1D, Bidirectional
from numpy import array
# define model
inputs1 = Input(shape=(3, 1))
lstm1 = LSTM(10, return_sequences=True)(inputs1)
globalpool = GlobalMaxPool1D()(lstm1)
model = Model(inputs=inputs1, outputs=globalpool)
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))



from keras.layers import Input
from keras.layers import LSTM
from numpy import array
# define model
inputs1 = Input(shape=(3, 1))
lstm1, state_h, state_c = LSTM(5, return_sequences=True, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# make and show prediction
print(model.predict(data))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# prepare sequence
length = 5
seq = np.array([i/float(length) for i in range(length)])
X = seq.reshape(len(seq), 1, 1)
y = seq.reshape(len(seq), 1)
# define LSTM configuration
n_neurons = length
n_batch = length
n_epoch = 1500
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
start = time.time()
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result:
	print('%.1f' % value)
end = time.time()
print(end-start)

This is calculated based on the number of inputs (1) and the number of outputs 
(5 for the 5 units in the hidden layer), as follows:
n = 4 * ((inputs + 1) * outputs + outputs^2)
n = 4 * ((1 + 1) * 5 + 5^2)
n = 4 * 35
n = 140

We can also see that the fully connected layer only has 6 parameters for the 
number of inputs (5 for the 5 inputs from the previous layer), 
number of outputs (1 for the 1 neuron in the layer), and the bias.

n = inputs * outputs + outputs
n = 5 * 1 + 1
n = 6



from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
# prepare sequence
length = 5
seq = np.array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 500
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1)))
model.add(Dense(length))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:]:
	print('%.1f' % value)


Many to Many LSTM For Sequence Prediction
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import LSTM
# prepare sequence
length = 5
seq = np.array([i/float(length) for i in range(length)])
X = seq.reshape(1, length, 1)
y = seq.reshape(1, length, 1)
# define LSTM configuration
n_neurons = length
n_batch = 1
n_epoch = 1500
# create LSTM
model = Sequential()
model.add(LSTM(n_neurons, input_shape=(length, 1), return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(loss='mean_squared_error', optimizer='adam')
print(model.summary())
start = time.time()
# train LSTM
model.fit(X, y, epochs=n_epoch, batch_size=n_batch, verbose=2)
# evaluate
result = model.predict(X, batch_size=n_batch, verbose=0)
for value in result[0,:,0]:
	print('%.1f' % value)
end = time.time()
print(end-start)













