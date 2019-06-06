#Pre-requisites 
    # Python 3+ and Anaconda Packages
    # Create a Folder called as QA_Model in C/Users/Xidfolder/
    #Download the required dev and training file from  https://rajpurkar.github.io/SQuAD-explorer/ and save them inside the created QA_Model Folder
    
#Purpose :- To process the JSON File and transform them into a PANDAS DATAFRAME
#Author :- Siddharth Mariappan (Aka:- Sid)
#Created Date :- 5/29/2019
#Last Modified Date :-
#Last Modified By :-
                                 
#Load the required Librarires
import json
import pandas as pd
import getpass
from pandas.io.json import json_normalize 

username = getpass.getuser()
directory = r'C:\Users\{}\QA_Model\\'.format(username)
#The file paths for JSON
dev_datapath = r'C:\Users\{}\QA_Model\dev-v2.0.json'.format(username)
train_datapath = r'C:\Users\{}\QA_Model\train-v2.0.json'.format(username)

write_df_excel = 1 #1 if needed, 0 if not needed
dev_req = 1
train_req = 0

#FUNCTION TO CONVERT JSON FILE INTO PANDAS DATAFRAME
def squad_json_pd_df(json_dict):
    mylistsize = len((list(json_normalize(json_dict,'data')['title'])))
    row = []
    for i in range(0,mylistsize):
        data = [c for c in json_dict['data']][i]
        df = pd.DataFrame()
        data_paragraphs = data['paragraphs']
        mytitle = data['title']
        for article_dict in data_paragraphs:
            for answers_dict in article_dict['qas']:
                Chk = 0
                
                for answer in answers_dict['answers']:
                    Chk = 1
                    row.append((
                                answers_dict['id'],
                                mytitle,
                                article_dict['context'], 
                                answers_dict['question'], 
                                answers_dict['is_impossible'],
                                answer['answer_start'],
                                answer['text']
                               ))
               
                if (Chk == 0) :
                    row.append((
                                answers_dict['id'],
                                mytitle,
                                article_dict['context'], 
                                answers_dict['question'], 
                                True,
                                '',
                                ''
                               ))                   
            
            df = pd.concat([df, pd.DataFrame.from_records(row, columns=['id', 'title','context', 'question','is_impossible', 'answer_start', 'answer'])], axis=0, ignore_index=True)
            df.drop_duplicates(inplace=True)
    return df

#Process the JSON file into PANDAS DATAFRAME
if (dev_req ==1):
    with open(dev_datapath) as file:
        dev_dict = json.load(file)
        dev_df = squad_json_pd_df(dev_dict)

#Process the JSON file into PANDAS DATAFRAME
if (train_req ==1):    
    with open(train_datapath) as file:
        train_dict = json.load(file)
        train_df = squad_json_pd_df(train_dict)

#WRITE THE PANDAS DATAFRAME INTO EXCEL FILE IF REQUIRED
if (write_df_excel == 1):
    OutputFileName = str('SQUAD_DATASET' + ' ' + str(pd.Timestamp.now())) [:-7] + '.xlsx'
    OutputFileName = OutputFileName.replace(':','-')
    OutputName = directory + OutputFileName
    writer = pd.ExcelWriter(OutputName)
    
    if (dev_req ==1):
        dev_df.to_excel(writer,'Dev data',index = False)
    if (train_req ==1):
        train_df.to_excel(writer,'Train data',index = False)
    writer.save()




