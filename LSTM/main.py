import data
lstm_data = data.read_file()
x_train, y_train = lstm_data.get_data()
x_test, y_test = lstm_data.get_data_test() # to be made/tested

import lstm
lstm_configs = lstm.configs()

from model import Get_Model
model = Get_Model()
model.build_model(lstm_configs)

from train import train_model
train = train_model(model)
model = train(x_train, y_train, model)

model.predict(x_test)

