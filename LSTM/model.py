import numpy as np
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential

from keras.models import load_model


class Get_Model():

    def __init__(self):
        self.model = Sequential()
        self.compiled_model = None

    def load_model(self, filepath):
        self.model = load_model(filepath)

    def lstm_layer(self):
        self.model = self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))

    def dense_layer(self):
        self.model = self.model.add(Dense(neurons, activation=activation))

    def dropout(self):
        self.model = self.model.add(Dropout(dropout_rate))

    def add_layer(self, lstm, dense, dropout):
        if lstm == True:        
            self.lstm_layer()
        elif dense == True:
            self.dense_layer()
        elif dropout == True:
            self.dropout()
            
            
	def build_model(self, lstm_configs):
        for layer in lstm_configs.model_layers():
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None
            self.add_layer(layer['add']['lstm'], layer['add']['dense'], layer['add']['dropout'])

        self.model.compile(loss=lstm_configs.loss, optimizer=lstm_configs.optimizer)
        self.compiled_mode = self.model
