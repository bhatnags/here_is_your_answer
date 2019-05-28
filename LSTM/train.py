from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import Get_Model

class train_model():
    def __init__(self, model):
        self.epochs = 1
        self.batch_size = 32
        self.monitor='val_loss'
        self.model = model
        
    ''' ToDo: model_save_file'''
    def callbacks(self):
        return [
                EarlyStopping(monitor = self.monitor, mode='min', verbose=1),
                ModelCheckpoint(filepath=model_save_file, monitor=monitor='val_loss', save_best_only=True)
                ]
        

    def train(self, x, y, self.model, self.epochs, self.batch_size):
        model_save_file = 'xyz_model_save_file'
        self.model.fit(x, y, epochs=self.epochs, batch_size=self.batch_size, callbacks=self.callbacks)
        self.model.save(model_save_file)
        
