class configs():
    def __init__(self):
        self.loss= 'mse',
        self.optimizer= 'adam',
        self.save_dir= 'xyz_dir',
    
    def model_layers(self):
        layer_one = {'add': {'lstm': True, 'dense': False, 'dropout':False},
                     'neurons': 100,
                     'input_timesteps': 49,
                     'input_dim': 2,
                     'return_seq': True},
        layer_two = {'add': {'lstm': False, 'dense': False, 'dropout':True}, 
                     'rate': 0.2},
        layer_three = {'add': {'lstm': True, 'dense': False, 'dropout':False}, 
                       'neurons': 100, 
                       'return_seq': True},
        layer_four = {'add': {'lstm': True, 'dense': False, 'dropout':False}, 
                       'neurons': 100, 
                       'return_seq': True},
        layer_five= {'add': {'lstm': True, 'dense': False, 'dropout':False}, 
                       'neurons': 100, 
                       'return_seq': True},
        layer_six= {'add': {'lstm': True, 'dense': False, 'dropout':False}, 
                       'neurons': 100, 
                       'return_seq': True},
        layer_seven= {'add': {'lstm': True, 'dense': False, 'dropout':False}, 
                       'neurons': 100, 
                       'return_seq': True},
        layer_eight = {'add': {'lstm': True, 'dense': False, 'dropout':False}, 
                      'neurons': 100, 
                      'return_seq': False},
        layer_nine = {'add': {'lstm': False, 'dense': False, 'dropout':True}, 
                      'rate': 0.2},
        layer_ten = {'add': {'lstm': False, 'dense': True, 'dropout':False}, 
                     'neurons': 1, 
                     'activation': 'linear'}

        return [layer_one, layer_two, layer_three, layer_four, layer_five, 
                layer_six, layer_seven, layer_eight, layer_nine, layer_ten]
    
