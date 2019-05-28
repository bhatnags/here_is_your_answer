
''' ToDo '''

class read_file():
    def __init__(self):
        self.filename = 'xyz'
        self.columns = ['context', 'ques', 'ans']
        self.seq_len = 50
        
    def get_data(self):
        return read_file(self.filename)

