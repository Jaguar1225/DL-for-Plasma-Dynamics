from .loader import Data_Loader

class Train_Data_Set(Data_Loader):
    def __init__(self):
        self.data_path = 'data/train_data.csv'
        super(Train_Data_Set, self).__init__(self.data_path)
    
