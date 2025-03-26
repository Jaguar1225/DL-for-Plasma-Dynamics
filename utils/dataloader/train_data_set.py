from .loader import Data_Loader

class Train_Data_Set(Data_Loader):
    def __init__(self, data_path, **kwargs):
        super(Train_Data_Set, self).__init__(data_path + 'Train/', **kwargs)
    
