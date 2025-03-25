from .loader import Data_Loader

class Test_Data_Set(Data_Loader):
    def __init__(self):
        self.data_path = 'data/test_data.csv'
        super(Test_Data_Set, self).__init__(self.data_path)