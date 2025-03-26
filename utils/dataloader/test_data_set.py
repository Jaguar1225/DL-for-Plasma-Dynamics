from .loader import Data_Loader

class Test_Data_Set(Data_Loader):
    def __init__(self, data_path):
        super(Test_Data_Set, self).__init__(data_path + 'Test/')