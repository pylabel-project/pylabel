from pylabelalpha.analyze import Analyze
from pylabelalpha.exporter import Export

class Dataset:
    def __init__(self, df):
        self.df = df
        self.name = "dataset"
        self.analyze = Analyze(self.df)   
        #self.export = Export(self.df)

    

    

