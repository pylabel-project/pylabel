from pylabelalpha.analyze import Analyze
from pylabelalpha.exporter import Export
from pylabelalpha.splitter import StratifiedGroupShuffleSplit2 

class Dataset:
    def __init__(self, df):
        self.df = df
        self.name = "dataset"
        self.analyze = Analyze(self.df) 
        self.StratifiedGroupShuffleSplit2 = StratifiedGroupShuffleSplit2
        #self.export = Export(self.df)

    

    

