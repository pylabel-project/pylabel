from pylabelalpha.analyze import Analyze
from pylabelalpha.exporter import Export
from pylabelalpha.splitter import GroupShuffleSplit
from pylabelalpha.splitter import StratifiedGroupShuffleSplit


class Dataset:
    def __init__(self, df):
        self.df = df
        self.name = "dataset"
        self.analyze = Analyze(self.df) 
        self.GroupShuffleSplit = GroupShuffleSplit
        self.StratifiedGroupShuffleSplit = StratifiedGroupShuffleSplit
        #self.export = Export(self.df)

    

    

