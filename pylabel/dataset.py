from pylabel.analyze import Analyze
from pylabel.exporter import Export
from pylabel.visualize import Visualize

from pylabel.splitter import GroupShuffleSplit
from pylabel.splitter import StratifiedGroupShuffleSplit


class Dataset:
    def __init__(self, df):
        self.df = df
        self.name = "dataset"
        self.path_to_annotations = ""
        self.analyze = Analyze(self.df) 
        self.GroupShuffleSplit = GroupShuffleSplit
        self.StratifiedGroupShuffleSplit = StratifiedGroupShuffleSplit
        self.export = Export()
        self.visualize = Visualize()

    

