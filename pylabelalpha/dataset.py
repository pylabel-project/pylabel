from pylabelalpha.analyze import Analyze
from pylabelalpha.exporter import Export
from pylabelalpha.visualize import Visualize

from pylabelalpha.splitter import GroupShuffleSplit
from pylabelalpha.splitter import StratifiedGroupShuffleSplit


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

    

