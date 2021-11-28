from pylabel.analyze import Analyze
from pylabel.exporter import Export
from pylabel.visualize import Visualize
from pylabel.labeler import Labeler
from pylabel.splitter import Split


#from pylabel.splitter import GroupShuffleSplit, test
#from pylabel.splitter import StratifiedGroupShuffleSplit

def FilterList(the_list):
    """To do: Check why this code is here"""
    the_list = [i for i in the_list if len(i.strip()) > 0]

class Dataset:
    def __init__(self, df):
        self.df = df
        self.name = "dataset"
        self.path_to_annotations = ""    
        self.export = Export(dataset=self)
        self.visualize = Visualize(dataset=self)
        self.analyze = Analyze(dataset=self)
        self.labeler = Labeler(self)
        self.splitter = Split(dataset=self)

