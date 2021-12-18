from pylabel.analyze import Analyze
from pylabel.exporter import Export
from pylabel.visualize import Visualize
from pylabel.labeler import Labeler
from pylabel.splitter import Split
from pylabel.shared import _ReindexCatIds


import numpy as np

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

    def ReindexCatIds(self, cat_id_index=0):
        """
        Reindex the values of the cat_id column so that that they start from an int (usually 0 or 1) and 
        then increment the cat_ids to index + number of categories. 
        It's useful if the cat_ids are not continuous, especially for dataset subsets, 
        or combined multiple datasets. Some models like Yolo require starting from 0 and others 
        like Detectron require starting from 1.
        """
        assert isinstance(cat_id_index, int), "cat_id_index must be an int."
        _ReindexCatIds(self.df, cat_id_index)