"""The dataset is the primary object that you will interactive with when using PyLabel.
All other modules are sub-modules of the dataset object. 
"""

from pylabel.analyze import Analyze
from pylabel.exporter import Export
from pylabel.visualize import Visualize
from pylabel.labeler import Labeler
from pylabel.splitter import Split
from pylabel.shared import _ReindexCatIds

import numpy as np


class Dataset:
    def __init__(self, df):
        self.df = df
        """Pandas Dataframe: The dataframe where annotations are stored. This dataframe can be read directly
        to query the contents of the dataset. You can also edit this dataframe to filter records or edit the 
        annotations directly. 

        Example: 
            >>> dataset.df
        """
        self.name = "dataset"
        """string: Default is 'dataset'. A friendly name for your dataset that is used as part of the filename(s)
        when exporting annotation files. 
        """
        self.path_to_annotations = ""
        """string: Default is ''. The path to the annotation files associated with the dataset. When importing, 
        this will be path to the directory where the annotations are stored.  By default, annotations will be exported
        to the same directory. Changing this value will change where the annotations are exported to.  
        """
        self.export = Export(dataset=self)
        """See pylabel.export module."""
        self.visualize = Visualize(dataset=self)
        """See pylabel.visualize module."""
        self.analyze = Analyze(dataset=self)
        """See pylabel.analyze module."""
        self.labeler = Labeler(self)
        """See pylabel.labeler module."""
        self.splitter = Split(dataset=self)
        """See pylabel.splitter module."""

    def ReindexCatIds(self, cat_id_index=0):
        """
        Reindex the values of the cat_id column so that that they start from an int (usually 0 or 1) and
        then increment the cat_ids to index + number of categories.
        It's useful if the cat_ids are not continuous, especially for dataset subsets,
        or combined multiple datasets. Some models like Yolo require starting from 0 and others
        like Detectron require starting from 1.

        Args:
            cat_id_index (int): Defaults to 0.
                The cat ids will increment sequentially the cat_index value. For example if there are 10
                classes then the cat_ids will be a range from 0-9.

        Example:
            >>> dataset.analyze.class_ids
                [1,2,4,5,6,7,8,9,11,12]
            >>> dataset.ReindexCatIds(cat_id_index) = 0
            >>> dataset.analyze.class_ids
                [0,1,2,3,4,5,6,7,8,9]
        """
        assert isinstance(cat_id_index, int), "cat_id_index must be an int."
        _ReindexCatIds(self.df, cat_id_index)
