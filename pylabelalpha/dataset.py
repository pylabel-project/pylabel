from pylabelalpha.analyze import Analyze
from pylabelalpha.exporter import Export

class Dataset:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        return Analyze(self.df)

    def export(self):
        return Export(self.df)

    

