class Analyze:
    def  __init__(self, df):
        self._df = df 
        self.class_counts = self._df["cat_name"].value_counts()
        self.num_classes = self._df["cat_name"].nunique()
        self.num_images = self._df["img_filename"].nunique()


