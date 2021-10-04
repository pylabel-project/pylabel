class Analyze:
    def  __init__(self, df):
        self._df = df 
        self.class_counts = self._df["cat_name"].value_counts(dropna=False)
        self.num_classes = self._df["cat_name"].nunique()
        self.classes = self._df["cat_name"].unique()
        self.num_images = self._df["img_filename"].nunique()
        self.split_counts = self._df["split"].value_counts(dropna=False)
        self.split_pct = self._df["split"].value_counts(normalize=True, dropna=False)



