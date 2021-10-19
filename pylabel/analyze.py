import pandas as pd

class Analyze:
    def  __init__(self, df):
        self._df = df 
        self.class_counts = self._df["cat_name"].value_counts(dropna=False)
        self.num_classes = self._df["cat_name"].nunique()
        self.classes = self._df["cat_name"].unique()
        self.num_images = self._df["img_filename"].nunique()
        self.split_counts = self._df["split"].value_counts(dropna=False)
        self.split_pct = self._df["split"].value_counts(normalize=True, dropna=False)

    def ShowClassSplits(self, df, normalize=True):

        def move_column_inplace(df, col, pos):
            """
            Assists to rearrange columns to a desired order. 
            """
            col = df.pop(col)
            df.insert(pos, col.name, col)

        df_value_counts = pd.DataFrame(df["cat_name"].value_counts(normalize=normalize), columns=["cat_name"])

        df_value_counts.index.name = "cat_name"  
        df_value_counts.columns = ["all"]

        split_df = df.groupby('split')

        if split_df.ngroups == 1:
            return df_value_counts

        for name, group in split_df:
            group_df = pd.DataFrame(group)
            df_split_value_counts = pd.DataFrame(group_df["cat_name"].value_counts(normalize=normalize), columns=["cat_name"])
            df_split_value_counts.index.name = "cat_name"
            df_split_value_counts.columns = [name]
            df_value_counts = pd.merge(df_value_counts, df_split_value_counts, how="left", on=["cat_name"])

        #Move 'train' to the left of the table since that is the usual convention.
        if 'train' in df_value_counts.columns:
            move_column_inplace(df_value_counts, 'train', 1)

        return df_value_counts



