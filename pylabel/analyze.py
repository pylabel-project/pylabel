import pandas as pd

class Analyze:
    def  __init__(self, dataset=None):
        self.dataset = dataset 
        ds = self.dataset

        self.class_counts = ds.df["cat_name"].value_counts(dropna=False)
        self.num_classes = ds.df["cat_name"].nunique()
        self.classes = ds.df["cat_name"].unique()
        self.num_images = ds.df["img_filename"].nunique()
        self.split_counts = ds.df["split"].value_counts(dropna=False)
        self.split_pct = ds.df["split"].value_counts(normalize=True, dropna=False)

    def ShowClassSplits(self, normalize=True):
        ds = self.dataset

        def move_column_inplace(df, col, pos):
            """
            Assists to rearrange columns to a desired order. 
            """
            col = df.pop(col)
            df.insert(pos, col.name, col)

        df_value_counts = pd.DataFrame(ds.df["cat_name"].value_counts(normalize=normalize), columns=["cat_name"])

        df_value_counts.index.name = "cat_name"  
        df_value_counts.columns = ["all"]

        split_df = ds.df.groupby('split')

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
