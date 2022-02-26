"""The analyze module analyzes the contents of the dataset and provides summary statistics 
such as the count of images and classes. """

import pandas as pd
import numpy as np


class Analyze:
    def __init__(self, dataset=None):
        self.dataset = dataset
        # self.split_counts = ds.df["split"].value_counts(dropna=False)
        # self.split_pct = ds.df["split"].value_counts(normalize=True, dropna=False)

    @property
    def classes(self):
        """Returns list of all cat names in the dataset sorted by cat_id value.

        Returns:
            List

        Example:
            >>> dataset.analyze.classes
            ["Squirrel", "Nut"]
        """

        self.dataset.df.cat_id = self.dataset.df.cat_id.replace(
            r"^\s*$", np.nan, regex=True
        )
        pd.to_numeric(self.dataset.df["cat_id"])

        filtered_df = self.dataset.df[self.dataset.df["cat_id"].notnull()]
        categories = dict(zip(filtered_df.cat_name, filtered_df.cat_id.astype("int")))
        categories = sorted(categories.items(), key=lambda x: x[1])
        return [c[0] for c in categories if str(c[0]).strip() != ""]

    @property
    def class_ids(self):
        """Returns a sorted list of all cat ids in the dataset.

        Returns:
            List

        Example:
            >>> dataset.analyze.class_ids
            [0,1]
        """
        filtered_df = self.dataset.df[self.dataset.df["cat_id"].notnull()]
        cat_ids = list(filtered_df.cat_id.astype("int").unique())
        cat_ids.sort()
        return cat_ids

    @property
    def class_counts(self):
        """Counts the numbers of instances of each class in the dataset. Uses the Pandas value_counts
        method and returns a Pandas Series.

        Returns:
            Pandas Series
        Example:
            >>> dataset.analyze.class_counts
            squirrel  50
            nut       100
        """
        return self.dataset.df["cat_name"].value_counts(dropna=False)

    @property
    def num_classes(self):
        """Counts the unique number of classes in the dataset.

        Returns:
            Int
        Example:
            >>> dataset.analyze.num_classes
            2
        """
        cat_names = list(self.dataset.df.cat_name.unique())
        return len([i for i in cat_names if str(i).strip() != ""])

    @property
    def num_images(self):
        """Counts the number of images in the dataset.

        Returns:
            Int
        Example:
            >>> dataset.analyze.num_images
            100
        """
        return self.dataset.df["img_filename"].nunique()

    @property
    def class_name_id_map(self):
        """Returns a dict where the class name is the key and class id is the value.

        Returns:
            Dict

        Example:
            >>> dataset.analyze.class_name_id_map
            {('Squirrel', 0),('Nut', 1)}
        """
        return dict(zip(self.dataset.df.cat_name, self.dataset.df.cat_id))

    def ShowClassSplits(self, normalize=True):
        """Show the distribution of classes across train, val, and
        test splits of the dataset.

        Args:
            normalize (bool): Defaults to True.
                If True, then will return the relative frequencies of the classes between 0 and 1.
                If False, then will return the absolute counts of each class.

        Returns:
            Pandas Dataframe

        Examples:
            >>> dataset.analyze.ShowClassSplits(normalize=True)
            cat_name  all  train  test  dev
            squirrel  .66  .64    .65   .63
            nut       .34  .34    .35   .37

            >>> dataset.analyze.ShowClassSplits(normalize=False)
            cat_name  all  train  test  dev
            squirrel  66   64     65    63
            nut       34   34     35    37

        """
        ds = self.dataset

        def move_column_inplace(df, col, pos):
            """
            Assists to rearrange columns to a desired order.
            """
            col = df.pop(col)
            df.insert(pos, col.name, col)

        df_value_counts = pd.DataFrame(
            ds.df["cat_name"].value_counts(normalize=normalize), columns=["cat_name"]
        )

        df_value_counts.index.name = "cat_name"
        df_value_counts.columns = ["all"]

        split_df = ds.df.groupby("split")

        if split_df.ngroups == 1:
            return df_value_counts

        for name, group in split_df:
            group_df = pd.DataFrame(group)
            df_split_value_counts = pd.DataFrame(
                group_df["cat_name"].value_counts(normalize=normalize),
                columns=["cat_name"],
            )
            df_split_value_counts.index.name = "cat_name"
            df_split_value_counts.columns = [name]
            df_value_counts = pd.merge(
                df_value_counts, df_split_value_counts, how="left", on=["cat_name"]
            )

        # Move 'train' to the left of the table since that is the usual convention.
        if "train" in df_value_counts.columns:
            move_column_inplace(df_value_counts, "train", 1)

        return df_value_counts
