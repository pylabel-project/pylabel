import numpy as np
import pandas as pd

#########
# This file has variables and functions that are used by the rest of the package
#########

# These are the valid columns in the pylabel annotations table.
schema = [
    "img_folder",
    "img_filename",
    "img_path",
    "img_id",
    "img_width",
    "img_height",
    "img_depth",
    "ann_segmented",
    "ann_bbox_xmin",
    "ann_bbox_ymin",
    "ann_bbox_xmax",
    "ann_bbox_ymax",
    "ann_bbox_width",
    "ann_bbox_height",
    "ann_area",
    "ann_segmentation",
    "ann_iscrowd",
    "ann_pose",
    "ann_truncated",
    "ann_difficult",
    "cat_id",
    "cat_name",
    "cat_supercategory",
    "split",
    "annotated",
]
# schema = ['id','img_folder','img_filename','img_path','img_id','img_width','img_height','img_depth','ann_segmented','ann_bbox_xmin','ann_bbox_ymin','ann_bbox_xmax','ann_bbox_ymax','ann_bbox_width','ann_bbox_height','ann_area','ann_segmentation','ann_iscrowd','ann_pose','ann_truncated','ann_difficult','cat_id','cat_name','cat_supercategory','split']


def _ReindexCatIds(df, cat_id_index=0):
    """
    Reindex the values of the cat_id column so that that they start from an int (usually 0 or 1) and
    then increment the cat_ids to index + number of categories.
    It's useful if the cat_ids are not continuous, especially for dataset subsets,
    or combined multiple datasets. Some models like Yolo require starting from 0 and others
    like Detectron require starting from 1.
    """
    assert isinstance(cat_id_index, int), "cat_id_index must be an int."
    
    # Convert empty strings to NaN and drop rows with NaN cat_id
    df_copy = df.replace(r"^\s*$", np.nan, regex=True)
    df_copy = df_copy[df.cat_id.notnull()]
    # Coerce cat_id to int
    df_copy["cat_id"] = pd.to_numeric(df_copy["cat_id"])

    # Map cat_ids to the range [cat_id_index, cat_id_index + num_cats)
    unique_ids = np.sort(df_copy["cat_id"].unique())
    ids_dict = dict((v, k)
                    for k, v in enumerate(unique_ids, start=cat_id_index))
    df_copy["cat_id"] = df_copy["cat_id"].map(ids_dict)
    
    # Write back to the original dataframe
    df["cat_id"] = df_copy["cat_id"]
