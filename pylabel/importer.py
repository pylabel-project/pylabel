import json
import pandas as pd
import xml.etree.ElementTree as ET
import os
from pathlib import Path, PurePath
import copy
import cv2

from pylabel.constants import schema
from pylabel.dataset import Dataset
from pylabel.exporter import Export

def _GetValueOrBlank(element, user_input=None):
    """
    If an element is missing from the XML file reading the .text value will return an error.
    If the element does not exist return ""
    """
    if user_input == None:
        return element.text 
    else:
        return user_input

#These are the valid columns in the pylabel annotations table.              
def ImportCoco(path, path_to_images=None, name=""):
    """
    This function takes the path to an xml file in coco format as input. 
    It returns a dataframe in the schema used by pylable to store annotations. 
    """

    with open(path) as cocojson:
        annotations_json = json.load(cocojson)

    #Store the 3 sections of the json as seperate json arrays
    images = pd.json_normalize(annotations_json["images"])
    images.columns = 'img_' + images.columns
    images["img_folder"] = _GetValueOrBlank(images["img_folder"], path_to_images)
    images = images.astype({'img_width': 'int64','img_height': 'int64','img_depth': 'int64'})

    annotations = pd.json_normalize(annotations_json["annotations"])
    annotations.columns = 'ann_' + annotations.columns

    categories = pd.json_normalize(annotations_json["categories"])
    categories.columns = 'cat_' + categories.columns

    df = annotations
    df[['ann_bbox_xmin','ann_bbox_ymin','ann_bbox_width','ann_bbox_height']] = pd.DataFrame(df.ann_bbox.tolist(), index= df.index)
    df.insert(8, 'ann_bbox_xmax', df['ann_bbox_xmin'] + df['ann_bbox_width'] )
    df.insert(10, 'ann_bbox_ymax', df['ann_bbox_ymin'] + df['ann_bbox_height'] )
    
    #debug print(df.info())

    #Join the annotions with the information about the image to add the image columns to the dataframe
    df = pd.merge(images, df, left_on='img_id', right_on='ann_image_id', how='left')
    df = pd.merge(df, categories, left_on='ann_category_id', right_on='cat_id', how='left')
    
    #Rename columns if needed from the coco column name to the pylabel column name 
    df.rename(columns={"img_file_name": "img_filename"}, inplace=True)

    #Drop columns that are not in the schema
    df = df[df.columns.intersection(schema)]

    #Add missing columns that are in the schema but not part of the table
    df[list(set(schema) - set(df.columns))] = ""

    #Reorder columns
    df = df[schema]
    df.index.name = "id"

    dataset = Dataset(df)

    #Assign the filemame (without extension) as the name of the dataset
    if name == "":
        dataset.name = Path(path).stem
    else:
        dataset.name = name

    dataset.path_to_annotations = PurePath(path).parent

    return dataset

def ImportVOC(path, path_to_images=None, name="dataset"):
    #Create an empty dataframe
    df = pd.DataFrame(columns=schema) 

    # the dictionary to pass to pandas dataframe
    d = {}

    row_id = 0
    img_id = 0
    cat_names = []

    def GetCatId(cat_name):
        """This will assign a numeric cat_id to each cat_name."""
        if cat_name not in cat_names:
            cat_names.append(cat_name)
        return cat_names.index(cat_name)

    # iterate over files in that directory
    for filename in os.scandir(path):
        if filename.is_file() and filename.name.endswith('.xml'):
            filepath = filename.path
            xml_data = open(filepath, 'r').read()  # Read file
            root = ET.XML(xml_data)  # Parse XML
            folder = _GetValueOrBlank(root.find("folder"), user_input=path_to_images)
            filename = root.find("filename").text
            size = root.find("size")
            size_width = size.find("width").text
            size_height = size.find("height").text
            size_depth = _GetValueOrBlank(size.find("depth"))
            segmented = _GetValueOrBlank(root.find("segmented"))

            row = {}
            #Build dictionary that will be become the row in the dataframe
            row["img_folder"] = folder
            row["img_filename"] = filename
            row["img_id"] = img_id
            row["img_width"] = size_width
            row["img_height"] = size_height
            row["img_depth"] = size_depth
            row["ann_segmented"] = segmented

            object = root.findall("object")

            for o in object:
                row["cat_name"] = o.find("name").text
                row["cat_id"] = GetCatId(row["cat_name"])
                row["ann_pose"] = _GetValueOrBlank(o.find("pose"))
                row["ann_truncated"] = _GetValueOrBlank(o.find("truncated"))
                row["ann_difficult"] = _GetValueOrBlank(o.find("difficult"))
                row["ann_bbox_xmin"] = float(o.find("bndbox").find("xmin").text)
                row["ann_bbox_ymin"] = float(o.find("bndbox").find("ymin").text)
                row["ann_bbox_xmax"] = float(o.find("bndbox").find("xmax").text)
                row["ann_bbox_ymax"] = float(o.find("bndbox").find("ymax").text)
                row["ann_bbox_width"] = row["ann_bbox_xmax"] - row["ann_bbox_xmin"] 
                row["ann_bbox_height"] = row["ann_bbox_ymax"] - row["ann_bbox_ymin"] 
                row["ann_area"] = row["ann_bbox_width"] * row["ann_bbox_height"] 
                row["split"] = ""

                #Add this row to the dict
                d[row_id] = copy.deepcopy(row)
                #increment the rowid
                row_id += 1

        #Increment the imageid because we are going to read annother file
        img_id += 1

    #Convert the dict with all of the annotation data to a dataframe
    df = pd.DataFrame.from_dict(d, "index", columns=schema)
    df.index.name = "id"

    #Reorder columns
    df = df[schema]
    dataset = Dataset(df)
    dataset.name = name
    dataset.path_to_annotations = path
    #Get the path without the filename 
    #dataset.path_to_annotations = "Alex ander"
    return dataset
        

def ImportYoloV5(path, img_ext="jpg",cat_names=[], path_to_images="", name="dataset",):
    
    def GetCatNameFromId(cat_id, cat_names):
        cat_id = int(cat_id)
        if len(cat_names) > int(cat_id):
            return cat_names[cat_id]

    #Create an empty dataframe
    df = pd.DataFrame(columns=schema) 


    # the dictionary to pass to pandas dataframe
    d = {}

    row_id = 0
    img_id = 0

    # iterate over files in that directory
    for filename in os.scandir(path):
        if filename.is_file() and filename.name.endswith('.txt'):
            filepath = filename.path
            file = open(filepath, 'r')  # Read file

            for line in file:
                row = {}

                cat_id, x_center_norm, y_center_norm, width_norm, height_norm = line.split()
                row["img_folder"] = path_to_images
                row["img_filename"] = filename.name.replace("txt",img_ext)

                #Get the path to the image file to extract the height, width, and depth
                image_path = PurePath(path, path_to_images, row["img_filename"])
                im = cv2.imread(str(image_path))
                img_height, img_width, img_depth =  im.shape

                row["img_id"] = img_id
                row["img_width"] = img_width
                row["img_height"] = img_height
                row["img_depth"] = img_depth

                row["ann_bbox_width"] = float(width_norm) * img_width
                row["ann_bbox_height"] = float(height_norm) * img_height
                row["ann_bbox_xmin"] = float(x_center_norm) * img_width  - ((row["ann_bbox_width"]  / 2))
                row["ann_bbox_ymax"] = float(y_center_norm) * img_height  + ((row["ann_bbox_height"]  / 2))
                row["ann_bbox_xmax"] = row["ann_bbox_xmin"] + row["ann_bbox_width"]
                row["ann_bbox_ymin"] = row["ann_bbox_ymax"] - row["ann_bbox_height"]

                row["ann_area"] = row["ann_bbox_width"] * row["ann_bbox_height"] 

                row["cat_id"] = cat_id
                row["cat_name"] = GetCatNameFromId(cat_id, cat_names)

                #Add this row to the dict
                d[row_id] = row
                row_id += 1
        img_id += 1

    df = pd.DataFrame.from_dict(d, "index", columns=schema)
    df.index.name = "id"

    #Reorder columns
    dataset = Dataset(df)
    dataset.name = name
    dataset.path_to_annotations = path

    return dataset