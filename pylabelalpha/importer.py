import json
import pandas as pd
import xml.etree.ElementTree as ET
import os

from pylabelalpha.constants import schema
from pylabelalpha.dataset import Dataset


#These are the valid columns in the pylabel annotations table.              
def ImportCoco(path):
    """
    This function takes the path to an xml file in coco format as input. 
    It returns a dataframe in the schema used by pylable to store annotations. 
    """

    with open(path) as cocojson:
        annotations_json = json.load(cocojson)

    #Store the 3 sections of the json as seperate json arrays
    images = pd.json_normalize(annotations_json["images"])
    images.columns = 'img_' + images.columns

    annotations = pd.json_normalize(annotations_json["annotations"])
    annotations.columns = 'ann_' + annotations.columns

    categories = pd.json_normalize(annotations_json["categories"])
    categories.columns = 'cat_' + categories.columns

    df = annotations
    df[['ann_bbox_xmin','ann_bbox_ymax','ann_bbox_width','ann_bbox_height']] = pd.DataFrame(df.ann_bbox.tolist(), index= df.index)
    df.insert(8, 'ann_bbox_xmax', df['ann_bbox_xmin'] + df['ann_bbox_width'] )
    df.insert(10, 'ann_bbox_ymin', df['ann_bbox_ymax'] - df['ann_bbox_height'] )
    
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

    return Dataset(df)

def ImportVOC(path):
    print(path)
    #Create an empty dataframe
    df = pd.DataFrame(columns=schema) 

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

            folder = root.find("folder").text
            filename = root.find("filename").text
            size = root.find("size")
            size_width = size.find("width").text
            size_height = size.find("height").text
            size_depth = size.find("depth").text
            segmented = root.find("segmented").text

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
                row["id"] = row_id
                row["cat_name"] = o.find("name").text
                row["cat_id"] = GetCatId(row["cat_name"])
                row["ann_pose"] = o.find("pose").text
                row["ann_truncated"] = o.find("truncated").text
                row["ann_difficult"] = o.find("difficult").text
                row["ann_bbox_xmin"] = int(o.find("bndbox").find("xmin").text)
                row["ann_bbox_ymin"] = int(o.find("bndbox").find("ymin").text)
                row["ann_bbox_xmax"] = int(o.find("bndbox").find("xmax").text)
                row["ann_bbox_ymax"] = int(o.find("bndbox").find("ymax").text)
                row["ann_bbox_width"] = row["ann_bbox_xmax"] - row["ann_bbox_xmin"] 

                df = df.append(row, ignore_index=True)
                #increment the rowid
                row_id += 1

        #Increment the imageid because we are going to read annother file
        img_id += 1

    #Reorder columns
    df = df[schema]
    
    print(f'{img_id} annotation files read. {row_id} annotations imported.')

    return df