import json
import pandas as pd
import xml.etree.ElementTree as ET
import os
from os.path import exists
from pathlib import Path, PurePath
import copy
import cv2
import yaml

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
    df.annotated = 1

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
    df.annotated = 1


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

                #Check if there is a file at this location.
                assert exists(image_path), f"File does not exist: {image_path}. Check path_to_images and img_ext arguments."

                
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
    df.annotated = 1

    #Reorder columns
    dataset = Dataset(df)
    dataset.name = name
    dataset.path_to_annotations = path

    return dataset


def ImportImagesOnly(path, ends_with=None, name="dataset"):
    """Import a directory of images as a dataset with no annotations.
    Then use PyLabel to annote the images.
    
        
    Args:
        path (str): 
            The path to the directory with the images. 
        ends_with, optional(tuple or None): 
            Specify which file types to import. 
            Use if there are files in the directly that you don't wnat to import.

    Returns:
        A dataset object with one row for each image and no annotations. 
    """
    #Create an empty dataframe
    df = pd.DataFrame(columns=schema) 

    # the dictionary to pass to pandas dataframe
    d = {}

    img_id = 0

    # iterate over files in that directory
    for filename in os.scandir(path):
        if filename.is_file() and filename.name.endswith(ends_with):
            filepath = filename.path
            #file = open(filepath, 'r')  # Read file

            row = {}
            row["img_folder"] = ""
            row["img_filename"] = filename.name
            image_path = PurePath(path, row["img_filename"])
            im = cv2.imread(str(image_path))
            img_height, img_width, img_depth =  im.shape

            row["img_id"] = img_id
            row["img_width"] = img_width
            row["img_height"] = img_height
            row["img_depth"] = img_depth
            row["cat_name"] = ''


            #Add this row to the dict
            d[img_id] = row
            img_id += 1

    df = pd.DataFrame.from_dict(d, "index", columns=schema)
    df.index.name = "id"

    #Reorder columns
    dataset = Dataset(df)
    dataset.name = name
    dataset.path_to_annotations = path

    return dataset

def yaml_reader(yaml_file):
    """Import the YAML File for the YOLOv5 data as dict."""
    with open(yaml_file) as file:
        data = yaml.safe_load(file)
    return(data)

def ImportYoloV5WithYaml(yaml_file, split_type=None, image_ext="jpg", name="coco128", name_of_annotations_folder="labels", path_to_annotations=None):
    """Convert the YAML file to the format needed for the YOLO import.
    yaml_file: the file name of the yaml file to be imported.
    split_type: if none, the full list, otherwise can be a list of the type of the data to be imported; 
                training set, testing set or validation set: ['train','test','val']
    path_to_annotations: the path to the annotations file; if path to annotations is none, file replaces name of images file from yaml file with annotations
    image_ending: the image file extension
    name: the type of format used"
    name_of_annotations_folder="annotations"; change this to "labels" if your folder is called "labels"
    make sure to define absolute path?
    
    As a note, the "path_to_images" variable in this code refers to the relative path relative to the path to annotations. 
    It is different than the path to images specified in the YAML file. 
    PyLabel uses the former to establish its pathing and the latter path to actually view the needed data.
    """

    path_to_annotations_copy = path_to_annotations
    
    if path_to_annotations == None:
        path_to_annotations_defined=False
    else:
        path_to_annotations_defined=True
    counter = 0
    
    data = yaml_reader(yaml_file)
    yoloclasses = data['names']  
    
    iterated_list = list(data.keys())

    for splitted in iterated_list:
        if splitted in ['nc', 'names']:
            pass
        else:     
            try:
                path_to_images = data[splitted]
            except:
                raise Exception("split type not in the YAML file.")
            #if counter > 0
            # change PoA to new split type

            #if 
            if path_to_annotations == None or counter != 0:
                # In case your folder is called labels or some thing else that doesn't jive with what we want you to call it.
                if path_to_annotations_defined == True and counter > 0:
                    path_to_annotations = str(PurePath(path_to_annotations_copy.replace(iterated_list[counter-1],splitted)))
                
                #This probably needs to be reworked but there's potentially an issue with resetting the path
                # to annotations as we iterate through each split type.
                elif name_of_annotations_folder != "labels" and counter > 0:
                    path_to_annotations = str(PurePath(path_to_annotations.replace(iterated_list[counter-1],splitted)))
                elif name_of_annotations_folder != "labels" and counter == 0:
                    path_to_annotations = str(PurePath(path_to_images.replace('images',name_of_annotations_folder)))
                else:
                    path_to_annotations = str(PurePath(path_to_images.replace('images','labels')))

                path_to_images = str(PurePath("../../images/",splitted))

            if counter == 0:
                dataset = ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=yoloclasses, img_ext=image_ext)
                dataset.df['split'] = splitted
                counter += 1
            else:      
                dataset2 = ImportYoloV5(path=path_to_annotations, path_to_images=path_to_images, cat_names=yoloclasses, img_ext=image_ext)
                dataset2.df['split'] = splitted
                dataset.df = dataset.df.append(dataset2.df)
                counter += 1
    return(dataset)