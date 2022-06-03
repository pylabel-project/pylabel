"""This module includes the commands to import an existing dataset. 
PyLabel current supports importing labels from COCO, YOLO, and VOC formats. 
You can also import set of images that do not have labels yet and label them manually using the PyLabel
labelling tool. """

import json
import pandas as pd
import xml.etree.ElementTree as ET
import os
from os.path import exists
from pathlib import Path, PurePath
import copy
import cv2
import yaml

from pylabel.shared import schema
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


# These are the valid columns in the pylabel annotations table.
def ImportCoco(path, path_to_images=None, name=None):
    """
    This function takes the path to a JSON file in COCO format as input. It returns a PyLabel dataset object that contains the annotations.

    Returns:
        PyLabel dataset object.

    Args:
        path (str):The path to the JSON file with the COCO annotations.
        path_to_images (str): The path to the images relative to the json file.
            If the images are in the same directory as the JSON file then omit this parameter.
            If the images are in a different directory on the same level as the annotations then you would
            set `path_to_images='../images/'`
        name (str): This will set the dataset.name property for this dataset.
            If not specified, the filename (without extension) of the COCO annotation file file will be used as the dataset name.
    Example:
        >>> from pylabel import importer
        >>> dataset = importer.ImportCoco("coco_annotations.json")
    """
    with open(path) as cocojson:
        annotations_json = json.load(cocojson)

    # Store the 3 sections of the json as seperate json arrays
    images = pd.json_normalize(annotations_json["images"])
    images.columns = "img_" + images.columns
    try:
        images["img_folder"]
    except:
        images["img_folder"] = ""
    # print(images)

    # If the user has specified a different image folder then use that one
    if path_to_images != None:
        images["img_folder"] = path_to_images

    astype_dict = {"img_width": "int64", "img_height": "int64", "img_depth": "int64"}
    astype_keys = list(astype_dict.keys())
    for element in astype_keys:
        if element not in images.columns:
            astype_dict.pop(element)
    # print(astype_dict)
    # images = images.astype({'img_width': 'int64','img_height': 'int64','img_depth': 'int64'})
    images = images.astype(astype_dict)

    annotations = pd.json_normalize(annotations_json["annotations"])
    annotations.columns = "ann_" + annotations.columns

    categories = pd.json_normalize(annotations_json["categories"])
    categories.columns = "cat_" + categories.columns

    # Converting this to string resolves issue #23
    categories.cat_id = categories.cat_id.astype(str)

    df = annotations

    # Converting this to string resolves issue #23
    df.ann_category_id = df.ann_category_id.astype(str)

    df[
        ["ann_bbox_xmin", "ann_bbox_ymin", "ann_bbox_width", "ann_bbox_height"]
    ] = pd.DataFrame(df.ann_bbox.tolist(), index=df.index)
    df.insert(8, "ann_bbox_xmax", df["ann_bbox_xmin"] + df["ann_bbox_width"])
    df.insert(10, "ann_bbox_ymax", df["ann_bbox_ymin"] + df["ann_bbox_height"])

    # debug print(df.info())

    # Join the annotions with the information about the image to add the image columns to the dataframe
    df = pd.merge(images, df, left_on="img_id", right_on="ann_image_id", how="left")
    df = pd.merge(
        df, categories, left_on="ann_category_id", right_on="cat_id", how="left"
    )

    # Rename columns if needed from the coco column name to the pylabel column name
    df.rename(columns={"img_file_name": "img_filename"}, inplace=True)

    # Drop columns that are not in the schema
    df = df[df.columns.intersection(schema)]

    # Add missing columns that are in the schema but not part of the table
    df[list(set(schema) - set(df.columns))] = ""

    # Reorder columns
    df = df[schema]
    df.index.name = "id"
    df.annotated = 1

    # Fill na values with empty strings which resolved some errors when
    # working with images that don't have any annotations
    df.fillna("", inplace=True)

    # These should be strings
    df.cat_id = df.cat_id.astype(str)

    # These should be integers
    df.img_width = df.img_width.astype(int)
    df.img_height = df.img_height.astype(int)

    dataset = Dataset(df)

    # Assign the filename (without extension) as the name of the dataset
    if name == None:
        dataset.name = Path(path).stem
    else:
        dataset.name = name

    dataset.path_to_annotations = PurePath(path).parent

    return dataset


def ImportVOC(path, path_to_images=None, name="dataset"):
    """
    Provide the path a directory with annotations in VOC Pascal XML format and it returns a PyLabel dataset object that contains the annotations.

    Returns:
        PyLabel dataset object.

    Args:
        path (str): The path to the directory with the annotations in VOC Pascal XML format.
        path_to_images (str): The path to the images relative to the annotations.
            If the images are in the same directory as the annotation files then omit this parameter.
            If the images are in a different directory on the same level as the annotations then you would
            set `path_to_images='../images/'`
        name (str): Default is 'dataset'. This will set the dataset.name property for this dataset.

    Example:
        >>> from pylabel import importer
        >>> dataset = importer.ImportVOC(path="annotations/", path_to_images="../images/")
    """
    # Create an empty dataframe
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
        if filename.is_file() and filename.name.endswith(".xml"):
            filepath = filename.path
            xml_data = open(filepath, "r").read()  # Read file
            root = ET.XML(xml_data)  # Parse XML
            folder = _GetValueOrBlank(root.find("folder"), user_input=path_to_images)
            filename = root.find("filename").text
            size = root.find("size")
            size_width = size.find("width").text
            size_height = size.find("height").text
            size_depth = _GetValueOrBlank(size.find("depth"))
            segmented = _GetValueOrBlank(root.find("segmented"))

            row = {}
            # Build dictionary that will be become the row in the dataframe
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

                # Add this row to the dict
                d[row_id] = copy.deepcopy(row)
                # increment the rowid
                row_id += 1

        # Increment the imageid because we are going to read annother file
        img_id += 1

    # Convert the dict with all of the annotation data to a dataframe
    df = pd.DataFrame.from_dict(d, "index", columns=schema)
    df.index.name = "id"
    df.annotated = 1

    # These should be strings
    df.cat_id = df.cat_id.astype(str)

    # These should be integers
    df.img_width = df.img_width.astype(int)
    df.img_height = df.img_height.astype(int)

    # Reorder columns
    df = df[schema]
    dataset = Dataset(df)
    dataset.name = name
    dataset.path_to_annotations = path
    # Get the path without the filename
    # dataset.path_to_annotations = "Alex ander"
    return dataset


def ImportYoloV5(
    path,
    img_ext="jpg,jpeg,png",
    cat_names=[],
    path_to_images="",
    name="dataset",
):
    """
    Provide the path a directory with annotations in YOLO format and it returns a PyLabel dataset object that contains the annotations.
    The Yolo format does not store much information about the images, such as the height and width. When you import a
    Yolo dataset PyLabel will extract this information from the images.

    Returns:
        PyLabel dataset object.

    Args:
        path (str): The path to the directory with the annotations in YOLO format.
        img_ext (str, comma separated): Specify the file extension(s) of the images used in your dataset:
         .jpeg, .png, etc. This is required because the YOLO format does not store the filename of the images.
         It could be any of the image formats supported by YoloV5. PyLabel will iterate through the file extensions
         specified until it finds a match.
        cat_names (list): YOLO annotations only store a class number, not the name. You can provide a list of class ids
            that correspond to the int used to represent that class in the annotations. For example `['Squirrel,'Nut']`.
            If you have the class names already stored in a YOLO YAML file then use the ImportYoloV5WithYaml method to
            automatically read the class names from that file.
        path_to_images (str): The path to the images relative to the annotations.
            If the images are in the same directory as the annotation files then omit this parameter.
            If the images are in a different directory on the same level as the annotations then you would
            set `path_to_images='../images/'`
        name (str): Default is 'dataset'. This will set the dataset.name property for this dataset.

    Example:
        >>> from pylabel import importer
        >>> dataset = importer.ImportYoloV5(path="labels/", path_to_images="../images/")
    """

    def GetCatNameFromId(cat_id, cat_names):
        cat_id = int(cat_id)
        if len(cat_names) > int(cat_id):
            return cat_names[cat_id]

    # Create an empty dataframe
    df = pd.DataFrame(columns=schema)

    # the dictionary to pass to pandas dataframe
    d = {}

    row_id = 0
    img_id = 0

    # iterate over files in that directory
    for filename in os.scandir(path):
        if filename.is_file() and filename.name.endswith(".txt"):
            filepath = filename.path
            file = open(filepath, "r")  # Read file
            row = {}

            # First find the image files and extract the metadata about the image
            row["img_folder"] = path_to_images

            # Figure out what the extension is of the corresponding image file
            # by looping through the extension in the img_ext parameter
            found_image = False
            for ext in img_ext.split(","):
                image_filename = filename.name.replace("txt", ext)

                # Get the path to the image file to extract the height, width, and depth
                image_path = PurePath(path, path_to_images, image_filename)
                if exists(image_path):
                    found_image = True
                    break

            # Check if there is a file at this location.
            assert (
                found_image == True
            ), f"No image file found: {image_path}. Check path_to_images and img_ext arguments."

            row["img_filename"] = image_filename

            im = cv2.imread(str(image_path))
            img_height, img_width, img_depth = im.shape

            row["img_id"] = img_id
            row["img_width"] = img_width
            row["img_height"] = img_height
            row["img_depth"] = img_depth

            # Read the annotation in the file
            # Check if the file has at least one line:
            numlines = len(open(filepath).readlines())
            if numlines == 0:
                # Create a row without annotations
                d[row_id] = row
                row_id += 1
            else:
                for line in file:
                    line = line.strip()

                    # check if the row is empty, leave annotation columns blank
                    if line:
                        d[row_id] = copy.deepcopy(row)
                        (
                            cat_id,
                            x_center_norm,
                            y_center_norm,
                            width_norm,
                            height_norm,
                        ) = line.split()

                        row["ann_bbox_width"] = float(width_norm) * img_width
                        row["ann_bbox_height"] = float(height_norm) * img_height
                        row["ann_bbox_xmin"] = float(x_center_norm) * img_width - (
                            (row["ann_bbox_width"] / 2)
                        )
                        row["ann_bbox_ymax"] = float(y_center_norm) * img_height + (
                            (row["ann_bbox_height"] / 2)
                        )
                        row["ann_bbox_xmax"] = (
                            row["ann_bbox_xmin"] + row["ann_bbox_width"]
                        )
                        row["ann_bbox_ymin"] = (
                            row["ann_bbox_ymax"] - row["ann_bbox_height"]
                        )

                        row["ann_area"] = row["ann_bbox_width"] * row["ann_bbox_height"]

                        row["cat_id"] = cat_id
                        row["cat_name"] = GetCatNameFromId(cat_id, cat_names)

                        d[row_id] = dict(row)
                        row_id += 1
                        # Copy the image data to use for the next row
                    else:
                        # Create a row without annotations
                        d[row_id] = row
                        row_id += 1

                # Add this row to the dict
        # increment the image id
        img_id += 1

    df = pd.DataFrame.from_dict(d, "index", columns=schema)
    df.index.name = "id"
    df.annotated = 1
    df.fillna("", inplace=True)

    # These should be strings
    df.cat_id = df.cat_id.astype(str)

    # These should be integers
    df.img_width = df.img_width.astype(int)
    df.img_height = df.img_height.astype(int)

    # Reorder columns
    dataset = Dataset(df)
    dataset.name = name
    dataset.path_to_annotations = path

    return dataset


def ImportImagesOnly(path, name="dataset"):
    """Import a directory of images as a dataset with no annotations.
    Then use PyLabel to annote the images. Will import images with these extensions:
    ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

    Args:
        path (str):
            The path to the directory with the images.
        name (str):
            Default is 'dataset'. Descriptive name, which is used when outputting files.
    Returns:
        A dataset object with one row for each image and no annotations.

    Example:
        >>> from pylabel import importer
        >>> dataset = importer.ImportImagesOnly(path="images/")
    """

    # Create an empty dataframe
    df = pd.DataFrame(columns=schema)

    # the dictionary to pass to pandas dataframe
    d = {}

    img_id = 0

    # iterate over files in that directory
    for filename in os.scandir(path):
        if filename.is_file() and filename.name.lower().endswith(
            (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif")
        ):
            row = {}
            row["img_folder"] = ""
            row["img_filename"] = filename.name
            image_path = PurePath(path, row["img_filename"])
            im = cv2.imread(str(image_path))

            try:
                # If the file is not an image then this will fail
                im.shape
            except:
                raise ValueError(
                    f"Error reading file '{image_path}'. Exclude non-image files by using the ends_width param."
                )

            img_height, img_width, img_depth = im.shape

            row["img_id"] = img_id
            row["img_width"] = img_width
            row["img_height"] = img_height
            row["img_depth"] = img_depth
            row["cat_name"] = ""

            # Add this row to the dict
            d[img_id] = row
            img_id += 1

    df = pd.DataFrame.from_dict(d, "index", columns=schema)
    df.index.name = "id"

    # Reorder columns
    dataset = Dataset(df)
    dataset.name = name
    dataset.path_to_annotations = path

    return dataset


def _yaml_reader(yaml_file):
    """Import the YAML File for the YOLOv5 data as dict."""
    with open(yaml_file) as file:
        data = yaml.safe_load(file)
    return data


def ImportYoloV5WithYaml(
    yaml_file,
    image_ext="jpg",
    name_of_annotations_folder="labels",
    path_to_annotations=None,
):
    """Import a YOLO dataset by reading the YAML file to extract the class names, image and label locations,
    and preserve if an image should be in the train, test, or val split.

    Returns:
        PyLabel dataset object.

    Args:
        yaml_file (str):
            Path to the yaml file that describes the dataset to be imported.
        image_ext (str):
            The image file extension.
        path_to_annotations (str):
            the path to the annotations file; if path to annotations is none, file replaces name of images file from yaml file with annotations.
        name_of_annotations_folder (str):
            Default is "labels". Change this to "annotations" if your folder is called "annotations"

    Example:
        >>> from pylabel import importer
        >>> dataset = importer.ImportYoloV5WithYaml(yaml_file='data/dataset.yaml')

    """

    """
    Note to other developers:
    As a note, the "path_to_images" variable in this code refers to the relative path relative to the path to annotations.
    It is different than the path to images specified in the YAML file.
    PyLabel uses the former to establish its pathing and the latter path to actually view the needed data.
    """

    path_to_annotations_copy = path_to_annotations

    if path_to_annotations == None:
        path_to_annotations_defined = False
    else:
        path_to_annotations_defined = True
    counter = 0

    data = _yaml_reader(yaml_file)
    yoloclasses = data["names"]

    iterated_list = list(data.keys())

    for splitted in iterated_list:
        if splitted in ["nc", "names"]:
            pass
        else:
            try:
                path_to_images = data[splitted]
            except:
                raise Exception("split type not in the YAML file.")
            # if counter > 0
            # change PoA to new split type

            # if
            if path_to_annotations == None or counter != 0:
                # In case your folder is called labels or some thing else that doesn't jive with what we want you to call it.
                if path_to_annotations_defined == True and counter > 0:
                    path_to_annotations = str(
                        PurePath(
                            path_to_annotations_copy.replace(
                                iterated_list[counter - 1], splitted
                            )
                        )
                    )

                # This probably needs to be reworked but there's potentially an issue with resetting the path
                # to annotations as we iterate through each split type.
                elif name_of_annotations_folder != "labels" and counter > 0:
                    path_to_annotations = str(
                        PurePath(
                            path_to_annotations.replace(
                                iterated_list[counter - 1], splitted
                            )
                        )
                    )
                elif name_of_annotations_folder != "labels" and counter == 0:
                    path_to_annotations = str(
                        PurePath(
                            path_to_images.replace("images", name_of_annotations_folder)
                        )
                    )
                else:
                    path_to_annotations = str(
                        PurePath(path_to_images.replace("images", "labels"))
                    )

                path_to_images = str(PurePath("../../images/", splitted))

            if counter == 0:
                dataset = ImportYoloV5(
                    path=path_to_annotations,
                    path_to_images=path_to_images,
                    cat_names=yoloclasses,
                    img_ext=image_ext,
                )
                dataset.df["split"] = splitted
                counter += 1
            else:
                dataset2 = ImportYoloV5(
                    path=path_to_annotations,
                    path_to_images=path_to_images,
                    cat_names=yoloclasses,
                    img_ext=image_ext,
                )
                dataset2.df["split"] = splitted

                # This code is added so that the image ids are unique when the multiple datasets are merged
                # It will take the max img_id of the first data set
                # And then add that to the image ids in the second dataset so they don't collide
                max_img_id = max(dataset.df["img_id"])
                dataset2.df["img_id"] += max_img_id + 1
                dataset.df = dataset.df.append(dataset2.df)
                dataset.df.reset_index(0, inplace=True)
                counter += 1
    return dataset
