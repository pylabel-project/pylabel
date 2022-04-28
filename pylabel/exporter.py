"""PyLabel currently supports exporting annotations in COCO, YOLO, and VOC PASCAL formats."""

import json
from typing import List
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os
import yaml
import shutil
from pylabel.shared import _ReindexCatIds

from pathlib import PurePath, Path


class Export:
    def __init__(self, dataset=None):
        self.dataset = dataset

    def ExportToVoc(
        self,
        output_path=None,
        segmented_=False,
        path_=False,
        database_=False,
        folder_=False,
        occluded_=False,
    ):
        """Writes annotation files to disk in VOC XML format and returns path to files.

        By default, tags with empty values will not be included in the XML output.
        You can optionally choose to include them if they are required for your solution.

        Args:
            output_path (str):
                This is where the annotation files will be written.
                If not-specified then the path will be derived from the .path_to_annotations and
                .name properties of the dataset object.
            segmented_ (bool) :
                Defaults to False. Set to true to include this field in the XML schema of the output files.
            path_ (bool) :
                Defaults to False. Set to true to include this field in the XML schema of the output files.
            database_ (bool) :
                Defaults to False. Set to true to include this field in the XML schema of the output files.
            folder_ (bool) :
                Defaults to False. Set to true to include this field in the XML schema of the output files.
            occluded_ (bool) :
                Defaults to False. Set to true to include this field in the XML schema of the output files.

        Returns:
            A list with 1 or more paths (strings) to annotations files.

        Example:
            >>> dataset.export.ExportToVoc()
            ['data/voc_annotations/000000000322.xml', ...]
        """
        ds = self.dataset

        if output_path == None:
            output_path = ds.path_to_annotations
        else:
            output_path = output_path

        os.makedirs(output_path, exist_ok=True)

        output_file_paths = []

        def voc_xml_file_creation(
            data,
            file_name,
            output_file_path,
            segmented=True,
            path=True,
            database=True,
            folder=True,
            occluded=True,
        ):

            index = 0
            df_smaller = data[data["img_filename"] == file_name].reset_index()

            if len(df_smaller) == 1:
                # print('test')
                annotation_text_start = "<annotation>"

                flder_lkp = str(df_smaller.loc[index]["img_folder"])
                if folder == True and flder_lkp != "":
                    folder_text = "<folder>" + flder_lkp + "</folder>"
                else:
                    folder_text = ""

                filename_text = (
                    "<filename>"
                    + str(df_smaller.loc[index]["img_filename"])
                    + "</filename>"
                )

                pth_lkp = str(df_smaller.loc[index]["img_path"])
                if path == True and pth_lkp != "":
                    path_text = "<path>" + pth_lkp + "</path>"
                else:
                    path_text = ""

                sources_text = ""

                size_text_start = "<size>"
                width_text = (
                    "<width>" + str(df_smaller.loc[index]["img_width"]) + "</width>"
                )
                height_text = (
                    "<height>" + str(df_smaller.loc[index]["img_height"]) + "</height>"
                )
                depth_text = (
                    "<depth>" + str(df_smaller.loc[index]["img_depth"]) + "</depth>"
                )
                size_text_end = "</size>"

                seg_lkp = str(df_smaller.loc[index]["ann_segmented"])
                if segmented == True and seg_lkp != "":
                    segmented_text = (
                        "<segmented>"
                        + str(df_smaller.loc[index]["ann_segmented"])
                        + "</segmented>"
                    )
                else:
                    segmented_text = ""

                # If the image has no annotations, skip this part of the output
                if not pd.isnull(df_smaller.loc[index]["cat_id"]):

                    object_text_start = "<object>"

                    name_text = (
                        "<name>" + str(df_smaller.loc[index]["cat_name"]) + "</name>"
                    )
                    pose_text = (
                        "<pose>" + str(df_smaller.loc[index]["ann_pose"]) + "</pose>"
                    )
                    truncated_text = (
                        "<truncated>"
                        + str(df_smaller.loc[index]["ann_truncated"])
                        + "</truncated>"
                    )
                    difficult_text = (
                        "<difficult>"
                        + str(df_smaller.loc[index]["ann_difficult"])
                        + "</difficult>"
                    )

                    occluded_text = ""

                    bound_box_text_start = "<bndbox>"

                    xmin_text = (
                        "<xmin>"
                        + str(int(df_smaller.loc[index]["ann_bbox_xmin"]))
                        + "</xmin>"
                    )
                    xmax_text = (
                        "<xmax>"
                        + str(int(df_smaller.loc[index]["ann_bbox_xmax"]))
                        + "</xmax>"
                    )
                    ymin_text = (
                        "<ymin>"
                        + str(int(df_smaller.loc[index]["ann_bbox_ymin"]))
                        + "</ymin>"
                    )
                    ymax_text = (
                        "<ymax>"
                        + str(int(df_smaller.loc[index]["ann_bbox_ymax"]))
                        + "</ymax>"
                    )

                    bound_box_text_end = "</bndbox>"
                    object_text_end = "</object>"
                else:
                    object_text_start = ""
                    name_text = ""
                    pose_text = ""
                    truncated_text = ""
                    difficult_text = ""
                    occluded_text = ""
                    bound_box_text_start = ""
                    xmin_text = ""
                    xmax_text = ""
                    ymin_text = ""
                    ymax_text = ""
                    bound_box_text_end = ""
                    object_text_end = ""

                # Continue this part even if there are no annotations for this image
                annotation_text_end = "</annotation>"

                xmlstring = (
                    annotation_text_start
                    + folder_text
                    + filename_text
                    + path_text
                    + sources_text
                    + size_text_start
                    + width_text
                    + height_text
                    + depth_text
                    + size_text_end
                    + segmented_text
                    + object_text_start
                    + name_text
                    + pose_text
                    + truncated_text
                    + difficult_text
                    + occluded_text
                    + bound_box_text_start
                    + xmin_text
                    + xmax_text
                    + ymin_text
                    + ymax_text
                    + bound_box_text_end
                    + object_text_end
                    + annotation_text_end
                )
                dom = xml.dom.minidom.parseString(xmlstring)
                pretty_xml_as_string = dom.toprettyxml()

                with open(output_file_path, "w") as f:
                    f.write(pretty_xml_as_string)

                return output_file_path

            else:
                # When there are more than one annotations for the image

                # print('test')
                annotation_text_start = "<annotation>"

                flder_lkp = str(df_smaller.loc[index]["img_folder"])
                if folder == True and flder_lkp != "":
                    folder_text = "<folder>" + flder_lkp + "</folder>"
                else:
                    folder_text = ""

                filename_text = (
                    "<filename>"
                    + str(df_smaller.loc[index]["img_filename"])
                    + "</filename>"
                )

                pth_lkp = str(df_smaller.loc[index]["img_path"])
                if path == True and pth_lkp != "":
                    path_text = "<path>" + pth_lkp + "</path>"
                else:
                    path_text = ""

                # db_lkp = str(df_smaller.loc[index]['Databases'])
                # if database == True and db_lkp != '':
                #    sources_text = '<source>'+'<database>'+ db_lkp +'</database>'+'</source>'
                # else:
                sources_text = ""

                size_text_start = "<size>"
                width_text = (
                    "<width>" + str(df_smaller.loc[index]["img_width"]) + "</width>"
                )
                height_text = (
                    "<height>" + str(df_smaller.loc[index]["img_height"]) + "</height>"
                )
                depth_text = (
                    "<depth>" + str(df_smaller.loc[index]["img_depth"]) + "</depth>"
                )
                size_text_end = "</size>"

                seg_lkp = str(df_smaller.loc[index]["ann_segmented"])
                if segmented == True and seg_lkp != "":
                    segmented_text = (
                        "<segmented>"
                        + str(df_smaller.loc[index]["ann_segmented"])
                        + "</segmented>"
                    )
                else:
                    segmented_text = ""

                xmlstring = (
                    annotation_text_start
                    + folder_text
                    + filename_text
                    + path_text
                    + sources_text
                    + size_text_start
                    + width_text
                    + height_text
                    + depth_text
                    + size_text_end
                    + segmented_text
                )

                for obj in range(len(df_smaller)):
                    object_text_start = "<object>"

                    name_text = (
                        "<name>" + str(df_smaller.loc[obj]["cat_name"]) + "</name>"
                    )
                    pose_text = (
                        "<pose>" + str(df_smaller.loc[obj]["ann_pose"]) + "</pose>"
                    )
                    truncated_text = (
                        "<truncated>"
                        + str(df_smaller.loc[obj]["ann_truncated"])
                        + "</truncated>"
                    )
                    difficult_text = (
                        "<difficult>"
                        + str(df_smaller.loc[obj]["ann_difficult"])
                        + "</difficult>"
                    )

                    # occ_lkp = str(df_smaller.loc[index]['Object Occluded'])
                    # if occluded==True and occ_lkp != '':
                    #    occluded_text = '<occluded>'+occ_lkp+'</occluded>'
                    # else:
                    occluded_text = ""

                    bound_box_text_start = "<bndbox>"

                    xmin_text = (
                        "<xmin>"
                        + str(int(df_smaller.loc[obj]["ann_bbox_xmin"]))
                        + "</xmin>"
                    )
                    xmax_text = (
                        "<xmax>"
                        + str(int(df_smaller.loc[obj]["ann_bbox_xmax"]))
                        + "</xmax>"
                    )
                    ymin_text = (
                        "<ymin>"
                        + str(int(df_smaller.loc[obj]["ann_bbox_ymin"]))
                        + "</ymin>"
                    )
                    ymax_text = (
                        "<ymax>"
                        + str(int(df_smaller.loc[obj]["ann_bbox_ymax"]))
                        + "</ymax>"
                    )

                    bound_box_text_end = "</bndbox>"
                    object_text_end = "</object>"
                    annotation_text_end = "</annotation>"
                    index = index + 1

                    xmlstring = (
                        xmlstring
                        + object_text_start
                        + name_text
                        + pose_text
                        + truncated_text
                        + difficult_text
                        + occluded_text
                        + bound_box_text_start
                        + xmin_text
                        + xmax_text
                        + ymin_text
                        + ymax_text
                        + bound_box_text_end
                        + object_text_end
                    )

                xmlstring = xmlstring + annotation_text_end
                dom = xml.dom.minidom.parseString(xmlstring)
                pretty_xml_as_string = dom.toprettyxml()

                with open(output_file_path, "w") as f:
                    f.write(pretty_xml_as_string)

                return output_file_path

        # Loop through all images in the dataframe and call voc_xml_file_creation for each one
        for file_title in list(set(self.dataset.df.img_filename)):

            file_name = Path(file_title)
            file_name = str(file_name.with_suffix(".xml"))
            file_path = str(Path(output_path, file_name))
            voc_file_path = voc_xml_file_creation(
                ds.df,
                file_title,
                segmented=segmented_,
                path=path_,
                database=database_,
                folder=folder_,
                occluded=occluded_,
                output_file_path=file_path,
            )
            output_file_paths.append(voc_file_path)

        return output_file_paths

    def ExportToYoloV5(
        self,
        output_path="training/labels",
        yaml_file="dataset.yaml",
        copy_images=False,
        use_splits=False,
        cat_id_index=None,
    ):
        """Writes annotation files to disk in YOLOv5 format and returns the paths to files.

        Args:

            output_path (str):
                This is where the annotation files will be written.
                If not-specified then the path will be derived from the .path_to_annotations and
                .name properties of the dataset object. If you are exporting images to train a model, the recommended path
                to use is 'training/labels'.
            yaml_file (str):
                If a file name (string) is provided, a YOLOv5 YAML file will be created with entries for the files
                and classes in this dataset. It will be created in the parent of the output_path directory.
                The recommended name for the YAML file is 'dataset.yaml'.
            copy_images (boolean):
                If True, then the annotated images will be copied to a directory next to the labels directory into
                a directory named 'images'. This will prepare your labels and images to be used as inputs to
                train a YOLOv5 model.
            use_splits (boolean):
                If True, then the images and annotations will be moved into directories based on the values in the split column.
                For example, if a row has the value split = "train" then the annotations for that row will be moved to directory
                /train. If a YAML file is specificied then the YAML file will use the splits to specify the folders user for the
                train, val, and test datasets.
            cat_id_index (int):
                Reindex the cat_id values so that that they start from an int (usually 0 or 1) and
                then increment the cat_ids to index + number of categories continuously.
                It's useful if the cat_ids are not continuous in the original dataset.
                Yolo requires the set of annotations to start at 0 when training a model.

        Returns:
            A list with 1 or more paths (strings) to annotations files. If a YAML file is created
            then the first item in the list will be the path to the YAML file.

        Examples:
            >>> dataset.export.ExportToYoloV5(output_path='training/labels',
            >>>     yaml_file='dataset.yaml', cat_id_index=0)
            ['training/dataset.yaml', 'training/labels/frame_0002.txt', ...]

        """
        ds = self.dataset

        # Inspired by https://github.com/aws-samples/groundtruth-object-detection/blob/master/create_annot.py
        yolo_dataset = ds.df.copy(deep=True)
        # Convert nan values in the split column from nan to '' because those are easier to work with with when building paths
        yolo_dataset.split = yolo_dataset.split.fillna("")

        # Create all of the paths that will be used to manage the files in this dataset
        path_dict = {}

        # The output path is the main path that will be used to create the other relative paths
        path = PurePath(output_path)
        path_dict["label_path"] = output_path
        # The /images directory should be next to the /labels directory
        path_dict["image_path"] = str(PurePath(path.parent, "images"))
        # The root directory is in parent of the /labels and /images directories
        path_dict["root_path"] = str(PurePath(path.parent))
        # The YAML file should be in root directory
        path_dict["yaml_path"] = str(PurePath(path_dict["root_path"], yaml_file))
        # The root directory will usually be next to the yolov5 directory.
        # Specify the relative path
        path_dict["root_path_from_yolo_dir"] = str(PurePath("../"))
        # If these default values to not match the users environment then they can manually edit the YAML file

        if copy_images:
            # Create the folder that the images will be copied to
            Path(path_dict["image_path"]).mkdir(parents=True, exist_ok=True)

        # Drop rows that are not annotated
        # Note, having zero annotates can still be considered annotated
        # in cases when are no objects in the image thats should be indentified
        yolo_dataset = yolo_dataset.loc[yolo_dataset["annotated"] == 1]

        # yolo_dataset["cat_id"] = (
        #     yolo_dataset["cat_id"].astype("float").astype(pd.Int32Dtype())
        # )

        yolo_dataset.cat_id = yolo_dataset.cat_id.replace(r"^\s*$", np.nan, regex=True)

        pd.to_numeric(yolo_dataset["cat_id"])

        # Convert empty bbox coordinates to nan to avoid math errors
        # If an image has no annotations then an empty label file will be created
        yolo_dataset.ann_bbox_xmin = yolo_dataset.ann_bbox_xmin.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_ymin = yolo_dataset.ann_bbox_ymin.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_width = yolo_dataset.ann_bbox_width.replace(
            r"^\s*$", np.nan, regex=True
        )
        yolo_dataset.ann_bbox_height = yolo_dataset.ann_bbox_height.replace(
            r"^\s*$", np.nan, regex=True
        )

        if cat_id_index != None:
            assert isinstance(cat_id_index, int), "cat_id_index must be an int."
            _ReindexCatIds(yolo_dataset, cat_id_index)

        yolo_dataset["center_x_scaled"] = (
            yolo_dataset["ann_bbox_xmin"] + (yolo_dataset["ann_bbox_width"] * 0.5)
        ) / yolo_dataset["img_width"]
        yolo_dataset["center_y_scaled"] = (
            yolo_dataset["ann_bbox_ymin"] + (yolo_dataset["ann_bbox_height"] * 0.5)
        ) / yolo_dataset["img_height"]
        yolo_dataset["width_scaled"] = (
            yolo_dataset["ann_bbox_width"] / yolo_dataset["img_width"]
        )
        yolo_dataset["height_scaled"] = (
            yolo_dataset["ann_bbox_height"] / yolo_dataset["img_height"]
        )
        yolo_dataset[
            [
                "cat_id",
                "center_x_scaled",
                "center_y_scaled",
                "width_scaled",
                "height_scaled",
            ]
        ]

        # Create folders to store annotations
        if output_path == None:
            dest_folder = PurePath(
                ds.path_to_annotations, yolo_dataset.iloc[0].img_folder
            )
        else:
            dest_folder = output_path

        os.makedirs(dest_folder, exist_ok=True)

        unique_images = yolo_dataset["img_filename"].unique()
        output_file_paths = []

        for img_filename in unique_images:
            df_single_img_annots = yolo_dataset.loc[
                yolo_dataset.img_filename == img_filename
            ]
            
            basename, _ = os.path.splitext(img_filename)
            annot_txt_file = basename + ".txt"
            # Use the value of the split collumn to create a directory
            # The values should be train, val, test or ''
            if use_splits:
                split_dir = df_single_img_annots.iloc[0].split
            else:
                split_dir = ""
            destination = str(PurePath(dest_folder, split_dir, annot_txt_file))
            Path(
                dest_folder,
                split_dir,
            ).mkdir(parents=True, exist_ok=True)

            df_single_img_annots.to_csv(
                destination,
                index=False,
                header=False,
                sep=" ",
                float_format="%.4f",
                columns=[
                    "cat_id",
                    "center_x_scaled",
                    "center_y_scaled",
                    "width_scaled",
                    "height_scaled",
                ],
            )
            output_file_paths.append(destination)

            if copy_images:
                source_image_path = str(
                    Path(
                        ds.path_to_annotations,
                        df_single_img_annots.iloc[0].img_folder,
                        df_single_img_annots.iloc[0].img_filename,
                    )
                )

                current_file = Path(source_image_path)
                assert (
                    current_file.is_file
                ), f"File does not exist: {source_image_path}. Check img_folder column values."
                Path(path_dict["image_path"], split_dir).mkdir(
                    parents=True, exist_ok=True
                )
                shutil.copy(
                    str(source_image_path),
                    str(PurePath(path_dict["image_path"], split_dir, img_filename)),
                )

        # Create YAML file
        if yaml_file:
            # Make a set with all of the different values of the split column
            splits = set(yolo_dataset.split)
            # Build a dict with all of the values that will go into the YAML file
            dict_file = {}
            dict_file["path"] = path_dict["root_path_from_yolo_dir"]

            # If train is one of the splits, append train to path
            if use_splits and "train" in splits:
                dict_file["train"] = str(PurePath(path_dict["image_path"], "train"))
            else:
                dict_file["train"] = path_dict["image_path"]

            # If val is one of the splits, append val to path
            if use_splits and "val" in splits:
                dict_file["val"] = str(PurePath(path_dict["image_path"], "val"))
            else:
                # If there is no val split, use the train split as the val split
                dict_file["val"] = dict_file["train"]

            # If test is one of the splits, make a test param and add test to the path
            if use_splits and "test" in splits:
                dict_file["test"] = str(PurePath(path_dict["image_path"], "test"))

            dict_file["nc"] = ds.analyze.num_classes
            dict_file["names"] = ds.analyze.classes

            # Save the yamlfile
            with open(path_dict["yaml_path"], "w") as file:
                documents = yaml.dump(dict_file, file)
                output_file_paths = [path_dict["yaml_path"]] + output_file_paths

        return output_file_paths

    def ExportToCoco(self, output_path=None, cat_id_index=None):
        """
        Writes COCO annotation files to disk (in JSON format) and returns the path to files.

        Args:
            output_path (str):
                This is where the annotation files will be written. If not-specified then the path will be derived from the path_to_annotations and
                name properties of the dataset object.
            cat_id_index (int):
                Reindex the cat_id values so that that they start from an int (usually 0 or 1) and
                then increment the cat_ids to index + number of categories continuously.
                It's useful if the cat_ids are not continuous in the original dataset.
                Some models like Yolo require starting from 0 and others like Detectron require starting from 1.

        Returns:
            A list with 1 or more paths (strings) to annotations files.

        Example:
            >>> dataset.exporter.ExportToCoco()
            ['data/labels/dataset.json']

        """
        # Copy the dataframe in the dataset so the original dataset doesn't change when you apply the export tranformations
        df = self.dataset.df.copy(deep=True)
        # Replace empty string values with NaN
        df = df.replace(r"^\s*$", np.nan, regex=True)
        pd.to_numeric(df["cat_id"])

        df["ann_iscrowd"] = df["ann_iscrowd"].fillna(0)

        if cat_id_index != None:
            assert isinstance(cat_id_index, int), "cat_id_index must be an int."
            _ReindexCatIds(df, cat_id_index)

        df_outputI = []
        df_outputA = []
        df_outputC = []
        list_i = []
        list_c = []
        json_list = []

        for i in range(0, df.shape[0]):
            images = [
                {
                    "id": df["img_id"][i],
                    "folder": df["img_folder"][i],
                    "file_name": df["img_filename"][i],
                    "path": df["img_path"][i],
                    "width": df["img_width"][i],
                    "height": df["img_height"][i],
                    "depth": df["img_depth"][i],
                }
            ]

            # Skip this if cat_id is na
            if not pd.isna(df["cat_id"][i]):

                annotations = [
                    {
                        "image_id": df["img_id"][i],
                        "id": df.index[i],
                        "segmented": df["ann_segmented"][i],
                        "bbox": [
                            df["ann_bbox_xmin"][i],
                            df["ann_bbox_ymin"][i],
                            df["ann_bbox_width"][i],
                            df["ann_bbox_height"][i],
                        ],
                        "area": df["ann_area"][i],
                        "segmentation": df["ann_segmentation"][i],
                        "iscrowd": df["ann_iscrowd"][i],
                        "pose": df["ann_pose"][i],
                        "truncated": df["ann_truncated"][i],
                        "category_id": int(df["cat_id"][i]),
                        "difficult": df["ann_difficult"][i],
                    }
                ]

                categories = [
                    {
                        "id": int(df["cat_id"][i]),
                        "name": df["cat_name"][i],
                        "supercategory": df["cat_supercategory"][i],
                    }
                ]

                # Check if the list is empty
                if list_c:
                    if categories[0]["id"] in list_c:
                        pass
                    else:
                        categories[0]["id"] = int(categories[0]["id"])
                        df_outputC.append(pd.DataFrame([categories]))
                elif not pd.isna(categories[0]["id"]):
                    categories[0]["id"] = int(categories[0]["id"])
                    df_outputC.append(pd.DataFrame([categories]))
                else:
                    pass
                list_c.append(categories[0]["id"])

            if list_i:
                if images[0]["id"] in list_i or np.isnan(images[0]["id"]):
                    pass
                else:
                    df_outputI.append(pd.DataFrame([images]))
            elif ~np.isnan(images[0]["id"]):
                df_outputI.append(pd.DataFrame([images]))
            else:
                pass
            list_i.append(images[0]["id"])

            # If the class id is blank, then there is no annotation to add
            if not pd.isna(categories[0]["id"]):
                df_outputA.append(pd.DataFrame([annotations]))

        mergedI = pd.concat(df_outputI, ignore_index=True)
        mergedA = pd.concat(df_outputA, ignore_index=True)
        mergedC = pd.concat(df_outputC, ignore_index=True)

        resultI = mergedI[0].to_json(orient="split", default_handler=str)
        resultA = mergedA[0].to_json(orient="split", default_handler=str)
        resultC = mergedC[0].to_json(orient="split", default_handler=str)

        parsedI = json.loads(resultI)
        del parsedI["index"]
        del parsedI["name"]
        parsedI["images"] = parsedI["data"]
        del parsedI["data"]

        parsedA = json.loads(resultA)
        del parsedA["index"]
        del parsedA["name"]
        parsedA["annotations"] = parsedA["data"]
        del parsedA["data"]

        parsedC = json.loads(resultC)
        del parsedC["index"]
        del parsedC["name"]
        parsedC["categories"] = parsedC["data"]
        del parsedC["data"]

        parsedI.update(parsedA)
        parsedI.update(parsedC)
        json_output = parsedI

        if output_path == None:
            output_path = Path(
                self.dataset.path_to_annotations, (self.dataset.name + ".json")
            )

        with open(output_path, "w") as outfile:
            json.dump(obj=json_output, fp=outfile, indent=4)
        return [str(output_path)]
