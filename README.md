# PyLabel 

```pip install pylabel```

PyLabel is a Python package to help computer vision practitioners get labelled data sets ready to be used in deep learning models. The core functionality is to translate bounding box annotations between different formats-for example, from coco to yolo. 

## PyLabel capabilities: 
-	**Import/Export:** Translate annotation formats. PyLabel currently supports Coco, VOC, and Yolo bounding box formats. The task of converting formats can be accomplished with 2 commands--import and export. 
    ```
    from pylabel import importer
    dataset = importer.ImportCoco(path_to_annotations)
    dataset.ExportToVOC()
    ```
-	**Analyze:** PyLabel can help you explore your visual datasets be providing summary statistics such as the count of images and classes to help you identify class imbalances. 
-	**Visualize:** Render images from your dataset with bounding boxes overlaid so you can confirm the accuracy of the annotations. PyLabel uses the [bbox-visualizer](https://github.com/shoumikchow/bbox-visualizer) package to draw bounding boxes. 
-	**Split (Coming Soon):** Spilt image datasets into train, test, and val with stratification to get consistent class distribution across the split datasets.  

## Sample Notebooks
See PyLabel in action in these [sample Jupyter notebooks](https://github.com/pylabel-project/samples/).<br>
Open directly in Google Colab: 
- [coco2voc.ipynb](https://githubtocolab.com/pylabel-project/samples/blob/main/coco2voc.ipynb)
- [coco2yolov5.ipynb](https://githubtocolab.com/pylabel-project/samples/blob/main/coco2yolov5.ipynb)
- [voc2coco.ipynb](https://githubtocolab.com/pylabel-project/samples/blob/main/voc2coco.ipynb)
- [yolo2coco.ipynb](https://githubtocolab.com/pylabel-project/samples/blob/main/yolo2coco.ipynb)
- [yolo2voc.ipynb](https://githubtocolab.com/pylabel-project/samples/blob/main/yolo2voc.ipynb)

## About PyLabel 
PyLabel is being developed by Jeremy Fraenkel, Alex Heaton, and Derek Topper as the Capstope project for the Master of Information and Data Science (MIDS) at the UC Berkeley School of Information. If you have any questions or feedback please [create an issue](https://github.com/pylabelalpha/package/issues). Please let us know how we can make PyLabel more useful. 