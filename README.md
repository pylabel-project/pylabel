# PyLabel 

<a href="https://pypi.org/project/pylabel/">
<img alt="PyPI" src="https://img.shields.io/pypi/v/pylabel?color=gre">&nbsp;&nbsp;
<img src="https://img.shields.io/pypi/dm/pylabel?style=plastic"></a>
&nbsp;&nbsp;

<a href='https://pylabel.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/pylabel/badge/?version=latest' alt='Documentation Status' />
</a>
&nbsp;&nbsp;<a href="https://colab.research.google.com/github/pylabel-project/samples/blob/main/coco2voc.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
&nbsp;

<p><p>
PyLabel is a Python package to help you prepare image datasets for computer vision models including PyTorch and YOLOv5. It can translate bounding box annotations between different formats. (For example, COCO to YOLO.) And it includes an AI-assisted labeling tool that runs in a Jupyter notebook. 

-	**Translate:** Convert annotation formats with a single line of code: 
    ```
    importer.ImportCoco(path_to_annotations).ExportToYoloV5()
    ```
-	**Analyze:** PyLabel stores annotatations in a pandas dataframe so you can easily perform analysis on image datasets. 
-	**Split:** Divide image datasets into train, test, and val with stratification to get consistent class distribution.  <br><img src="https://raw.githubusercontent.com/pylabel-project/datasets_models/main/pylabel_assets/train_test_split.png" width=400>
-  **Label:** PyLabel also includes an image labeling tool that runs in a Jupyter notebook that can annotate images manually or perform automatic labeling using a pre-trained model.<br><br><img src="https://raw.githubusercontent.com/pylabel-project/datasets_models/main/pylabel_assets/pylaber_screenshot.png" width=400>
-	**Visualize:** Render images from your dataset with bounding boxes overlaid so you can confirm the accuracy of the annotations. 


## Tutorial Notebooks
See PyLabel in action in these [sample Jupyter notebooks](https://github.com/pylabel-project/samples/):<br>
- [Convert COCO to YOLO](https://github.com/pylabel-project/samples/blob/main/coco2yolov5.ipynb)
- [Convert COCO to VOC](https://github.com/pylabel-project/samples/blob/main/coco2voc.ipynb)
- [Convert VOC to COCO](https://github.com/pylabel-project/samples/blob/main/voc2coco.ipynb)
- [Convert YOLO to COCO](https://github.com/pylabel-project/samples/blob/main/yolo2coco.ipynb)
- [Convert YOLO to VOC](https://github.com/pylabel-project/samples/blob/main/yolo2voc.ipynb)
- [Import a YOLO YAML File](https://github.com/pylabel-project/samples/blob/main/yolo_with_yaml_importer.ipynb) 
- [Splitting Images Datasets into Train, Test, Val](https://github.com/pylabel-project/samples/blob/main/dataset_splitting.ipynb)
- [Labeling Tool Demo with AI Assisted Labeling](https://github.com/pylabel-project/samples/blob/main/pylabeler.ipynb)

Find more docs at https://pylabel.readthedocs.io. 

## About PyLabel 
PyLabel was developed by Jeremy Fraenkel, Alex Heaton, and Derek Topper as the Capstope project for the Master of Information and Data Science (MIDS) at the UC Berkeley School of Information. If you have any questions or feedback please [create an issue](https://github.com/pylabel-project/pylabel/issues). Please let us know how we can make PyLabel more useful. 
    
