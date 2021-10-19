# PyLabel 

```pip install pylabel```

PyLabel is a Python package to help computer vision practitioners get labelled data sets ready to be used in deep learning models. The core functionality is to translate bounding box annotations between different formats-for example, from coco to yolo. 

## PyLabel capabilities: 
-	**Import/Export:** Translate annotation formats. PyLabel currently supports Coco, VOC, and Yolo bounding box formats.
-	**Analyze:** PyLabel can help you explore your visual datasets be providing summary statistics such as the count of images and classes to help you identify class imbalances. 
-	**Visualize:** Render images from your dataset with bounding boxes overlaid so you can confirm the accuracy of the annotations. 
-	**Split (Coming Soon):** Spilt image datasets into train, test, and val with stratification to get consistent class distribution across the split datasets.  

## Sample Notebooks
See PyLabel in action in these [sample Jupyter notebooks](https://github.com/pylabelalpha/notebook).<br>
Open directly in Google Colab: 
- [coco2voc.ipynb](https://githubtocolab.com/pylabelalpha/notebook/blob/main/coco2voc.ipynb)
- [coco2yolov5.ipynb](https://githubtocolab.com/pylabelalpha/notebook/blob/main/coco2yolov5.ipynb)
- [voc2coco.ipynb](https://githubtocolab.com/pylabelalpha/notebook/blob/main/voc2coco.ipynb)
- [yolo2coco.ipynb](https://githubtocolab.com/pylabelalpha/notebook/blob/main/yolo2coco.ipynb)
- [yolo2voc.ipynb](https://githubtocolab.com/pylabelalpha/notebook/blob/main/yolo2voc.ipynb)

## About PyLabel 
PyLabel is being developed by Jeremy Fraenkel, Alex Heaton, and Derek Topper as the Capstope project for the Master of Information and Data Science (MIDS) at the UC Berkeley School of Information. If you have any questions or feedback please [create an issue](https://github.com/pylabelalpha/package/issues). Please let us know how we can make PyLabel more useful. 