import os 
import zipfile
    
#Download the datasets during the setup phase so they are available when the tests are run

#Download sample coco dataset 
os.makedirs("data", exist_ok=True)
#Download sample dataset
os.system("wget 'https://github.com/pylabelalpha/notebook/blob/main/BCCD_coco.zip?raw=true' -O data/BCCD_coco.zip")
with zipfile.ZipFile("data/BCCD_coco.zip", 'r') as zip_ref:
    zip_ref.extractall("data")

#Download sample yolo dataset 
os.makedirs("data", exist_ok=True)
os.system("wget 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip' -O data/coco128.zip")
with zipfile.ZipFile("data/coco128.zip", 'r') as zip_ref:
   zip_ref.extractall("data")

#Download sample yolo dataset 
os.system("git clone https://github.com/pylabel-project/samples.git")

