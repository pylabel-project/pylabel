import os
import zipfile

# Download the datasets during the setup phase so they are available when the tests are run

# Download sample coco dataset
os.makedirs("data", exist_ok=True)
# Download sample dataset
os.system(
    "wget 'https://github.com/pylabel-project/datasets_models/raw/main/BCCD/BCCD_coco.zip' -O data/BCCD_coco.zip"
)
with zipfile.ZipFile("data/BCCD_coco.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# Download sample yolo dataset
os.system(
    "wget 'https://github.com/pylabel-project/datasets_models/blob/main/coco128.zip?raw=true' -O data/coco128.zip"
)
with zipfile.ZipFile("data/coco128.zip", "r") as zip_ref:
    zip_ref.extractall("data")

# Download sample notebooks so they can be tested using nbmake
os.system("rm -rf samples")
os.system("git clone https://github.com/pylabel-project/samples.git")
os.system("cp samples/*.ipynb .")
