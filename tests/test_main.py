import pytest
from pylabel import importer
import pandas as pd
import copy

# Test the importing
# Create a dataset
@pytest.fixture()
def coco_dataset():
    """Returns a dataset object imported from a COCO dataset"""
    # Specify path to the coco.json file
    path_to_annotations = "data/BCCD_Dataset.json"
    # Specify the path to the images (if they are in a different folder than the annotations)
    path_to_images = ""
    # Import the dataset into the pylable schema
    dataset = importer.ImportCoco(
        path_to_annotations, path_to_images=path_to_images, name="BCCD_coco"
    )
    return dataset


@pytest.fixture()
def unlabeled_dataset():
    """Returns a dataset object imported from a COCO dataset"""
    # Specify path to the coco.json file
    # Specify the path to the images (if they are in a different folder than the annotations)
    path_to_images = "data"

    # Import the dataset into the pylable schema
    dataset = importer.ImportImagesOnly(path=path_to_images)
    return dataset


@pytest.fixture()
def yolo_dataset():
    """Returns a dataset object imported from a COCO dataset"""

    path_to_annotations = "data/coco128/labels/train2017/"

    # Identify the path to get from the annotations to the images
    path_to_images = "../../images/train2017/"

    # Import the dataset into the pylable schema
    # Class names are defined here https://github.com/ultralytics/yolov5/blob/master/data/coco128.yaml
    yoloclasses = [
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    dataset = importer.ImportYoloV5(
        path=path_to_annotations,
        path_to_images=path_to_images,
        cat_names=yoloclasses,
        img_ext="jpg",
        name="coco128",
    )

    return dataset


@pytest.mark.parametrize(
    "dataset",
    [pytest.lazy_fixture("coco_dataset"), pytest.lazy_fixture("unlabeled_dataset")],
)
def test_num_images(dataset):
    assert isinstance(
        dataset.analyze.num_images, int
    ), "analyze.num_images should return an int"


def test_df_is_dataframe(coco_dataset):
    assert isinstance(
        coco_dataset.df, pd.DataFrame
    ), "dataset.df should be a pandas dataframe"


def test_num_classes(coco_dataset):
    assert isinstance(
        coco_dataset.analyze.num_classes, int
    ), "analyze.num_images should return an int"


def test_classes_coco(coco_dataset):
    assert isinstance(
        coco_dataset.analyze.classes, list
    ), "analyze.classes should return a list"


def test_classes_unlabeled(unlabeled_dataset):
    assert isinstance(
        unlabeled_dataset.analyze.classes, list
    ), "analyze.classes should return a list"


def test_export_coco(coco_dataset):
    path_to_coco_export = coco_dataset.export.ExportToCoco()
    assert isinstance(
        path_to_coco_export[0], str
    ), "ExportToCoco should return a list with one or more strings."


def test_ReindexCatIds(coco_dataset):
    # Check if the ReindexCatIds function is working by checking the
    # cat ids after the function is called. The cat ids should be continuous
    # starting with the index
    index = 5
    ds_copy = copy.deepcopy(coco_dataset)
    ds_copy.ReindexCatIds(index)

    assert (
        min(ds_copy.analyze.class_ids) == index
    ), "The min value should equal the index"

    assert (
        max(ds_copy.analyze.class_ids) == index + len(ds_copy.analyze.class_ids) - 1
    ), "ReindexCatIds: The max value should equal the index + number of classes"

    assert (
        list(range(index, index + len(ds_copy.analyze.class_ids)))
        == ds_copy.analyze.class_ids
    ), "ReindexCatIds: The class ids should be continuous"


# Add tests for non-labeled datasets
