import bbox_visualizer as bbv
import cv2
from PIL import Image
from pathlib import Path

class Visualize:
    def  __init__(self, dataset=None):
        self.dataset = dataset

    """Functions to visualize inspect images and annotations."""
    def ShowBoundingBoxes(self, img_id:int=0, img_filename:str="") -> Image:
        """Enter a filename or index number and return the image with the bounding boxes drawn."""

        ds = self.dataset

        #Handle cases where user enters image name in default field
        if type(img_id) == str:
            img_filename = img_id

        if img_filename == "": 
            df_single_img_annots = ds.df.loc[ds.df.img_id == img_id]
        else:
            df_single_img_annots = ds.df.loc[ds.df.img_filename == img_filename]

        full_image_path = str(Path(ds.path_to_annotations, df_single_img_annots.iloc[0].img_folder, df_single_img_annots.iloc[0].img_filename))
        img = cv2.imread(str(full_image_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        labels = []
        bboxes = []

        for index, row in df_single_img_annots.iterrows():
            labels.append(row['cat_name'])
            bboxes.append([int(row['ann_bbox_xmin']),int(row['ann_bbox_ymin']),int(row['ann_bbox_xmax']),int(row['ann_bbox_ymax'])])

        img_with_boxes = bbv.draw_multiple_rectangles(img, bboxes)
        img_with_boxes = bbv.add_multiple_labels(img_with_boxes, labels, bboxes)

        rendered_img = Image.fromarray(img_with_boxes)
        #rendered_img.save("bbox-visualizer/jpeg.jpg")
        return rendered_img

