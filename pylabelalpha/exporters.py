def ImportCoco(path):
    """
    This function takes the path to an xml file in coco format as input. 
    It returns a dataframe in schema used by pylable to store annotations. 
    """
    with open(path) as cocojson:
        annotations_json = json.load(cocojson)

    #Store the 3 sections of the json as seperate json arrays
    images = pd.json_normalize(annotations_json["images"])
    images.columns = 'img_' + images.columns

    annotations = pd.json_normalize(annotations_json["annotations"])
    annotations.columns = 'ann_' + annotations.columns

    categories = pd.json_normalize(annotations_json["categories"])
    categories.columns = 'cat_' + categories.columns

    df = annotations
    df[['ann_bbox_xmin','ann_bbox_ymax','ann_bbox_width','ann_bbox_height']] = pd.DataFrame(df.ann_bbox.tolist(), index= df.index)
    df.insert(8, 'ann_bbox_xmax', df['ann_bbox_xmin'] + df['ann_bbox_width'] )
    df.insert(10, 'ann_bbox_ymin', df['ann_bbox_ymax'] - df['ann_bbox_height'] )
    
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

    return df

    ##sdsdf
