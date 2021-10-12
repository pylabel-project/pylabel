import json
import pandas as pd
import pandas as pd
import xml.etree.ElementTree as ET
import xml.dom.minidom

class Export:
    #def  __init__(self, df):
    #    self.df = df 

    def ExportToVoc(self, data, segmented_=False, path_=False, database_=False, folder_=False, occluded_=False, write_to_file_=True, output_file_path_ = 'pascal_voc.xml'):
        def voc_xml_file_creation(file_name, data, segmented=True, path=True, database=True, folder=True, occluded=True, write_to_file=False, output_file_path = 'pascal_voc.xml'):
            '''Note: the function will print no tags where the value consists of an empty string. 
            Required Parameter is the filename where all of the information to be converted is in a DataFrame (data param).
            Optional Parameters: Do you want to include Pascal VOC tags for your annotation for
                segmented, path, database, folder, or occluded? This often depends on the Pascal version.
            Optional Parameters: Do you want to write to file? What do you want the output file name to be?'''
            
            index = 0
            
            df_smaller = data[data['img_filename'] == file_name].reset_index()
            
            if len(df_smaller) == 1:
                #print('test')
                annotation_text_start = '<annotation>'

                flder_lkp = str(df_smaller.loc[index]['img_folder'])
                if folder==True and flder_lkp != '':
                    folder_text = '<folder>'+flder_lkp+'</folder>'
                else:
                    folder_text = ''
                    
                filename_text = '<filename>'+str(df_smaller.loc[index]['img_filename'])+'</filename>'
                
                pth_lkp = str(df_smaller.loc[index]['img_path'])
                if path == True and pth_lkp != '':
                    path_text = '<path>'+ pth_lkp +'</path>'
                else:
                    path_text = ''
                    
                #db_lkp = str(df_smaller.loc[index]['Databases'])
                #if database == True and db_lkp != '':
                #    sources_text = '<source>'+'<database>'+ db_lkp +'</database>'+'</source>'
                #else:
                sources_text = ''
                
                size_text_start = '<size>'
                width_text = '<width>'+str(df_smaller.loc[index]['img_width'])+'</width>'
                height_text = '<height>'+str(df_smaller.loc[index]['img_height'])+'</height>'
                depth_text = '<depth>'+str(df_smaller.loc[index]['img_depth'])+'</depth>'
                size_text_end = '</size>'
                
                seg_lkp = str(df_smaller.loc[index]['ann_segmented'])
                if segmented == True and seg_lkp != '':
                    segmented_text = '<segmented>'+str(df_smaller.loc[index]['ann_segmented'])+'</segmented>'
                else:
                    segmented_text = ''

                object_text_start = '<object>'

                name_text = '<name>'+str(df_smaller.loc[index]['cat_name'])+'</name>'
                pose_text = '<pose>'+str(df_smaller.loc[index]['ann_pose'])+'</pose>'
                truncated_text = '<truncated>'+str(df_smaller.loc[index]['ann_truncated'])+'</truncated>'
                difficult_text = '<difficult>'+str(df_smaller.loc[index]['ann_difficult'])+'</difficult>'
                
                #occ_lkp = str(df_smaller.loc[index]['Object Occluded'])
                #if occluded==True and occ_lkp != '':
                #    occluded_text = '<occluded>'+occ_lkp+'</occluded>'
                #else:
                occluded_text = ''

                bound_box_text_start = '<bndbox>'

                xmin_text = '<xmin>'+str(df_smaller.loc[index]['ann_bbox_xmin'])+'</xmin>'
                xmax_text = '<xmax>'+str(df_smaller.loc[index]['ann_bbox_xmax'])+'</xmax>'
                ymin_text = '<ymin>'+str(df_smaller.loc[index]['ann_bbox_ymin'])+'</ymin>'
                ymax_text = '<ymax>'+str(df_smaller.loc[index]['ann_bbox_ymax'])+'</ymax>'

                bound_box_text_end = '</bndbox>'
                object_text_end = '</object>'
                annotation_text_end = '</annotation>'
                        
                xmlstring = annotation_text_start + folder_text  +filename_text  + \
                    path_text  + sources_text + size_text_start + width_text  + \
                    height_text  + depth_text  + size_text_end + segmented_text  + \
                    object_text_start + name_text  + pose_text  +truncated_text + \
                    difficult_text + occluded_text + bound_box_text_start  +xmin_text  + \
                    xmax_text  +ymin_text  +ymax_text  +bound_box_text_end  + \
                    object_text_end  + annotation_text_end
                dom = xml.dom.minidom.parseString(xmlstring)
                pretty_xml_as_string = dom.toprettyxml()
                
                if write_to_file == True:
                    with open(output_file_path, "w") as f:
                        f.write(pretty_xml_as_string)  
                
                return(pretty_xml_as_string)
            
            else:

                #print('test')
                annotation_text_start = '<annotation>'
                
                flder_lkp = str(df_smaller.loc[index]['img_folder'])
                if folder==True and flder_lkp != '':
                    folder_text = '<folder>'+flder_lkp+'</folder>'
                else:
                    folder_text = ''
                
                
                filename_text = '<filename>'+str(df_smaller.loc[index]['img_filename'])+'</filename>'
                
                pth_lkp = str(df_smaller.loc[index]['img_path'])
                if path == True and pth_lkp != '':
                    path_text = '<path>'+ pth_lkp +'</path>'
                else:
                    path_text = ''
                
                #db_lkp = str(df_smaller.loc[index]['Databases'])
                #if database == True and db_lkp != '':
                #    sources_text = '<source>'+'<database>'+ db_lkp +'</database>'+'</source>'
                #else:
                sources_text = ''
                    
                size_text_start = '<size>'
                width_text = '<width>'+str(df_smaller.loc[index]['img_width'])+'</width>'
                height_text = '<height>'+str(df_smaller.loc[index]['img_height'])+'</height>'
                depth_text = '<depth>'+str(df_smaller.loc[index]['img_depth'])+'</depth>'
                size_text_end = '</size>'
                
                seg_lkp = str(df_smaller.loc[index]['ann_segmented'])
                if segmented == True and seg_lkp != '':
                    segmented_text = '<segmented>'+str(df_smaller.loc[index]['ann_segmented'])+'</segmented>'
                else:
                    segmented_text = ''

                xmlstring = annotation_text_start + folder_text  +filename_text  + \
                        path_text  + sources_text + size_text_start + width_text  + \
                        height_text  + depth_text  + size_text_end + segmented_text
                
                for obj in range(len(df_smaller)):
                    object_text_start = '<object>'

                    name_text = '<name>'+str(df_smaller.loc[index]['cat_name'])+'</name>'
                    pose_text = '<pose>'+str(df_smaller.loc[index]['ann_pose'])+'</pose>'
                    truncated_text = '<truncated>'+str(df_smaller.loc[index]['ann_truncated'])+'</truncated>'
                    difficult_text = '<difficult>'+str(df_smaller.loc[index]['ann_difficult'])+'</difficult>'
                    
                    #occ_lkp = str(df_smaller.loc[index]['Object Occluded'])
                    #if occluded==True and occ_lkp != '':
                    #    occluded_text = '<occluded>'+occ_lkp+'</occluded>'
                    #else:
                    occluded_text = ''

                    bound_box_text_start = '<bndbox>'

                    xmin_text = '<xmin>'+str(df_smaller.loc[index]['ann_bbox_xmin'])+'</xmin>'
                    xmax_text = '<xmax>'+str(df_smaller.loc[index]['ann_bbox_xmax'])+'</xmax>'
                    ymin_text = '<ymin>'+str(df_smaller.loc[index]['ann_bbox_ymin'])+'</ymin>'
                    ymax_text = '<ymax>'+str(df_smaller.loc[index]['ann_bbox_ymax'])+'</ymax>'

                    bound_box_text_end = '</bndbox>'
                    object_text_end = '</object>'
                    annotation_text_end = '</annotation>'
                    index = index + 1

                    
                
                    xmlstring = xmlstring + object_text_start + name_text  + pose_text  +truncated_text + \
                        difficult_text + occluded_text + bound_box_text_start  +xmin_text  + \
                        xmax_text  +ymin_text  +ymax_text  +bound_box_text_end  + \
                        object_text_end  

                xmlstring = xmlstring + annotation_text_end
                dom = xml.dom.minidom.parseString(xmlstring)
                pretty_xml_as_string = dom.toprettyxml()
                
                if write_to_file == True:
                    with open(output_file_path, "w") as f:
                        f.write(pretty_xml_as_string)  

                return(pretty_xml_as_string)

        #Loop through all images in the dataframe and call voc_xml_file_creation for each one
        for file_title in list(set(data.img_filename)):
            path2 = output_file_path_.replace('.','_')+'_'+file_title.replace('.','_')+'.xml'
            #print(path2)
            voc_xml_file_creation(file_title, data, segmented=segmented_, path=path_, database=database_, folder=folder_, occluded=occluded_, write_to_file=write_to_file_, output_file_path=str(path2))
            
        return()