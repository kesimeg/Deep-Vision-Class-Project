"""
This code extracts facial landmarks from the dataset generated with the other dataset creator code.
Image paths, class id and facial landmarks of every image is written in to csv files. Every sub set has one csv.
"""
from dlib_landmark import land_marks,get_landmark_dict
import numpy as np
import os
import cv2
from tqdm import tqdm
import pandas as pd
data_path="../Processed_data/Olulu_Casia_one_subject_last_three"



cross_sets=os.listdir(data_path)
cross_sets.sort()

cross_sets=np.array(cross_sets)




set_num=0


pbar=tqdm(cross_sets)
landmark_dict=get_landmark_dict()

for element in pbar:
    class_path=os.path.join(data_path, element)
    class_files=os.listdir(class_path)
    class_files.sort()
    
    my_file=open(str(element)+".csv", "a")
    i=0
    for class_elem in class_files:
       
        single_class_path=os.path.join(class_path, class_elem)
        image_files=os.listdir(single_class_path)
        image_files.sort()
        
        #last_num_elements=single_class_files[-3:len(single_class_files)]

       
        for image in image_files:

             full_image_path=os.path.join(single_class_path, image)
             image=cv2.imread(full_image_path)
             
             landmark_dict,_,_=land_marks(image,landmark_dict)
             landmark_dict["img_path"]=full_image_path
             landmark_dict["class"]=i

             df = pd.DataFrame(landmark_dict,index=[0])
            
             df.to_csv(my_file, header=False,index=False)
        i+=1
    my_file.close()

