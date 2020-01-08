"""

This code turns the dataset in to sub sets. These sets are in the format which Pytorch ImageFolder uses

"""
from dlib_facecrop import face_recog
import numpy as np
import os
import cv2
from tqdm import tqdm

data_path="../B_OriginalImg/B_OriginalImg/OriginalImg/VL/Strong"


subject_files=os.listdir(data_path)
subject_files.sort()

subject_files=np.array(subject_files)
np.random.seed(2)
np.random.shuffle(subject_files)



set_num=0


pbar=tqdm(subject_files)


for element in pbar:
    subject_path=os.path.join(data_path, element)
    class_files=os.listdir(subject_path)
    class_files.sort()
    print(element)
    for class_elem in class_files:
        single_class=os.path.join(subject_path, class_elem)
        single_class_files=os.listdir(single_class)
        single_class_files.sort()
        #print(single_class_files)
        last_num_elements=single_class_files[-3:len(single_class_files)]
        i=0
        for image in last_num_elements:
             i+=1
             full_image_path=os.path.join(single_class, image)
             image=cv2.imread(full_image_path)
             if image is not None:
                 image_write,face_recognized=face_recog(image)
                 if face_recognized==True:
                     image_write_path="Olulu_Casia_one_subject_last_three/Set"+str(set_num//8)+"/"+str(class_elem)
                     if not os.path.exists(image_write_path):
                         os.makedirs(image_write_path)
                     image_name=str(element)+str("_")+str(class_elem)+"_img"+str(i)+".jpeg"
                     image_write_full_path=os.path.join(image_write_path,image_name)
                     cv2.imwrite(image_write_full_path,image_write)
    set_num+=1
    print(set_num)

