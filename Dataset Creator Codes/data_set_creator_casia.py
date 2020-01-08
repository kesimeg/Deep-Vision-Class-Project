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
        #print(class_elem)
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


"""
for element in class_files:
    image_path=path_name+"/"+element
    files=os.listdir(image_path)
    files.sort()
    for image_name in files:

        full_image_path=image_path+"/"+image_name
        image=cv2.imread(full_image_path,0)
        #print(full_image_path)
        image=face_recog(image)
        if type(image)!=type(None):
            aug=data_augment.augment(image,number_of_copies,72)

            for i in range(0,number_of_copies):
                #cv2.imwrite("./ege3.tiff",aug[0][3]*255)



                name=image_name.split(".jpeg")
                    #print("./Casia_process/processed/"+element+"/"+name[0]+"_"+str(i)+".jpeg",type(image))#,image*255)
                cv2.imwrite("./Olulu-Casia_one_subject_augment/processed/"+element+"/"+name[0]+"_"+str(i)+".jpeg",aug[0][i]*255) #has to be converted to 0-255
                cv2.imwrite("./Olulu-Casia_one_subject_augment/lbp/"+element+"/"+name[0]+"_"+str(i)+".jpeg",aug[1][i]*255)
"""
