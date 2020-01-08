from dlib_landmark import land_marks,get_landmark_dict
import numpy as np
import os
import cv2
from tqdm import tqdm
import pandas as pd
data_path="../Processed_data/Olulu_Casia_one_subject_last_three"
#data_path="../Processed_data/Process_deneme"


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
    #print(class_files)
    my_file=open(str(element)+".csv", "a")
    i=0
    for class_elem in class_files:
        #print(class_elem)
        single_class_path=os.path.join(class_path, class_elem)
        image_files=os.listdir(single_class_path)
        image_files.sort()
        #print(single_class_files)
        #last_num_elements=single_class_files[-3:len(single_class_files)]

        #print(image_files)
        for image in image_files:

             full_image_path=os.path.join(single_class_path, image)
             image=cv2.imread(full_image_path)
             #if image is not None:
             landmark_dict,_,_=land_marks(image,landmark_dict)
             landmark_dict["img_path"]=full_image_path
             landmark_dict["class"]=i

             df = pd.DataFrame(landmark_dict,index=[0])#,index=[0])#index=[0] denmezse hata veriyor
             #print(df)
             df.to_csv(my_file, header=False,index=False)
        i+=1
    my_file.close()



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
