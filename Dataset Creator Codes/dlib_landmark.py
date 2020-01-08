import cv2
import numpy as np
import dlib

import matplotlib.pyplot as plt

import scipy.ndimage
from collections import OrderedDict
import data_augment

def align_face(image,landmarks): #aling face

    left_eye_range=range(36, 42)
    right_eye_range=range(42, 48)

    left_eye_x=0.
    left_eye_y=0.

    right_eye_x=0.
    right_eye_y=0.

    for n in left_eye_range:
        left_eye_x+= landmarks.part(n).x
        left_eye_y+= landmarks.part(n).y

    for n in right_eye_range:
        right_eye_x+= landmarks.part(n).x
        right_eye_y+= landmarks.part(n).y

    left_eye_x/=6.
    left_eye_y/=6.
    right_eye_x/=6.
    right_eye_y/=6.

    eye1_center=[left_eye_x,left_eye_y]
    eye2_center=[right_eye_x,right_eye_y]

    dY = eye1_center[1] - eye2_center[1]
    dX = eye1_center[0] - eye2_center[0]
    angle = np.degrees(np.arctan2(dY, dX)) - 180
    #print(angle)
    #if abs(angle)<20:
    rotated=scipy.ndimage.interpolation.rotate(image,angle)

    return rotated


def get_landmark_dict():
    landmark_dict= OrderedDict()
    landmark_dict["img_path"]=""
    landmark_dict["class"]=100
    for i in range(0,68):
        landmark_dict["point_"+str(i)+"x"]=500
        landmark_dict["point_"+str(i)+"y"]=500
    return landmark_dict


def land_marks(img,landmark_dict):
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detector(gray)
    #print(face)
    landmark_list=[]
    if len(face)==0:
        for n in range(0,68):
            x = 0
            y = 0
            landmark_list.append(x)
            landmark_list.append(y)
            landmark_dict["point_"+str(n)+"x"]=x
            landmark_dict["point_"+str(n)+"y"]=y

        return landmark_dict,np.zeros(136),False
    else:
        landmarks = predictor(gray, face[0])


        for n in range(0,landmarks.num_parts):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_list.append(x)
            landmark_list.append(y)
            landmark_dict["point_"+str(n)+"x"]=x
            landmark_dict["point_"+str(n)+"y"]=y

    return landmark_dict,landmark_list,True







"""
img = cv2.imread("P030_Anger_img1.jpeg")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = detector(gray)
for face in faces:
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    #cv2.rectangle(img, (x1, y1), (x2, y2),(0, 255, 0),3)

landmarks = predictor(gray, face)
#'num_parts', 'part', 'parts', 'rect']

print(landmarks.num_parts)
for n in range(0,landmarks.num_parts):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    cv2.circle(img, (x, y), 1, (255, 0, 0), -1)

cropped_img=img[y1:y2,x1:x2]


_,a,_=land_marks(img,get_landmark_dict())
print(len(a))
print(a)

plt.imshow(img)
plt.show()
"""
