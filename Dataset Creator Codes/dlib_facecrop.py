import cv2
import numpy as np
import dlib

import matplotlib.pyplot as plt

import scipy.ndimage

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


def face_recog(img):
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = detector(gray)
    #print(face)
    if len(face)==0:
        return img,False
    else:
        landmarks = predictor(gray, face[0])




        rotated=align_face(img,landmarks)

        face2 = detector(rotated)
        if len(face2)==0:
            x1 = face[0].left()
            y1 = face[0].top()
            x2 = face[0].right()
            y2 = face[0].bottom()
            cropped_img=img[y1-10:y2+10,x1-10:x2+10]

            return cropped_img,True
        else:
            x1 = face2[0].left()
            y1 = face2[0].top()
            x2 = face2[0].right()
            y2 = face2[0].bottom()
            cropped_img=rotated[y1-10:y2+10,x1-10:x2+10]

            return cropped_img,True

"""
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


left_eye_range=range(36, 42)
right_eye_range=range(42, 48)

for n in right_eye_range:
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    #cv2.circle(img, (x, y), 4, (255, 0, 0), -1)

cropped_img=img[y1:y2,x1:x2]

plt.imshow(cropped_img)
plt.show()
"""


"""
img = cv2.imread('024.jpeg')
img,_=face_recog(img)
plt.imshow(img)
plt.show()

img_aug=data_augment.augment(img,5)

for image in img_aug:
    plt.imshow(image)
    plt.show()
"""
