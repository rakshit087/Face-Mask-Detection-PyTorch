import numpy as np
import cv2
import os
from tqdm import tqdm

base_dir = os.path.dirname(__file__)

def detect_face(image,name):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    for (x, y, w, h) in faces:
        roi_color = image[y:y + h, x:x + w]
        cv2.imwrite(name, roi_color)

count=0
for filename in tqdm(os.listdir(base_dir+"/images/")):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image = cv2.imread(base_dir+"/images/"+filename)
        name = "Masked_"+str(count)+".jpg"
        try:
            detect_face(image,base_dir+"/saved/"+name)
            count += 1
        except:
            pass
    else:
        continue
