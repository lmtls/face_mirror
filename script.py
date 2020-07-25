import cv2
import sys
import os
import glob
import numpy as np
from PIL import Image

def image_flip(image_path, ):
    image_obj = Image.open('temp/' + image_path)
    flipped_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_image.save('temp/flip_{}'.format(image_path))

def image_concat(image1, image2):
    final = Image.new('RGB', (image1.width + image2.width, image1.height))
    final.paste(image1, (0,0))
    final.paste(image2, (image1.width, 0))
    final.save('final/shronk_{}'.format(image1.filename[image1.filename.index('/') + 1:]))


image_path = sys.argv[1]
cascade_path = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

image = cv2.imread(image_path)
image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(
    image_grayscale,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(20, 20)
)

print("{} faces found".format(len(faces)))

for x,y,w,h in faces:
    roi_color_left = image[y: y + h, x: x + w//2]
    cv2.imwrite("temp/{}x{}.jpg".format(w, h), roi_color_left)

for image in os.listdir('temp/'):
    image_flip(image)

temp_list = os.listdir('temp/')
for image in temp_list:
    try:
        image1 = Image.open('temp/' + image)
        image2 = Image.open('temp/flip_' + image)
        image_concat(image1, image2)
        
    except FileNotFoundError:
        pass


files = glob.glob('temp/*')
for f in files:
    os.remove(f)




