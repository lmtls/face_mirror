import cv2
import sys
import os
import glob
import numpy as np
from PIL import Image

def face_detect(image_path, cascade_path):
    face_cascade = cv2.CascadeClassifier(cascade_path)
    initial_image = cv2.imread(image_path)
    image_grayscale = cv2.cvtColor(initial_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        image_grayscale,
        scaleFactor=1.05,
        minNeighbors=5,
        minSize=(20, 20)
    )

    print("{} faces found".format(len(faces)))
    return faces      

def image_flip(image_path, ):
    image_obj = Image.open('temp/' + image_path)
    flipped_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_image.save('temp/flip_{}'.format(image_path))

def image_concat(image1, image2):
    final = Image.new('RGB', (image1.width + image2.width, image1.height))
    final.paste(image1, (0,0))
    final.paste(image2, (image1.width, 0))
    final.save('temp/shronk_{}'.format(image1.filename[image1.filename.index('/') + 1:]))

def temp_clear():
    files = glob.glob('temp/*')
    for f in files:
        os.remove(f)

image_path = sys.argv[1]
cascade_path = sys.argv[2]
initial_image = cv2.imread(image_path)
faces = face_detect(image_path,cascade_path)

print("{} faces found".format(len(faces)))

for x,y,w,h in faces:
    roi_color_left = initial_image[y: y + h, x: x + w//2]
    cv2.imwrite("temp/{}x{}.jpg".format(w, h), roi_color_left)

for image in os.listdir('temp/'):
    image_flip(image)

temp_list = os.listdir('temp/')
for image in temp_list:
    try:
        image1 = Image.open('temp/' + image)
        image2 = Image.open('temp/flip_' + image)
        temp_clear()
        image_concat(image1, image2)     
    except FileNotFoundError:
        pass

temp_list = os.listdir('temp/')
faces = zip(faces, temp_list)

for cords,path in faces:
    s_img = cv2.imread('temp/' + path, -1)
    s_img = cv2.cvtColor(s_img, cv2.COLOR_RGB2RGBA).copy()
    x,y,w,h = cords[0],cords[1],cords[2],cords[3]
    y1, y2 = y, y + s_img.shape[0]
    x1, x2 = x, x + s_img.shape[1]

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        initial_image[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * initial_image[y1:y2, x1:x2, c])
cv2.imwrite('final/{}x{}.jpg'.format(w,h), initial_image)





