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
        scaleFactor=scale_factor,
        minNeighbors=5,
        minSize=(20, 20)
    )

    print("{} faces found".format(len(faces)))
    return faces      

def image_flip(image_path):
    try:
        image_obj = Image.open('temp/half/' + image_path)
        flipped_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_image.save('temp/half/flip_{}'.format(image_path))
    except:
        pass

def image_concat(image1, image2, image):
    final = Image.new('RGB', (image1.width + image2.width, image1.height))
    final.paste(image1, (0,0))
    final.paste(image2, (image1.width, 0))
    final.save('temp/shronk/{}'.format(image))

def temp_clear(directory):
    files = glob.glob('temp/' + directory + '*')
    for f in files:
        os.remove(f)

def folder_create(name):
    if not os.path.exists(name):
        os.makedirs(name)

image_path = sys.argv[1]
cascade_path = sys.argv[2]
scale_factor = float(sys.argv[3])

folder_create('temp/')
folder_create('temp/half/')
folder_create('temp/shronk/')

initial_image = cv2.imread(image_path)
faces = face_detect(image_path,cascade_path)

print("{} faces found".format(len(faces)))

for x,y,w,h in faces:
    roi_color_left = initial_image[y: y + h, x: x + w//2]
    cv2.imwrite("temp/half/{}x{}x{}x{}.jpg".format(x,y,w,h), roi_color_left)

temp_list = os.listdir('temp/half/')

for image in temp_list:
    image_flip(image)

    image1 = Image.open('temp/half/' + image)
    image2 = Image.open('temp/half/flip_' + image)
    image_concat(image1, image2, image)     
    

temp_list = os.listdir('temp/shronk/')

for x,y,w,h in faces:
    path = str(x) +'x'+ str(y) +'x'+ str(w) +'x'+ str(h) + '.jpg'
    s_img = cv2.imread('temp/shronk/' + path, -1)
    s_img = cv2.cvtColor(s_img, cv2.COLOR_RGB2RGBA).copy()
    y1, y2 = y, y + s_img.shape[0]
    x1, x2 = x, x + s_img.shape[1]

    alpha_s = s_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(0, 3):
        initial_image[y1:y2, x1:x2, c] = (alpha_s * s_img[:, :, c] +
                                  alpha_l * initial_image[y1:y2, x1:x2, c])
cv2.imwrite('final/{}x{}.jpg'.format(w,h), initial_image)
temp_clear('half/')
temp_clear('shronk/')


