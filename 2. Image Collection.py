# 0. Image Enhancement

get_ipython().system('pip install scikit-image')
get_ipython().system('pip install opencv-python')


get_ipython().system('pip install opencv-contrib-python ')


from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_float
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import image_enhancement
import os


images = os.listdir("./Images Raw")
#print(images)
count = 122
for imagePath in images:
    #print("./Images Raw/" + imagePath)
    image = imread("./Images Raw/" + imagePath)

    image = Image.fromarray(image)
    image = image.convert('RGB')

    enhanced_image = image_enhancement.enhance_image(image)

    img = np.array(enhanced_image, dtype=np.uint8)
    
    imsave("./Enhanced Images/EI_" + str(count) + ".jpg", img)
    count += 1


# # 1. Import Dependencies

get_ipython().system('pip install opencv-python')



# Import opencv
import cv2 

# Import uuid
import uuid

# Import Operating System
import os

# Import time
import time


# # 2. Define Images to Collect

labels = ['Fishes', 'Submarines', 'Torpedoes']

## 3. Setup Folders

IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')


if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        get_ipython().system('mkdir -p {IMAGES_PATH}')
    if os.name == 'nt':
         get_ipython().system('mkdir {IMAGES_PATH}')
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        get_ipython().system('mkdir {path}')


# 5. Image Labelling

get_ipython().system('pip install --upgrade pyqt5 lxml')

LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')

if not os.path.exists(LABELIMG_PATH):
    get_ipython().system('mkdir {LABELIMG_PATH}')
    get_ipython().system('git clone https://github.com/tzutalin/labelImg {LABELIMG_PATH}')

if os.name == 'posix':
    get_ipython().system('make qt5py3')
if os.name =='nt':
    get_ipython().system('cd {LABELIMG_PATH} && pyrcc5 -o libs/resources.py resources.qrc')

get_ipython().system('cd {LABELIMG_PATH} && python labelImg.py')


# 6. Move them into a Training and Testing Partition






