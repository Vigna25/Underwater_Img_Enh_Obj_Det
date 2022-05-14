#!/usr/bin/env python
# coding: utf-8

# # 0. Image Enhancement

# In[1]:


get_ipython().system('pip install scikit-image')
get_ipython().system('pip install opencv-python')


# In[2]:


get_ipython().system('pip install opencv-contrib-python ')


# In[2]:


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

# In[ ]:


get_ipython().system('pip install opencv-python')


# In[ ]:


# Import opencv
import cv2 

# Import uuid
import uuid

# Import Operating System
import os

# Import time
import time


# # 2. Define Images to Collect

# In[ ]:


labels = ['Fishes', 'Submarines', 'Torpedoes']
#number_imgs = 5


# # 3. Setup Folders 

# In[ ]:


IMAGES_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'collectedimages')


# In[ ]:


if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        get_ipython().system('mkdir -p {IMAGES_PATH}')
    if os.name == 'nt':
         get_ipython().system('mkdir {IMAGES_PATH}')
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        get_ipython().system('mkdir {path}')


# # 4. Capture Images

# In[ ]:


#for label in labels:
    #cap = cv2.VideoCapture(0)
    #print('Collecting images for {}'.format(label))
    #time.sleep(5)
    #for imgnum in range(number_imgs):
        #print('Collecting image {}'.format(imgnum))
        #ret, frame = cap.read()
        #imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        #cv2.imwrite(imgname, frame)
        #cv2.imshow('frame', frame)
        #time.sleep(2)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break
#cap.release()
#cv2.destroyAllWindows()


# # 5. Image Labelling

# In[ ]:


get_ipython().system('pip install --upgrade pyqt5 lxml')


# In[ ]:


LABELIMG_PATH = os.path.join('Tensorflow', 'labelimg')


# In[ ]:


if not os.path.exists(LABELIMG_PATH):
    get_ipython().system('mkdir {LABELIMG_PATH}')
    get_ipython().system('git clone https://github.com/tzutalin/labelImg {LABELIMG_PATH}')


# In[ ]:


if os.name == 'posix':
    get_ipython().system('make qt5py3')
if os.name =='nt':
    get_ipython().system('cd {LABELIMG_PATH} && pyrcc5 -o libs/resources.py resources.qrc')


# In[ ]:


get_ipython().system('cd {LABELIMG_PATH} && python labelImg.py')


# # 6. Move them into a Training and Testing Partition

# # OPTIONAL - 7. Compress them for Colab Training

# In[ ]:


TRAIN_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'train')
TEST_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'test')
ARCHIVE_PATH = os.path.join('Tensorflow', 'workspace', 'images', 'archive.tar.gz')


# In[ ]:


get_ipython().system('tar -czf {ARCHIVE_PATH} {TRAIN_PATH} {TEST_PATH}')


# In[ ]:




