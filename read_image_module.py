from skimage.io import imread, imsave
from skimage.util import img_as_ubyte, img_as_float
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import image_enhancement
import os


images = os.listdir("./img read")
#print(images)
count = 1
for imagePath in images:
    #print("./img read/" + imagePath)
    image = imread("./img read/" + imagePath)

    image = Image.fromarray(image)
    image = image.convert('RGB')

    enhanced_image = image_enhancement.enhance_image(image)

    img = np.array(enhanced_image, dtype=np.uint8)

    imsave("./enh_img/EI_" + str(count) + ".jpg", img)
    count += 1