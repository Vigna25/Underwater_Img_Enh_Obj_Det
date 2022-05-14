import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters
from skimage.io import imread
from skimage.util import img_as_ubyte, img_as_float


def linearStretching(img, t1, t2):
    s1 = img.min()
    s2 = img.max()
    linImg = (img - s1) * ((t2 - t1) / (s2 - s1)) + t1
    return linImg


def gray_world(image):
    #  print(image.mean(axis=(0, 1)))
    # print(image.mean())
    image_grayworld = ((image * (image.mean() /
                                 image.mean(axis=(0, 1)))).
                       clip(0, 255).astype('uint8'))
    # for images having a transparency channel

    if image.shape[2] == 4:
        image_grayworld[:, :, 3] = 255
    return image_grayworld


def redChannelCompensation(image):
    floatImage = img_as_float(image)
    # print(floatImage.shape[0])
    channelMeans = floatImage.mean(axis=(0, 1))
    # print(channelMeans)
    redChannelCompensatedImage = floatImage
    # print(1 - redChannelCompensatedImage[0])
    #  print(redChannelCompensatedImage[0].shape)

    for i in range(0, redChannelCompensatedImage.shape[0]):

        for j in range(0, redChannelCompensatedImage.shape[1]):
            if redChannelCompensatedImage[i][j][1] <= redChannelCompensatedImage[i][j][2]:

                redChannelCompensatedImage[i][j][0] = (redChannelCompensatedImage[i][j][0] +
                                                       (3 * (channelMeans[2] - channelMeans[0]) * (
                                                               1 - redChannelCompensatedImage[i][j][0]) *
                                                        redChannelCompensatedImage[i][j][2]).clip(0, 1)).clip(0, 1)
            else:
                redChannelCompensatedImage[i][j][0] = (redChannelCompensatedImage[i][j][0] +
                                                       (3 * (channelMeans[1] - channelMeans[0]) * (
                                                               1 - redChannelCompensatedImage[i][j][0]) *
                                                        redChannelCompensatedImage[i][j][1]).clip(0, 1)).clip(0, 1)

    # print(redChannelCompensatedImage.mean(axis=(0, 1)))
    return img_as_ubyte(redChannelCompensatedImage)


def gammaCorrection(image, gamma=1.0):
    return (image / 255) ** (gamma)


def enhance_image(image):

    redChannelCompensatedImage = redChannelCompensation(image)
    grayWorldImage = gray_world(redChannelCompensatedImage)
    gammaCorrectedImage = gammaCorrection(grayWorldImage, 1.5)
    tempImage = img_as_float(skimage.filters.gaussian(img_as_float(grayWorldImage), 1, multichannel=False))
    sharpImage = img_as_float(img_as_float(grayWorldImage) - tempImage)
    contrastStretchedImage = linearStretching(sharpImage, 0, 0.7)
    sharpImage = img_as_ubyte((img_as_float(image) + contrastStretchedImage) / 2)
    sharpImageLab = skimage.color.rgb2lab(sharpImage)
    gammaCorrectedImageLab = skimage.color.rgb2lab(gammaCorrectedImage)
    sharpImageLightChannel = []
    gammaCorrectedImageLightChannel = []
    row, col = sharpImageLab.shape[0], sharpImageLab.shape[1]

    for i in range(0, row):
        eachrowList = []
        for j in range(0, col):
            eachrowList.append(gammaCorrectedImageLab[i][j][0])
        gammaCorrectedImageLightChannel.append(eachrowList)

    for i in range(0, row):
        eachrowList = []
        for j in range(0, col):
            eachrowList.append(sharpImageLab[i][j][0])
        sharpImageLightChannel.append(eachrowList)

    sharpImageLightChannel = np.array(sharpImageLightChannel)
    gammaCorrectedImageLightChannel = np.array(img_as_float(gammaCorrectedImageLightChannel))
    # print(sharpImageLightChannel.shape)
    # print(filteredImageLuminanceChannel.shape)
    # print(filteredImageLuminanceChannel)
    laplacianWeight1 = img_as_float(abs(skimage.filters.laplace(sharpImageLightChannel, 3)))
    laplacianWeight2 = img_as_float(abs(skimage.filters.laplace(gammaCorrectedImageLightChannel, 3)))
    saliency1 = cv.saliency.StaticSaliencyFineGrained_create()

    (success, saliencyMap) = saliency1.computeSaliency(sharpImage)
    # print(laplacianWeight1)
    saliencyMap1 = img_as_float(img_as_ubyte(saliencyMap))

    saliency2 = cv.saliency.StaticSaliencyFineGrained_create()

    (success, saliencyMap) = saliency2.computeSaliency(img_as_ubyte(gammaCorrectedImage))

    saliencyMap2 = img_as_float(img_as_ubyte(saliencyMap))

    # print(sharpImage[:, :, 2].shape)

    wsat1 = img_as_float(np.sqrt(((sharpImageLightChannel - sharpImage[:, :, 0]) ** 2 + (
            sharpImageLightChannel - sharpImage[:, :, 1]) ** 2 + (
                                          sharpImageLightChannel - sharpImage[:, :, 2]) ** 2) / 3))
    wsat2 = img_as_float(np.sqrt(((gammaCorrectedImageLightChannel - gammaCorrectedImage[:, :, 0]) ** 2 + (
            gammaCorrectedImageLightChannel - gammaCorrectedImage[:, :, 1]) ** 2 + (
                                          gammaCorrectedImageLightChannel - gammaCorrectedImage[:, :, 2]) ** 2) / 3))
    wk1 = laplacianWeight1 + wsat1 + saliencyMap1
    wk2 = laplacianWeight2 + wsat2 + saliencyMap2

    wk1Norm = (wk1 + 0.1) / (wk1 + wk2 + 0.2)
    wk2Norm = (wk2 + 0.1) / (wk1 + wk2 + 0.2)

    naiveFusionImage = []
    sharpImage = img_as_float(sharpImage)
    gammaCorrectedImage = img_as_float(gammaCorrectedImage)
    for i in range(0, row):
        eachrowList = []
        for j in range(0, col):
            eachTriplet = []
            for k in range(0, 3):
                eachTriplet.append(
                    (wk1Norm[i][j] * sharpImage[i][j][k] + wk2Norm[i][j] * gammaCorrectedImage[i][j][k]).clip(0, 1))
            eachrowList.append(eachTriplet)
        naiveFusionImage.append(eachrowList)

    return img_as_ubyte(naiveFusionImage)
