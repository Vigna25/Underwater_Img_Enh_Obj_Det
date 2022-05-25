import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import skimage.filters
import skimage.measure
from skimage.util import img_as_ubyte, img_as_float
from skimage.io import imread

def gray_world(image):

    image_grayworld = ((image * image.mean() /
                            image.mean(axis=(0, 1)))).clip(0, 255).astype('uint8')

    if image.shape[2] == 4:
        image_grayworld[:, :, 3] = 255
    return image_grayworld


def redChannelCompensation(image):
    floatImage = img_as_float(image)

    channelMeans = floatImage.mean(axis=(0, 1))

    redChannelCompensatedImage = floatImage


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

    return img_as_ubyte(redChannelCompensatedImage)


def gammaCorrection(image, gamma=1.0):
    return (image / 255) ** (gamma)


def enhance_image(image):

    a = skimage.measure.shannon_entropy(image)
    print("Original image entropy: " + str(a))

    image = np.array(image)
    redChannelCompensatedImage = redChannelCompensation(image)
    grayWorldImageWithoutCompensation = gray_world(image)
    grayWorldImage = gray_world(redChannelCompensatedImage)

    b = skimage.measure.shannon_entropy(grayWorldImage)
    print("Grayworld entropy: " + str(b))
    gammaCorrectedImage = gammaCorrection(grayWorldImage, 1.5)
    tempImage = img_as_float(skimage.filters.gaussian(img_as_float(grayWorldImage), 1, multichannel=False))

    sharpImage = img_as_float(img_as_float(grayWorldImage) - tempImage)
    sharpImage = img_as_ubyte((img_as_float(image) + sharpImage) / 2)
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

    laplacianWeight1 = img_as_float(abs(skimage.filters.laplace(sharpImageLightChannel, 3)))
    laplacianWeight2 = img_as_float(abs(skimage.filters.laplace(gammaCorrectedImageLightChannel, 3)))
    saliency1 = cv.saliency.StaticSaliencyFineGrained_create()

    (success, saliencyMap) = saliency1.computeSaliency(sharpImage)

    saliencyMap1 = img_as_float(img_as_ubyte(saliencyMap))

    saliency2 = cv.saliency.StaticSaliencyFineGrained_create()

    (success, saliencyMap) = saliency2.computeSaliency(img_as_ubyte(gammaCorrectedImage))

    saliencyMap2 = img_as_float(img_as_ubyte(saliencyMap))

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

    fig1, ax1 = plt.subplots(1, 2)
    ax1[0].imshow(image)
    ax1[0].set_title("Original Image")
    ax1[1].imshow(redChannelCompensatedImage)
    ax1[1].set_title("RedChannel Compensated Image")
    fig2, ax2 = plt.subplots(1, 2)
    ax2[0].imshow(grayWorldImage)
    ax2[0].set_title("Gray world Image after rcc")
    ax2[1].imshow(grayWorldImageWithoutCompensation)
    ax2[1].set_title("Gray world Image without rcc")
    fig3, ax3 = plt.subplots(1, 2)
    ax3[0].imshow(image)
    ax3[0].set_title("Original image")
    ax3[1].imshow(gammaCorrectedImage)
    ax3[1].set_title("Gamma corrected image")
    fig4, ax4 = plt.subplots(1, 2)
    ax4[0].imshow(sharpImage)
    ax4[0].set_title("Sharpened image")
    ax4[1].imshow(naiveFusionImage)
    ax4[1].set_title("Fusion image")
    plt.show()
    c = skimage.measure.shannon_entropy(img_as_ubyte(naiveFusionImage))
    print("Naive fusion image entropy: " + str(c))
    print(" ")
    return img_as_ubyte(naiveFusionImage)
