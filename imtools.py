import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import PIL

#   Maximum Intensity Projection
def MIP(im_collection):
    im_max = np.zeros_like(im_collection[0]) # something like (1024, 1024)

    for i in range(len(im_collection)):
        im_max = np.fmax(im_max, im_collection[i])
    return im_max

def get_imlist(path):
    #get all jpg-filename in directly  specified
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.png')]


def imresize(im,sz):
    #resize image by PIL
    pil_im = Image.fromayyar(np.uint8(im))
    return np.array(pil_im.resize(sz))


def histeq(img, nbr_bins=256):
    #histogram equalization
    #obtain histogram of image
    fl = img.flatten()
    im = fl[np.where(fl != 0)]
    imhist, bins = np.histogram(im, nbr_bins, normed=True)
    cdf = imhist.cumsum()
    #normalization
    cdf = 255 * cdf / cdf[-1]
    #inear interpolation of cdf
    im2 = np.interp(fl,bins[:-1],cdf)
    #plt.imshow(Image.fromarray(im2.reshape(img.shape)))

    return im2.reshape(img.shape), cdf


def compute_average(imlist):
    #compute ImageList-average

    averageim = np.array(Image.open(imlist[0]), 'f')
    for imname in imlist[1:]:
        try:
            averageim += np.array(Image.open(imname))
        except:
            print (imname + '...skipped')
        averageim /= len(imlist)

    #convert uint8
    return np.array(averageim, 'uint8')


def draw_text_in_image(img, text, position_x, position_y):
    draw = PIL.ImageDraw.Draw(img)
    draw.font = PIL.ImageFont.truetype(
    "C:\Windows\Fonts\Calibri.ttf", 42)

    #img_size = np.array(img.size)
    #txt_size = np.array(draw.font.getsize(text))
    pos = (position_y, position_x)

    draw.text(pos, text, (255, 0, 255, 255))
    return img

