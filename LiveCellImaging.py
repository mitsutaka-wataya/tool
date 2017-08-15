# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


import glob
import os
from skimage.io import ImageCollection, imread
import numpy as np
from skimage.morphology import white_tophat, square, skeletonize
from skimage.filters import median
from skimage.measure import regionprops
from mahotas import cwatershed, distance
from mahotas.thresholding import otsu
from mahotas.labeled import label, labeled_size, remove_regions_where, relabel
import imtools
import matplotlib.pyplot as plt


class LiveCellImaging:
    def __init__(self, path, Vol, *stagePos):
        self.path = path
        self.Voltage = Vol
        if stagePos != ():
            self.stagePos = stagePos[0]
        else:
            self.stagePos = None
        self.files = self.fnames()
        
    def fnames(self):
        if self.stagePos != None:
            fnames = glob.glob(self.path +"/split/*/*" + self.Voltage + '_s%i_t????.tif' %(self.stagePos))
        else:
            fnames = glob.glob(self.path +"/split/*100V/*V_[0-9]????.tif")
        return fnames
        
    def to_MultiImage(self, interval):
        #loadingList = [os.path.join(self.path+"", name) for name in self.files]
        loadingList = self.files        
        return MultiImage(ImageCollection(loadingList, load_func=imread), interval)
      
      
class MultiImage:
    def __init__(self, imCollection, interval):
        if type(imCollection) == map:
            imCollection = list(imCollection)
        self.images = imCollection
        self.planeNum = len(imCollection)
        self.height = len(imCollection[0][0])
        self.width = len(imCollection[0][1])
        self.interval = interval
        self.timePoints = range(0, self.planeNum*self.interval, self.interval)
        self.duration = self.timePoints[-1]

    @staticmethod
    def tophat(image, squareSize=100):
        """
        Parameters
        ==========
        :img: gray scale image  
        
        This is tophat-function for 'map' (built in function).  
        
        Returns
        =======
        :img_tophat: tophat image  
        """
        img_tophat = white_tophat(image, selem=square(squareSize))
        return img_tophat
        
    @staticmethod
    def smooth(image, squareSize=15):
        return median(image, selem=square(squareSize))
        
    def tophatAllPlanes(self):
        return list(map(self.tophat, self.images))
        
    def smoothAllPlane(self):
        return list(map(self.smooth, self.images))

    def maximumIntensityProjection(self, *images):
        if images == ():
            im_max = np.zeros_like(self.images[0])
            for image in self.images:
                im_max = np.fmax(im_max, image)
        else:
            im_max = np.zeros_like(images[0][0])
            for image in self.images:
                im_max = np.fmax(im_max, image)
        return im_max



class MultiMyotubeImage(MultiImage):
    def __init__(self, imCollection, interval):
        MultiImage.__init__(self, imCollection, interval)
    
    @staticmethod
    def segmentTubes1(img_labeled, img_intensity):
        """
        Parameters
        ==========
        :img_labeled: labeled image before segmentation  
        
        img_intensity: intensity image  
        
        Returns
        =======
        labeld image segmented by Otsu's binarization after the histogram equalization by each region.
        """
        img_seg = np.zeros_like(img_intensity)    
        labels = np.unique(img_labeled.flatten())
        for cell_id in labels[1:]:
            bin_region = (img_labeled == cell_id)
            region = bin_region * img_intensity
            region, _ = imtools.histeq(region)
            # segmentation
            T = otsu(np.uint(region), ignore_zeros=True)
            img_seg += (np.uint(region) > T)
        labeled , counts = label(img_seg)        
        return labeled, counts
        
    @staticmethod
    def segmentTubes2(img_labeled, img_intensity):
        """
        Parameters
        ==========
        :img_labeled: labeled image before segmentation  
        
        img_intensity: intensity image  
        
        Returns
        =======
        labeld image segmented by Otsu's binarization by each region.  
        * This function do not perform histogram equalization by each region.  
        If you want to do that before the segmentation, use 'segementation1' function.
        """
        img_seg = np.zeros_like(img_intensity)    
        labels = np.unique(img_labeled.flatten())
        for cell_id in labels[1:]:
            bin_region = (img_labeled == cell_id)
            region = bin_region * img_intensity
            region, _ = imtools.histeq(region)
            # segmentation
            T = otsu(np.uint(region), ignore_zeros=True)
            img_seg += (np.uint(region) > T)
        labeled, counts = label(img_seg)        
        return labeled, counts
    
    @staticmethod        
    def segmentTubes3(img_labeled, img_intensity):
        """
        Parameters
        ==========
        :img_labeled: labeled image before segmentation  
        
        img_intensity: intensity image  
        
        Returns
        =======
        labeld image segmented by mean-intensity by each region.
        * This function perform histogram equalization by each region before segmentation. 
        """
        img_seg = np.zeros_like(img_intensity)    
        labels = np.unique(img_labeled.flatten())
        for cell_id in labels[1:]:
            bin_region = (img_labeled == cell_id)
            region = bin_region * img_intensity
            region, _ = imtools.histeq(region)
            # segmentation
            T = np.mean(np.ma.masked_equal(region, 0))
            img_seg += (region > T)
        labeled, counts = label(img_seg)        
        return labeled, counts

    @staticmethod    
    def segmentTubes4(img_labeled, img_intensity):
        """
        Parameters
        ==========
        :img_labeled: labeled image before segmentation  
        
        img_intensity: intensity image  
        
        Returns
        =======
        labeld image segmented by mean-intensity by each region.  
        * This function do not perform histogram equalization by each region.  
        If you want to do that before the segmentation, use 'segementation3' function
        """
        img_seg = np.zeros_like(img_intensity)    
        labels = np.unique(img_labeled.flatten())
        for cell_id in labels[1:]:
            bin_region = (img_labeled == cell_id)
            region = bin_region * img_intensity
            # segmentation
            T = np.mean(np.ma.masked_equal(region, 0))
            img_seg += (region > T)
        labeled, counts = label(img_seg)        
        return labeled, counts
    
    @staticmethod
    def labeledSkeleton(labeledImage):
        canvas = np.zeros_like(labeledImage)
        count = np.unique(labeledImage).max()
        for i in range(count):
            img_bin_afterRelabeled   = (labeledImage == i+1)
            skeleton                 = skeletonize(img_bin_afterRelabeled)
            canvas                   += skeleton * labeledImage
        return canvas

    def detectMyotube(self, segmentationMethod='seg1', sizeThresh=0, tophat=True, tophatImgList=[]):
        if tophat==True and tophatImgList == []:
            tophatImgList = self.tophatAllPlanes()
        elif tophat==True and tophatImgList != []:
            #tophatImgList = tophatImgList[tophatImgList.keys()[0]]
            tophatImgList = tophatImgList[0]
        elif tophat==False:
            tophatImgList = self.images
        
        # median -> histeq -> otsu -> segmentation (histeq and otsu by region)
        if segmentationMethod == 'seg1':
            img_mip                              = self.maximumIntensityProjection(tophatImgList)
            img_median                           = self.smooth(img_mip)
            img_histeq, _                        = imtools.histeq(img_median)
            T                                    = otsu(np.uint(img_histeq), ignore_zeros=True)
            img_bin                              = (np.uint(img_histeq) > T)
            img_labeled, _                       = label(img_bin)
            markers, counts                      = self.segmentTubes1(img_labeled, img_histeq)
            # segmentation by watershed
            img_labeled                          = img_bin * cwatershed(-distance(img_bin), markers)
            result = {'MIP':img_mip, 'median':img_median, 'histEq':img_histeq, 'otsu':T, 'bin':img_bin}
            
        # median -> otsu -> segmentation (histeq and otsu by region)
        elif segmentationMethod == 'seg2':
            img_mip                              = self.maximumIntensityProjection(tophatImgList)
            img_median                           = self.smooth(img_mip)
            T                                    = otsu(np.uint(img_median), ignore_zeros=True)
            img_bin                              = (np.uint(img_median) > T)
            img_labeled, _                       = label(img_bin)
            markers, counts                      = self.segmentTubes2(img_labeled, img_median)            
            # segmentation by watershed
            img_labeled                          = img_bin * cwatershed(-distance(img_bin), markers)
            result = {'MIP':img_mip, 'median':img_median, 'otsu':T, 'bin':img_bin}

        # median -> histeq -> otsu -> segmentation (histeq and cut regions less than mean-intensity by region)
        elif segmentationMethod == 'seg3':
            img_mip                              = self.maximumIntensityProjection(tophatImgList)
            img_median                           = self.smooth(img_mip)
            img_histeq, _                        = imtools.histeq(img_median)
            T                                    = otsu(np.uint(img_histeq), ignore_zeros=True)
            img_bin                              = (np.uint(img_histeq) > T)
            img_labeled, _                       = label(img_bin)
            markers, counts                      = self.segmentTubes3(img_labeled, img_histeq)
            # segmentation by watershed
            img_labeled                          = img_bin * cwatershed(-distance(img_bin), markers)
            result = {'MIP':img_mip, 'median':img_median, 'histEq':img_histeq, 'otsu':T, 'bin':img_bin}
        
        # median -> histeq -> otsu -> segmentation (cut regions less than mean-intensity by region)
        elif segmentationMethod == 'seg4':
            img_mip                              = self.maximumIntensityProjection(tophatImgList)
            img_median                           = self.smooth(img_mip)
            img_histeq, _                        = imtools.histeq(img_median)
            T                                    = otsu(np.uint(img_histeq), ignore_zeros=True)
            img_bin                              = (np.uint(img_histeq) > T) 
            img_labeled, _                       = label(img_bin)
            markers, counts                      = self.segmentTubes4(img_labeled, img_histeq)
            # segmentation by watershed
            img_labeled                          = img_bin * cwatershed(-distance(img_bin), markers)
            result = {'MIP':img_mip, 'median':img_median, 'histEq':img_histeq, 'otsu':T, 'bin':img_bin}
        
        # median -> otsu -> segmentation (cut regions less than mean-intensity by region)
        elif segmentationMethod == 'seg5':
            img_mip                              = self.maximumIntensityProjection(tophatImgList)
            img_median                           = self.smooth(img_mip)
            T                                    = otsu(np.uint(img_median), ignore_zeros=True)
            img_bin                              = (np.uint(img_median) > T)
            img_labeled, _                       = label(img_bin)
            markers, counts                      = self.segmentTubes4(img_labeled, img_median)            
            # segmentation by watershed
            img_labeled                          = img_bin * cwatershed(-distance(img_bin), markers)
            result = {'MIP':img_mip, 'median':img_median, 'otsu':T, 'bin':img_bin}            
            
        # non-segmentation
        else:
            img_mip                              = self.maximumIntensityProjection(tophatImgList)
            img_median                           = self.smooth(img_mip)
            img_histeq, _                        = imtools.histeq(img_median)
            T                                    = otsu(np.uint(img_histeq))
            img_bin                              = (np.uint(img_histeq) > T)
            img_labeled, counts                  = label(img_bin)
            result = {'MIP':img_mip, 'median':img_median, 'histEq':img_histeq, 'otsu':T, 'bin':img_bin}
            print('non-segmentation')
        print ('Otsu\'s threshold:', T)
        print('Found {} objects.'.format(counts))       
        sizes = labeled_size(img_labeled)
        img_labeled                          = remove_regions_where(img_labeled, sizes < sizeThresh)#origin 10000, triangle 8585    
        img_relabeled, counts                = relabel(img_labeled)
        result['label']                      = img_relabeled
        result['count']                      = counts
        print ('After filtering and relabeling, there are {} objects left.'.format(counts))        
        result['labeledSkeleton']            = self.labeledSkeleton(img_relabeled)        
        return ProcessImages(result)


class ProcessImages:
    def __init__(self, ProcessImageInDict):
        self.allProcessImage = ProcessImageInDict
        self.MIP = ProcessImageInDict['MIP']
        self.median = ProcessImageInDict['median']
        self.T_otsu = ProcessImageInDict['otsu']
        self.binary = ProcessImageInDict['bin']
        self.labelImage = ProcessImageInDict['label']
        self.count = ProcessImageInDict['count']
        self.labeledSkelton = ProcessImageInDict['labeledSkeleton']
        #if ProcessImageInDict.has_key('histEq'):
        if "histEq" in ProcessImageInDict:                 
                 self.histEqImage = ProcessImageInDict['histEq']
            
    def regionprops(self, intensityImages, skeleton=False):
        """
        Parameters
        ==========
        :args[0]: Labeled image
        :args[1]: Intensity image
        
        Returns
        =======
        :Region properties: list type
        """
        if skeleton==True:
            return MultiImageProps([regionprops(self.labeledSkelton, intensityImages[plane]) for plane in range(len(intensityImages))])
        else:
            return MultiImageProps([regionprops(self.labelImage, intensityImages[plane]) for plane in range(len(intensityImages))])
        
            
    def show(self, key, **cmap):
        """
        plot imegeprocessing-images
        ===========================
        keys
        :mip: maximum intensity projection
        :median: median filtered image
        :histEqImage: histogram-equalized image
        :binary: binary image
        :labeled: labeled image
        
        Returns
        =======
        figure object
        """
        fig, axes = plt.subplots(1,1)
        if cmap is not None:
            axes.imshow(self.allProcessImage[key], cmap=cmap[list(cmap.keys())[0]])
        else:
            axes.imshow(self.allProcessImage[key])            
        plt.close()
        return fig
        
    def hist(self, key):
        """
        plot images histogram
        =====================
        keys
        :mip: maximum intensity projection
        :median: median filtered image
        :histEqImage: histogram-equalized image
        :binary: binary image
        :labeled: labeled image
        
        Returns
        =======
        figure object
        """
        fig, axes = plt.subplots(1,1, num=key)
        axes.hist(self.allProcessImage[key].flatten(), 128)
        plt.close(num=key)
        return fig
        
    def showAll(self):
        fig, axes = plt.subplots(2,3, num='All', figsize=(20,12))
        axes = axes.flatten()
        axes[0].imshow(self.MIP, cmap='gray')
        axes[1].hist(self.MIP.flatten(), 128)
        axes[2].imshow(self.median, cmap='gray')
        axes[3].imshow(self.binary, cmap='binary_r')
        axes[4].imshow(self.MIP, cmap='gray')
        axes[4].contour(self.labelImage, [0.5], linewidths=1.2, colors='r')
        axes[5].imshow(self.labelImage)
        # set titles
        axes[0].set_title('MIP')
        axes[1].set_title('Histogram of MIP')
        axes[2].set_title('Median')
        axes[3].set_title('Binary')
        axes[4].set_title('Contour')
        axes[5].set_title('Label')
        plt.tight_layout()
        return fig
      
    def plotIDs(self, idList, coordinates):
        """
        Parameters
        ==========
        mip: maximum intensity projection-image  
        img_labeled: labeled image  
        labels: label number or string  
        coodinates: coordinates of centroid of region  
        
        Returns
        =======
        figure
        """
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(self.MIP, cmap='gray')
        plt.contour(self.labelImage, [0.5], linewidth=1.2, colors='r')
        for label_, (x,y) in zip(idList[0,:], coordinates.ix[0,:]):
            plt.text(y, x, label_, color='y')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def saveProcess(fig, path, title):
        fullPath = path + "/"+ title + '.png'
        fig.savefig(fullPath, transparent=True, dpi=300)
        plt.close()
        
    def saveAllProcess(self, path, title):
        fig = self.showAll()
        self.saveProcess(fig, path, title)

    def saveIDplot(self, idList, coodinates, path, title):
        fig = self.plotIDs(idList, coodinates)
        fullPath = os.path.join(path, title + '.png')
        fig.savefig(fullPath, transparent=True, dpi=300)
        plt.close()



import pandas as pd


class MultiImageProps():
    def __init__(self, props):
        self.props = props
        self.label = np.array(self.getID())
        self.area = np.array(self.getArea())
        self.intensity = np.array(self.getIntensity())
        self.perimeter = np.array(self.getPerimeter())
        self.centroid = pd.DataFrame(self.getCentroid())
    
    def getID(self, *props):
        if props == ():
            return [[self.props[plane][i].label for i in range(len(self.props[plane]))] for plane in range(len(self.props))]
        else:
            return [[props[0][plane][i].label for i in range(len(props[0][plane]))] for plane in range(len(props[0]))]
        
    def getArea(self, *props):
        if props == ():
            return [[self.props[plane][i].area for i in range(len(self.props[plane]))] for plane in range(len(self.props))]
        else:
            return [[props[0][plane][i].area for i in range(len(props[0][plane]))] for plane in range(len(props[0]))]
        
    def getIntensity(self, *props):
        if props == ():
            return [[self.props[plane][i].mean_intensity for i in range(len(self.props[plane]))] for plane in range(len(self.props))]
        else:
            return [[props[0][plane][i].mean_intensity for i in range(len(props[0][plane]))] for plane in range(len(props[0]))]
        
    def getPerimeter(self, *props):
        if props == ():
            return [[self.props[plane][i].perimeter for i in range(len(self.props[plane]))] for plane in range(len(self.props))]
        else:
            return [[props[0][plane][i].perimeter for i in range(len(props[0][plane]))] for plane in range(len(props[0]))]

    def getCentroid(self, *props):
        if props == ():
            return [[self.props[plane][i].centroid for i in range(len(self.props[plane]))] for plane in range(len(self.props))]
        else:
            return [[props[0][plane][i].centroid for i in range(len(props[0][plane]))] for plane in range(len(props[0]))]
        
    def setLengthGetWidth(self, LengthDf):
        self.length = LengthDf
        self.width = np.divide(np.float64(self.area), LengthDf)
        return self.width
        
    def getRatio(self, intensity):
        """
        Parameters
        ==========
        :intensity: intensity time-courses. vertical axis=time, horizontal axis=cells. 2D array like object.
        Return
        ======
        Intensity ratio. vertical axis=time, horizontal axis=cells. 2D array.
        """
        return FRETRatio(self.intensity / intensity)
    
    def saveProp(self, path, propName, title):
        df = getattr(self, propName)
        df.to_csv(path +"/"+ title + '.csv')

class FRETRatio():
    """
    Parameters
    ==========
    ratioDf: ratio values. vertical axis = time, horizontal axis = cells. 2D array or DataFrame object.
    """
    def __init__(self, ratioArr):
        self.ratio = ratioArr
    
    def toFC(self, frame):    
        return self.ratio / self.ratio[frame,:]
        
    def normalize(self, frame):
        meanBasal = self.ratio[:frame+1,:].mean(axis=0)
        stdBasal = self.ratio[:frame+1,:].std(axis=0)
        return (self.ratio - meanBasal) / stdBasal
        
    # this class will be expanded for tsa class

     
#if __name__=='__main__':
def main(path):
    #path = r'data/2_1_10s_interval'
    try:
        os.mkdir("output")
    except:
        print("output have existed")
    outputpath = r'output'
    path = path.replace('\\', '/')
    outputpath = outputpath.replace('\\', '/')
    baseTitle = path.split('/')[-1]
    #stagePositionNum = 0
    #stimFrame =0 # start from 0
    interval = 70
    duration = 63630
    #timePoints = range(0, duration+interval, interval)
    Vol = '0V'
    #fluo2 = 'FRET-YFP'
    #indexNameForCsv = 'time(ms)'

    #T = pd.read_csv(r'E:\OneDrive\UT\KurodaLab\Analysis\FRET\Insulin_Step\CalcTriangleThresh\160601\trianglethresh_mean_bins=300-500.csv', header=None)
    #sizethresh = T.iloc[0,1]
    sizethresh = 1000
    print(sizethresh)
    print (baseTitle)
    
    #fig1, axes1 = plt.subplots(2,2, sharex=True, sharey=True)
    #fig2, axes2 = plt.subplots(2,2, sharex=True, sharey=True)
    #fig3, axes3 = plt.subplots(2,2, sharex=True, sharey=True)
    

    LCI1 = LiveCellImaging(path, Vol)
    #LCI2 = LiveCellImaging(path, fluo2, stagePosition)
    ic1 = LCI1.to_MultiImage(interval)
    #ic2 = LCI2.to_MultiImage(interval)
    subBG1 = ic1.tophatAllPlanes()
    #subBG2 = ic2.tophatAllPlanes()
    myotube1  = MultiMyotubeImage(subBG1, interval)
    processImages = myotube1.detectMyotube(sizeThresh=sizethresh, tophatImgList=subBG1, segmentationMethod='seg1')
    np.savetxt(outputpath +"/"+ baseTitle +"_labelImage.csv",processImages.labelImage,delimiter=",")    
    #for i,j in zip(LCI1.files,subBG1):
    #    np.savetxt(i.split(".")[0] +"_subBG.csv",j,delimiter=",")
    props1 = processImages.regionprops(subBG1)
    #props2 = processImages.regionprops(subBG2)
    propsforlength = processImages.regionprops(subBG1, skeleton=True)
    length = propsforlength.area
    width = props1.setLengthGetWidth(length)
    
    processImages.saveAllProcess(outputpath, baseTitle + '_process')
    processImages.saveProcess(processImages.show('histEq', cmap='gray'), outputpath, baseTitle + '_histEq')
    processImages.saveIDplot(props1.label, props1.centroid, outputpath, baseTitle + 'label')
    
    del(LCI1)
    del(ic1)
    del(subBG1)
    del(myotube1)
    del(processImages)
    del(props1)
    del(propsforlength)
    del(length)
    del(width)
    #ratio = props2.getRatio(props1.intensity)
    #ratioFC = ratio.toFC(stimFrame)
    #ratioNorm = ratio.normalize(stimFrame)
"""
    keys = list(vars(props1).keys())
    keys.remove('props')
    for key in keys:
        val = getattr(props1, key)
        df = pd.DataFrame(val, index=timePoints)
        if key == 'intensity':
            df.to_csv(outputpath +"/"+ baseTitle + '.csv')
        elif key == 'centroid':
            val.index = timePoints
            val.to_csv(outputpath +"/"+ baseTitle+ '.csv')
        else:
            df.to_csv(outputpath +"/"+ baseTitle+ '.csv')
"""
    #df = pd.DataFrame(props2.intensity, index=timePoints)
    #df.to_csv(os.path.join(outputpath, baseTitle+ '_FRET-YFP_s%s.csv' %stagePosition))
    #ratioDf = pd.DataFrame(ratio.ratio, index=timePoints)
    #ratioFCDf = pd.DataFrame(ratioFC, index=timePoints)
    #ratioNormDf = pd.DataFrame(ratioNorm, index=timePoints)
    #ratioDf.to_csv(os.path.join(outputpath, baseTitle + '_FRETRatio_s%s.csv' %stagePosition))
    #ratioFCDf.to_csv(os.path.join(outputpath, baseTitle + '_FRETRatio_FC_s%s.csv' %stagePosition))
    #ratioNormDf.to_csv(os.path.join(outputpath, baseTitle) + '_FRETRatio_norm_s%s.csv' %stagePosition)
"""
    axes1 = axes1.flatten()
    axes2 = axes2.flatten()
    axes3 = axes3.flatten()
    axes1[i].plot(ratioDf.index, ratioDf.values)
    axes2[i].plot(ratioFCDf.index, ratioFCDf.values)
    axes3[i].plot(ratioNormDf.index, ratioNormDf.values)
    fig1.savefig(os.path.join(outputpath, baseTitle + '.png'), dpi=300, transparent=True)
    fig2.savefig(os.path.join(outputpath, baseTitle + '_fc.png'), dpi=300, transparent=True)
    fig3.savefig(os.path.join(outputpath, baseTitle + '_norm.png'), dpi=300, transparent=True)"""