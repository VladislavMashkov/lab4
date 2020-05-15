import cv2
import mahotas
import numpy as np
from numpy.linalg import norm
bins = 8

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

# feature-descriptor-4: SIFT
''''''''''
def fd_4(image, mask=None):
    imag = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)
    imag=cv2.drawKeypoints(gray,kp,imag)
    cv2.normalize(imag, imag)
    return imag.flatten()
'''''''''
#KAZE detector
def kaze_func(image):
    alg = cv2.KAZE_create()
    # Dinding image keypoints
    kps = alg.detect(image)
    # Getting first 32 of them.
    # Number of keypoints is varies depend on image size and color pallet
    # Sorting them based on keypoint response value(bigger is better)
    kps = sorted(kps, key=lambda x: -x.response)[:32]
    # computing descriptors vector
    kps, dsc = alg.compute(image, kps)
    cv2.normalize(dsc, dsc)
    # Flatten all of them in one big vector - our feature vector
    return dsc.flatten()
    '''''''''
    sobelx8u = cv2.Sobel(image, cv2.CV_8U, 1, 0, ksize=5)

    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobelx64f = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    cv2.normalize(sobelx8u, sobelx8u)
    return sobelx8u.flatten()
'''''''''

def hog(img):
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bin_n = 16 # Number of bins
    bin = np.int32(bin_n*ang/(2*np.pi))

    bin_cells = []
    mag_cells = []

    cellx = celly = 8

    for i in range(0,int(img.shape[0]/celly)):
        for j in range(0,int(img.shape[1]/cellx)):
            bin_cells.append(bin[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])
            mag_cells.append(mag[i*celly : i*celly+celly, j*cellx : j*cellx+cellx])

    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)

    # transform to Hellinger kernel
    eps = 1e-7
    hist /= hist.sum() + eps
    hist = np.sqrt(hist)
    hist /= norm(hist) + eps

    return hist

#FAST feature detector
def fd_Fast(image):
    imag = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(imag, cv2.COLOR_BGR2GRAY)
    fast = cv2.FastFeatureDetector_create()
    # find and draw the keypoints
    kp = fast.detect(gray, None)
    img2 = cv2.drawKeypoints(gray, kp, None, color=(255, 0, 0))
    cv2.normalize(img2, img2)
    return img2.flatten()

def lin(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.Laplacian(gray, cv2.CV_64F)
    cv2.normalize(gray, gray)
    return gray.flatten()

def fd_ORB(image):
    orb = cv2.ORB_create()

    kp = orb.detect(image, None)

    kp, des = orb.compute(image, kp)

    img2 = cv2.drawKeypoints(des, kp, None, color=(255, 0, 0))
    cv2.normalize(img2, img2)
    return img2.flatten()