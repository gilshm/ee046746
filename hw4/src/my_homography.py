import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from matplotlib import pyplot as plt


def getPoints(im1, im2, N):
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB))

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB))

    cord = plt.ginput(2*N)
    plt.show()
    plt.tight_layout()

    cord = np.array(cord).transpose()

    p1 = cord[:, :N]
    p2 = cord[:, N:]

    return p1, p2


def computeH(p1, p2):
    assert (p1.shape[1] == p2.shape[1])
    assert (p1.shape[0] == 2)

    # Create A matrix
    A = np.zeros((2*p1.shape[1], 9))

    # Even rows
    A[0::2, 0:2] = p1.transpose()
    A[0::2, 2] = 1
    A[0::2, 6:8] = -1 * p1.transpose() * p2[0, :][:, None]
    A[0::2, 8] = -1 * p2[0, :]

    # Odd rows
    A[1::2, 3:5] = p1.transpose()
    A[1::2, 5] = 1
    A[1::2, 6:8] = -1 * p1.transpose() * p2[1, :][:, None]
    A[1::2, 8] = -1 * p2[1, :]

    u, s, vh = np.linalg.svd(A, full_matrices=True)
    H2to1 = vh[-1, :].reshape((3, 3))

    return H2to1


def warpH(im1, H, out_size):
    """
    Your code here
    """
    return warp_im1


def imageStitching(img1, wrap_img2):
    """
    Your code here
    """
    return panoImg


def ransacH(matches, locs1, locs2, nIter, tol):
    """
    Your code here
    """
    return bestH


def getPoints_SIFT(im1,im2):
    """
    Your code here
    """
    return p1,p2


if __name__ == '__main__':
    print('my_homography')
    im1_color = cv2.imread('data/incline_L.png')
    im2_color = cv2.imread('data/incline_R.png')

    im1 = cv2.normalize(im1_color, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    im2 = cv2.normalize(im2_color, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    p1, p2 = getPoints(im1, im2, 3)
    computeH(p1, p2)

