import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy import interpolate
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
    H1to2 = vh[-1, :].reshape((3, 3))

    return H1to2


def warpH(im1, H, out_size):
    im1_lab = cv2.cvtColor(im1, cv2.COLOR_RGB2LAB)

    x_out, y_out = np.meshgrid(np.arange(out_size[1]), np.arange(out_size[0]))
    x_out, y_out = x_out.flatten(), y_out.flatten()
    px_out = np.stack((x_out, y_out, np.ones_like(x_out)))

    px_in = np.matmul(H, px_out)
    px_in = px_in / px_in[2, :]
    x_in, y_in = px_in[0], px_in[1]

    valid_pixels = np.where((x_in >= 0) & (y_in >= 0) & (x_in < im1.shape[1]) & (y_in < im1.shape[0]))
    x_in = x_in[valid_pixels]
    y_in = y_in[valid_pixels]
    x_out = x_out[valid_pixels]
    y_out = y_out[valid_pixels]

    out_size = (out_size[0], out_size[1], 3)
    warp_im = np.zeros(out_size, dtype=np.uint8)
    warp_im[:, :, 1:] = 128

    for c in range(im1_lab.shape[2]):
        f_interp = scipy.interpolate.interp2d(np.arange(im1_lab.shape[1]),
                                              np.arange(im1_lab.shape[0]),
                                              im1_lab[:, :, c], kind='cubic')

        # TODO: parallelize implementation
        for _px_in, _py_in, _px_out, _py_out in zip(x_in, y_in, x_out, y_out):
            warp_im[_py_out, _px_out, c] = f_interp(_px_in, _py_in)

    return cv2.cvtColor(warp_im, cv2.COLOR_LAB2RGB)


def imageStitching(img1, warp_img2):
    panoImg = np.zeros_like(warp_img2)

    img2_mask = warp_img2 > 0
    panoImg[img2_mask] = warp_img2[img2_mask]

    panoImg[:img1.shape[0], :img1.shape[1], :] = img1

    return panoImg


def ransacH(matches, locs1, locs2, nIter, tol):
    """
    Your code here
    """
    return bestH


def getPoints_SIFT(im1, im2):
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    # Create Brute Force (BF) matcher object
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)

    # Draw first 10 matches.
    # im3 = cv2.drawMatches(im1, kp1, im2, kp2, matches[:10], None)
    # plt.imshow(im3), plt.show()

    p1, p2 = [], []
    for match in matches[:10]:
        p1.append(kp1[match.queryIdx].pt)
        p2.append(kp2[match.trainIdx].pt)

    return np.stack(p1).T, np.stack(p2).T


if __name__ == '__main__':
    print('my_homography')
    im1_color = cv2.imread('data/incline_L.png')
    im2_color = cv2.imread('data/incline_R.png')

    im1_color = cv2.cvtColor(im1_color, cv2.COLOR_BGR2RGB)
    im2_color = cv2.cvtColor(im2_color, cv2.COLOR_BGR2RGB)

    im1 = cv2.cvtColor(im1_color, cv2.COLOR_BGR2GRAY)
    im2 = cv2.cvtColor(im2_color, cv2.COLOR_BGR2GRAY)

    # Remove comments to find the H matrix based on manual point selection
    # p1, p2 = getPoints(im1, im2, 6)
    # H = computeH(p1, p2)

    # H matrix based on manual point selection
    # H = np.array([[2.78963306e-03,  1.36841979e-05, -9.98415937e-01],
    #               [2.97656603e-04,  2.35700294e-03, -5.61163434e-02],
    #               [1.12866686e-06, -2.33769309e-07,  1.77398500e-03]])

    # Finding H with SIFT keypoints
    p1, p2 = getPoints_SIFT(im1, im2)
    H = computeH(p1, p2)

    # warped_img = warpH(im2_color, H, (800, 2000))
    # pano_img = imageStitching(im1_color, warped_img)

    # plt.figure(figsize=(16, 12))
    # plt.imshow(pano_img)
    # plt.axis('off')
    # plt.show()


