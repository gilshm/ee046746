import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import cdist

sigma0 = 1
k = 2 ** 0.5
levels = [-1, 0, 1, 2, 3, 4]
th_c = 0.03
th_r = 12


def createGaussianPyramid(im, sigma0, k, levels):
    GaussianPyramid = []

    for i in range(len(levels)):
        sigma_ = sigma0 * k ** levels[i]
        size = int(np.floor(3 * sigma_ * 2) + 1)
        blur = cv2.GaussianBlur(im, (size, size), sigma_)
        GaussianPyramid.append(blur)

    return np.stack(GaussianPyramid)


def displayPyramid(pyramid):
    plt.figure(figsize=(16, 5))
    plt.imshow(np.hstack(pyramid), cmap='gray')
    plt.axis('off')
    plt.show()


def createDoGPyramid(GaussianPyramid, levels):
    DoGPyramid = []

    for i, _ in enumerate(GaussianPyramid):
        if i == 0:
            continue

        diff = GaussianPyramid[i] - GaussianPyramid[i-1]
        DoGPyramid.append(diff)

    DoGLevels = levels

    return np.stack(DoGPyramid), DoGLevels


def computePrincipalCurvature(DoGPyramid):
    PrincipalCurvature = []

    for im in DoGPyramid:
        # Compute the Hessian using Sobel filter
        d_xx = cv2.Sobel(im, cv2.CV_64F, 2, 0, ksize=3)
        d_yy = cv2.Sobel(im, cv2.CV_64F, 0, 2, ksize=3)
        d_xy = cv2.Sobel(im, cv2.CV_64F, 1, 1, ksize=3)

        _curvature = np.zeros_like(im)

        for im_y in range(im.shape[0]):
            for im_x in range(im.shape[1]):
                _curvature[im_y, im_x] = calcCurvature(d_xx[im_y, im_x], d_yy[im_y, im_x], d_xy[im_y, im_x])

        PrincipalCurvature.append(_curvature)

    return np.stack(PrincipalCurvature)


def calcCurvature(d_xx, d_yy, d_xy):
    tr_H = d_xx + d_yy
    det_H = d_xx * d_yy - d_xy ** 2
    return tr_H ** 2 / det_H


def getLocalExtrema(DoGPyramid, DoGLevels, PrincipalCurvature, th_contrast, th_r):
    locsDoG = []

    suspect_coord = np.where((np.abs(DoGPyramid) > th_contrast) & (np.abs(PrincipalCurvature) < th_r))

    for coord in zip(suspect_coord[0], suspect_coord[1], suspect_coord[2]):
        vals = getAdjacentValues(DoGPyramid, coord[2], coord[1], coord[0])

        if (DoGPyramid[coord] >= max(vals)) or (DoGPyramid[coord] <= min(vals)):
            locsDoG.append([coord[1], coord[2], coord[0]])

    return locsDoG


def getAdjacentValues(pyramid, x, y, z):
    vals = []

    # Space dimension
    for y_offset in range(-1, 2):
        for x_offset in range(-1, 2):
            y_t = y + y_offset
            x_t = x + x_offset
            if (x_offset == 0 and y_offset == 0) or \
               (x_t < 0) or (x_t >= pyramid.shape[2]) or \
               (y_t < 0) or (y_t >= pyramid.shape[1]):
                continue

            vals.append(pyramid[z, y_t, x_t])

    # Scale dimension
    if z - 1 >= 0:
        vals.append(pyramid[z - 1, y, x])
    if z + 1 < pyramid.shape[0]:
        vals.append(pyramid[z + 1, y, x])

    return vals


def DoGdetector(im, sigma0, k, levels, th_contrast, th_r):

    # 1.2
    pyramid = createGaussianPyramid(im, sigma0, k, levels)
    # displayPyramid(pyramid)

    # 1.3
    pyramid_dog, _ = createDoGPyramid(pyramid, levels)
    # displayPyramid(pyramid_dog)

    # 1.4
    curvature = computePrincipalCurvature(pyramid_dog)

    # 1.5
    locs_dog = getLocalExtrema(pyramid_dog, levels, curvature, th_contrast, th_r)

    return locs_dog, pyramid


def makeTestPattern(patchWidth, nbits):
    compareX = np.random.randint(-patchWidth / 2, patchWidth / 2, (2, nbits))
    compareY = np.random.randint(-patchWidth / 2, patchWidth / 2, (2, nbits))

    return compareX, compareY


def computeBrief(im, GaussianPyramid, locsDoG, k, levels, compareX, compareY):
    locs, desc = [], []

    for coord in locsDoG:
        if (coord[0] - 4 < 0) or (coord[1] - 4 < 0) or (coord[0] + 4 >= im.shape[0]) or (coord[1] + 4 >= im.shape[1]):
            continue

        n_bits = []
        for test_pair in zip(list(zip(compareX[0], compareY[0])), list(zip(compareX[1], compareY[1]))):
            intensities = GaussianPyramid[coord[2], coord[0] + test_pair[0], coord[1] + test_pair[1]]

            n_bits.append(1) if intensities[0] < intensities[1] else n_bits.append(0)

        # Reverse coordinates order
        locs.append([coord[1], coord[0], coord[2]])
        desc.append(n_bits)

    return np.stack(locs), np.stack(desc)


def briefLite(im, compare_x, compare_y):
    locs_dog, gauss_pyramid = DoGdetector(im, sigma0, k, levels, th_c, th_r)
    locs, desc = computeBrief(im, gauss_pyramid, locs_dog, k, levels, compare_x, compare_y)

    return locs, desc


def briefMatch(desc1, desc2, ratio):
    D = cdist(np.float32(desc1), np.float32(desc2), metric='hamming')

    # Find the smallest distance
    ix2 = np.argmin(D, axis=1)
    d1 = D.min(1)

    # Find the second smallest distance
    d12 = np.partition(D, 2, axis=1)[:, 0:2]
    d2 = d12.max(1)
    r = d1 / (d2 + 1e-10)
    is_discr = r < ratio
    ix2 = ix2[is_discr]
    ix1 = np.arange(D.shape[0])[is_discr]
    matches = np.stack((ix1, ix2), axis=-1)

    return matches


def plotMatches(im1, im2, matches, locs1, locs2):
    fig = plt.figure()

    # Draw two images side by side
    imH = max(im1.shape[0], im2.shape[0])
    im = np.zeros((imH, im1.shape[1]+im2.shape[1]), dtype='uint8')
    im[0:im1.shape[0], 0:im1.shape[1]] = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im[0:im2.shape[0], im1.shape[1]:] = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    plt.imshow(im, cmap='gray')

    for i in range(matches.shape[0]):
        pt1 = locs1[matches[i, 0], 0:2]
        pt2 = locs2[matches[i, 1], 0:2].copy()
        pt2[0] += im1.shape[1]

        x = np.asarray([pt1[0], pt2[0]])
        y = np.asarray([pt1[1], pt2[1]])

        plt.plot(x, y, 'r', lw=0.1)
        plt.plot(x, y, 'g.', lw=0.1)

    plt.axis('off')
    plt.show()


def main():
    part1, part2 = False, True

    im_color1 = cv2.imread('../data/chickenbroth_03.jpg')
    im1 = cv2.normalize(im_color1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

    im_color2 = cv2.imread('../data/chickenbroth_04.jpg')
    im2 = cv2.normalize(im_color2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    # --------------
    # --- Part I ---
    # --------------
    if part1:
        # Run the DoG detector
        locs_dog, gauss_pyramid = DoGdetector(im2, sigma0, k, levels, th_c, th_r)

        # Visualize
        x, y = [], []
        for p in locs_dog:
            if p[2] == 0:
                x.append(p[1])
                y.append(p[0])

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(im_color2, cv2.COLOR_BGR2RGB))
        plt.scatter(x, y, c='r')
        plt.show()

    # ---------------
    # --- Part II ---
    # ---------------
    if part2:
        # Make BRIEF test pattern
        compare_x, compare_y = makeTestPattern(9, 256)

        # Run BRIEF
        locs1, desc1 = briefLite(im1, compare_x, compare_y)
        locs2, desc2 = briefLite(im2, compare_x, compare_y)
        matches = briefMatch(desc1, desc2, 0.5)

        # Visualize
        plotMatches(im_color1, im_color2, matches, locs1, locs2)


if __name__ == "__main__":
    main()