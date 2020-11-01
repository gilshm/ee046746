import numpy as np
import matplotlib.pyplot as plt
import cv2


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


def main():
    sigma0 = 1
    k = 2**0.5
    levels = [-1, 0, 1, 2, 3, 4]
    th_c = 0.03
    th_r = 12

    # 1.1
    im_color = cv2.imread('../data/model_chickenbroth.jpg')
    # im_color = cv2.imread('../data/pf_floor.jpg')

    # Normalize and turn into grayscale
    im = cv2.normalize(im_color, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Run the DoG detector
    locs_dog, gauss_pyramid = DoGdetector(im, sigma0, k, levels, th_c, th_r)

    # Visualize
    x, y = [], []
    for p in locs_dog:
        if p[2] == 0:
            x.append(p[1])
            y.append(p[0])

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB))
    plt.scatter(x, y, c='r')
    plt.show()


if __name__ == "__main__":
    main()