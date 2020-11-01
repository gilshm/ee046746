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
        if i+1 == len(GaussianPyramid):
            break

        diff = GaussianPyramid[i] - GaussianPyramid[i+1]
        DoGPyramid.append(diff)

    DoGLevels = levels

    return DoGPyramid, DoGLevels


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


def main():
    sigma0 = 1
    k = 2**0.5
    levels = [-1, 0, 1, 2, 3, 4]

    # 1.1
    im = cv2.imread('../data/model_chickenbroth.jpg')
    plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # 1.2
    im = cv2.normalize(im, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    pyramid = createGaussianPyramid(im, sigma0, k, levels)
    displayPyramid(pyramid)

    # 1.3
    pyramid_dog, _ = createDoGPyramid(pyramid, levels)
    displayPyramid(pyramid_dog)

    # 1.4
    curvature = computePrincipalCurvature(pyramid_dog)


if __name__ == "__main__":
    main()