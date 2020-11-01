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


if __name__ == "__main__":
    main()