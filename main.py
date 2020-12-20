# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

fig, ((ax1, ax2), (ax4, ax5)) = plt.subplots(nrows=2, ncols=2)


def getPatch(xy, img):
    patch = img[xy[1] - 15:xy[1] + 15, xy[0] - 15:xy[0] + 15]
    return patch


def calcVariance(patch):
    x = cv2.Sobel(patch, -1, 1, 0)
    y = cv2.Sobel(patch, -1, 0, 1)

    return np.abs(x) + np.abs(y)


def pickBestAround(xy, values, image):
    bestV = 0
    best_xy = None
    for move in np.array([(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]) * 30:
        xy_m = xy + move
        if xy_m[xy_m < 0].sum() < 0:
            continue
        patch = getPatch(xy_m, values)
        variance = 1 / calcVariance(patch).sum()
        if variance > bestV:
            bestV = variance
            best_xy = xy_m

        return getPatch(best_xy, values), getPatch(best_xy, image)


def onMouseEvent(action, x, y, flags, img):
    if action == cv2.EVENT_LBUTTONDBLCLK or action == cv2.EVENT_RBUTTONDOWN:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        xy = (x, y)
        patch = getPatch(xy, gray)
        patchimg = getPatch(xy, img)
        sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=5)
        fig.tight_layout()
        plt.subplot(2, 3, 1)
        plt.imshow(patch, cmap='gray')
        plt.title('gray'),
        plt.subplot(2, 3, 2)
        plt.imshow(sobelx, cmap='gray')
        plt.title('sobelx'),
        plt.subplot(2, 3, 3)
        plt.imshow(sobely, cmap='gray')
        plt.title('sobely'),
        plt.subplot(2, 3, 4)
        plt.imshow(patchimg)
        plt.title('patchimg'),
        #

        (grayPatch, colorPatch) = pickBestAround((x, y), gray, img)
        plt.subplot(2, 3, 5)
        plt.imshow(grayPatch, cmap='gray')
        plt.title('grayPatch'),
        plt.subplot(2, 3, 6)
        plt.imshow(colorPatch)
        plt.title('colorPatch')
        plt.show()
        # src_mask = np.ones_like(grayPatch) * 255
        # cv2.seamlessClone(
        #     colorPatch, img, src_mask, (x, y), cv2.NORMAL_CLONE, blend=img)


img = cv2.imread("blemish.png", 1)

#onMouseEvent(cv2.EVENT_LBUTTONDBLCLK, 299, 137, True, img)

print("Project1")

cv2.namedWindow("Project1")

cv2.setMouseCallback("Project1", onMouseEvent, img)
# plt.interactive(False)
#
k = 0
while k != 27:
    cv2.imshow("Project1", img)
    k = cv2.waitKey(20)

cv2.destroyAllWindows()
