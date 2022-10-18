import cv2
import dlib
import random
import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
import faceBlendCommon as fbc

import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0,8.0)
matplotlib.rcParams['image.cmap'] = 'gray'
matplotlib.rcParams['image.interpolation'] = 'bilinear'

# Landmark model location
PREDICTOR_PATH =  "shape_predictor_68_face_landmarks.dat"

lipstick_colors = {"vamptastic_plum": (97, 45, 130),
                   "red_dahlia": (51, 30, 136),
                   "flamenco_red": (42, 31, 192),
                   "chery_red": (63, 45, 222),
                   "caramel_nude": (120, 131, 201),
                   "mango_tango": (103, 92, 223),
                   "neon_red": (79, 32, 223),
                   "electric_orchid": (139, 64, 243),
                   "forbbiden_fuchsia": (105, 39, 184),
                   "sweet_marsala": (93, 67, 164)}

# Get the face detector
faceDetector = dlib.get_frontal_face_detector()
# The landmark detector is implemented in the shape_predictor class
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

# im = cv2.imread("girl.jpeg")
# im = cv2.imread("personNotExist.jpeg")
# im = cv2.imread("me.png")
im = cv2.imread("personNotExistDark.jpeg")

imDlib = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
# plt.imshow(imDlib)

points = fbc.getLandmarks(faceDetector, landmarkDetector, imDlib)

lips = [points[x] for x in range(48, 68)]
mouth = [points[x] for x in range(60, 68)]

clone_lips = im.copy()

for point in lips:
    cv2.circle(clone_lips, point, 1, (0, 255, 0), -1)
cv2.imshow('lips', clone_lips)

for point in mouth:
    cv2.circle(clone_lips, point, 1, (0, 0, 255), -1)
# clone_lips = cv2.cvtColor(clone_lips,cv2.COLOR_BGR2RGB)
# cv2.imshow('lips', clone_lips)
# key = cv2.waitKey(10000)
# 1/0

# cv2.imshow('mouth', clone_lips)
# key = cv2.waitKey(5000)
# 1/0

def getLipsMask(size, lips):
    # Find Convex hull of all points
    hullIndex = cv2.convexHull(np.array(lips), returnPoints=False)
    # Convert hull index to list of points
    hullInt = []
    for hIndex in hullIndex:
        hullInt.append(lips[hIndex[0]])
    # Create mask such that convex hull is white
    mask = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(hullInt), (255, 255, 255))
    return mask

def apply_color_to_mask(mask):
    # Get random lipstick color
    color_name, color = random.choice(list(lipstick_colors.items()))
    print("[INFO] Color Name: {}".format(color_name))
    b, g, r = cv2.split(mask)
    b = np.where(b > 0, color[0], 0).astype('uint8')
    g = np.where(g > 0, color[1], 0).astype('uint8')
    r = np.where(r > 0, color[2], 0).astype('uint8')
    return cv2.merge((b, g, r)), color_name

contours = [np.asarray(lips, dtype=np.int32)]
(x, y, w, h) = cv2.boundingRect(contours[0])
center = (int(x+w/2), int(y+h/2))
mask = getLipsMask(im.shape, lips)
mouth_mask = getLipsMask(im.shape, mouth)
mouth_mask = cv2.bitwise_not(mouth_mask)
mask = cv2.bitwise_and(mask, mask, mask=mouth_mask[:, :, 0])

# cv2.imshow('mask', mouth_mask)
# key = cv2.waitKey(5000)
# 1/0

## Dilate lips mask to include some skin around the mouth
maskHeight, maskWidth = mask.shape[0:2]
maskSmall = cv2.resize(mask, (600, int(maskHeight * 600.0 / maskWidth)))
maskSmall = cv2.dilate(maskSmall, (3, 3))
maskSmall = cv2.GaussianBlur(maskSmall, (5, 5), 0, 0)
mask = cv2.resize(maskSmall, (maskWidth, maskHeight))
mask_rgb = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)

## Apply color to mask
color_mask, color_name = apply_color_to_mask(mask)
color_mask_rgb = cv2.cvtColor(color_mask,cv2.COLOR_BGR2RGB)
plt.imshow(color_mask_rgb)

def alphaBlend(alpha, foreground, background):
    fore = np.zeros(foreground.shape, dtype=foreground.dtype)
    fore = cv2.multiply(alpha, foreground, fore, 1 / 255.0)
    alphaPrime = np.ones(alpha.shape, dtype=alpha.dtype) * 255 - alpha
    back = np.zeros(background.shape, dtype=background.dtype)
    back = cv2.multiply(alphaPrime, background, back, 1 / 255.0)
    outImage = cv2.add(fore, back)
    return outImage

masked_lips = cv2.bitwise_and(im, im, mask=mask[:, :, 0])
output = cv2.seamlessClone(masked_lips, color_mask, mask[:, :, 0], center, cv2.MIXED_CLONE)
output_rgb = cv2.cvtColor(output,cv2.COLOR_BGR2RGB)
plt.imshow(output_rgb)

## Alpha Blending
final = alphaBlend(mask, output, im)
# cv2.putText(final, "Lipstick Color: {}".format(color_name), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
# final_image = np.hstack((im, final))
# final_image = cv2.cvtColor(final_image,cv2.COLOR_BGR2RGB)
# plt.imshow(final_image)

cv2.imshow('final_image', final)

key = cv2.waitKey(10000)