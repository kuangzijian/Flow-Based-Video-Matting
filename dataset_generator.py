import cv2
import numpy as np

# extract frames from video
vidcap = cv2.VideoCapture('4.mp4')
success, image = vidcap.read()
count = 1
while success:
    cv2.imwrite("frames/original4_%d.jpg" % count, image)
    sucess, image = vidcap.read()
    print('Read a new frame: ', success)
    count +=1

# generate ground truth mask for each frame
for i in range(1000):
    img = cv2.imread('frames/original4_' + str(i + 1) + '.jpg')
    print('Generating mask for : ', i)
    mask = []
    for r in img:
        new_c = []
        for c in r:
            # check if the pixel value is green mask or original image pixels
            if True in (abs([7, 255, 41] - c) > [60, 60, 60]):
            #if True in (abs([40, 150, 50] - c) > [60, 60, 60]):
                new_c.append([0, 0, 0])
            else:
                # keep the green mask pixels in the img
                new_c.append([1, 1, 1])
        mask.append(new_c)
    mask = np.array(mask, dtype=np.uint8)

    cv2.imwrite('ground_truth/gt4_' + str(i + 1) + '.jpg', 255 - (255 * mask))

# apply mask to each frame to remove green screen background
for i in range(1000):
    img = cv2.imread('frames/original4_' + str(i + 1) + '.jpg')
    mask = cv2.imread('ground_truth/gt4_' + str(i + 1) + '.jpg')
    mask = mask / 255  # make the mask into 0/1 matrix for multiplication
    masked_img = img * mask
    cv2.imwrite('after_mask/original4_' + str(i + 1) + '.jpg', masked_img)
