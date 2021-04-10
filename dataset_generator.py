import cv2
import numpy as np
import re

class DatasetGenerator():
    def __init__(self, seq, src_dir = 'dataset/video_matting_dataset/frames/',
                        bg_dir = 'dataset/background/',
                        gt_dir = 'dataset/video_matting_dataset/ground_truth/',
                        pred_mask_dir = 'dataset/intermediate_mask_testing/input/',
                        output_dir = 'dataset/original_testing/'):
        self.src_dir = src_dir  # Original image with green background
        self.bg_dir = bg_dir    # Background image
        self.gt_dir = gt_dir    # Mask image needed to replace the background
        self.pred_mask_dir = pred_mask_dir  # Predicted masks by the model
        self.output_dir = output_dir    # Directory to save the newly generated image
        self.seq = seq

    def extract_frames_and_ground_truth(self, seq):
        # extract frames from video
        vidcap = cv2.VideoCapture(seq + '.mp4')
        success, image = vidcap.read()
        count = 1
        while success:
            cv2.imwrite(self.src_dir + "original{}_{}.jpg".format(seq, count), image)
            sucess, image = vidcap.read()
            print('Read a new frame: ', success)
            count += 1

        # generate ground truth mask for each frame
        for i in range(count):
            img = cv2.imread(self.src_dir + 'original{}_{}.jpg'.format(seq, i+1))
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

            cv2.imwrite(self.gt_dir + 'gt{}_{}.jpg'.format(seq, i+1), 255 - (255 * mask))

    def replace_background(self, original_img, ground_truth, background_img):
        # replace background for a single image
        idx = re.split('_', original_img)[1].split('.')[0]
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(ground_truth, cv2.MORPH_CLOSE, kernel)
        background_img[np.where(opening == 255)] = original_img[np.where(opening == 255)]

        print("Replacing the background for original_{}.jpg".format(idx))

        cv2.imwrite(self.output_dir + 'original_{}.jpg'.format(idx), background_img)