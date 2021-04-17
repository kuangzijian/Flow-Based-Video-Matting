import cv2
import numpy as np
import re, os

class DatasetGenerator():
    def __init__(self, seq = 4,
                        src_dir = 'frames/',
                        bg_dir = 'dataset/background/',
                        gt_dir = 'ground_truth/',
                        output_dir = 'dataset/mask_ouput/'):
        self.src_dir = src_dir  # Original image with green background
        self.bg_dir = bg_dir    # Background image
        self.gt_dir = gt_dir    # Mask image needed to replace the background
        self.output_dir = output_dir    # Directory to save the newly generated image
        self.seq = str(seq)

    def extract_frames_and_ground_truth(self, seq):
        # extract frames from video
        vidcap = cv2.VideoCapture(str(seq) + '.mp4')
        success, image = vidcap.read()

        count = 1

        while success:
            cv2.imwrite(self.src_dir + "original{}_{}.jpg".format(seq, count), image)
            success, image = vidcap.read()
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
                    #if True in (abs([7, 255, 41] - c) > [80, 80, 80]):
                    if True in (abs([40, 150, 50] - c) > [60, 60, 60]):
                        new_c.append([0, 0, 0])
                    else:
                        # keep the green mask pixels in the img
                        new_c.append([1, 1, 1])
                mask.append(new_c)
            mask = np.array(mask, dtype=np.uint8)

            cv2.imwrite(self.gt_dir + 'gt{}_{}.jpg'.format(seq, i+1), 255 - (255 * mask))

    def replace_background(self, original_img, mask):
        # replace background for a single image
        background_img = cv2.imread(os.path.join(self.bg_dir, self.seq+'.jpg'))
        original_img = cv2.imread(os.path.join(self.src_dir, original_img))
        background_img[np.where(mask == True)] = original_img[np.where(mask == True)]
        return background_img

    def apply_mask(self, img, mask):
        # apply mask to each frame to remove green screen background
        img = cv2.imread(os.path.join(self.src_dir, img))
        img[np.where(mask == False)] = 255
        return img

if __name__ == "__main__":
    datasetGenerator = DatasetGenerator(seq=10)
    datasetGenerator.extract_frames_and_ground_truth(10)