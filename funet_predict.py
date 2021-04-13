import argparse
import logging
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import numpy
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from dice_loss import dice_coeff
from model import FlowUNetwork
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from pwc_net import runPWC, PWCNet
from dataset_generator import DatasetGenerator

def predict_img(net,
                int_mask,
                org_img,
                device,
                scale_factor=0.5,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess_input_with_int_mask(int_mask, org_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(int_mask.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask, full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/CP_epoch6.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--img', '-img', default='dataset/original_training/', metavar='INPUT', nargs='+',
                        help='path of original image dataset')
    parser.add_argument('--mask', '-mask', default='dataset/ground_truth_training/', metavar='INPUT', nargs='+',
                        help='path of ground truth mask dataset')
    parser.add_argument('--output', '-o', default='dataset/mask_output/', metavar='INPUT', nargs='+',
                        help='path of ouput dataset')
    parser.add_argument('--no-viz', '-v', action='store_true',
                        help="No visualize the dataset as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input dataset",
                        default=1)
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='Set the gpu for cuda')

    return parser.parse_args()


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    org_img_path = args.img
    gt_mask_path = args.mask
    net = FlowUNetwork(n_channels=4, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(args.gpu.split(','))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    print("Model loaded !")
    alphanum_key = lambda key: [int(re.split('_', key)[1].split('.')[0])]
    img_files = sorted(os.listdir(org_img_path), key=alphanum_key)
    true_masks = sorted(os.listdir(gt_mask_path), key=alphanum_key)
    i = 0
    pwcNetwork = PWCNet().cuda().eval()

    datasetGenerator = DatasetGenerator(src_dir=org_img_path)
    if not args.no_viz:
        plt.ion()
        fig, ax = plt.subplots(2, 2, figsize=(8, 4))
        plt.show()

    tot = 0
    while i < len(img_files):
        true_mask = Image.open(os.path.join(gt_mask_path, true_masks[i].split('.')[0] + '.jpg')).convert('L')
        logging.info("\nPredicting image {} ...".format(img_files[i]))
        print("\nPredicting image {} ...".format(img_files[i]))
        if 'png' in img_files[i] or 'jpg' in img_files[i] or 'bmp' in img_files[i]:
            org_img = Image.open(os.path.join(org_img_path, img_files[i].split('.')[0] + '.jpg'))
            if i == 0:
                # for the first frame, since there is no previous frame, we estimate the optical flow using it self
                previous_img = img = os.path.join(org_img_path, img_files[i])
            else:
                # we estimate the optical flow for each two frames
                previous_img = os.path.join(org_img_path, img_files[i-1])
                img = os.path.join(org_img_path, img_files[i])

            # estimate the optical flow
            tenFlow_raw = runPWC(previous_img, img, pwcNetwork, org_img.size)
            tenFlow = np.array(tenFlow_raw[0].detach().cpu().numpy(), np.float32)
            w, h = org_img.size
            mag = tenFlow[0, :, :]

            # split the moving object and background using a mask threshold
            mag = [[0 if x < args.mask_threshold else 255 for x in y] for y in mag]
            hsv = np.zeros((h, w, 3), numpy.float32)
            hsv[..., 2] = mag
            int_mask = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)
            int_mask = Image.fromarray(np.uint8(int_mask))

            # predict mask using flowUNet
            mask_pred, mask = predict_img(net=net,
                               int_mask=int_mask,
                               org_img=org_img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)
            mask_pred = torch.from_numpy(mask_pred).type(torch.FloatTensor)
            mask_pred = mask_pred.to(device=device, dtype=torch.float32)
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            true_mask = BasicDataset.preprocess(true_mask, 1)
            true_mask = torch.from_numpy(true_mask).type(torch.FloatTensor)
            true_mask = true_mask.to(device=device, dtype=torch.float32)
            dc = dice_coeff(pred.unsqueeze(0).unsqueeze(0), true_mask.unsqueeze(0)).item()
            tot += dc
            print("Dice Coefficient: " + str(dc))

            if not args.no_save:
                output_file = args.output + 'output_' + img_files[i]
                result = mask_to_image(mask)
                result.save(output_file)

                # replace background based on mask and provided bg
                removed_bg = datasetGenerator.apply_mask(img_files[i], mask)
                new_img = datasetGenerator.replace_background(img_files[i], mask)
                cv2.imwrite(args.output + 'new_' + img_files[i], new_img)

                logging.info("Mask saved to {}".format(output_file))

            if not args.no_viz:
                logging.info("Visualizing results for image {}, close to continue ...".format(img_files[i]))
                plot_img_and_mask(plt, ax, org_img, mask, removed_bg, new_img)

        i += 1

    print('Averaging Dice Coefficient on ' + str(len(img_files)) + ' testing frames: ' + str(tot/len(img_files)))