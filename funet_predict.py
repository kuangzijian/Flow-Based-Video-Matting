import argparse
import logging
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
import numpy
import cv2
from PIL import Image
from torchvision import transforms

from model import FlowUNetwork
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from pwc_net_predict import runPWC, PWCNet

def predict_img(net,
                int_mask,
                org_img,
                device,
                scale_factor=0.5,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess_input(int_mask, org_img, scale_factor))

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

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/CP_epoch6.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--img', '-img', default='dataset/original_training/', metavar='INPUT', nargs='+',
                        help='path of original image dataset')
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

    return parser.parse_args()


def get_output_filenames(inputs, output_path):
    in_files = inputs
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    org_img_path = args.img
    net = FlowUNetwork(n_channels=4, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    print("Model loaded !")
    alphanum_key = lambda key: [int(re.split('_', key)[1].split('.')[0])]
    img_files = sorted(os.listdir(org_img_path), key=alphanum_key)
    i = 0
    pwcNetwork = PWCNet().cuda().eval()
    while i < len(img_files):
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
            mask = predict_img(net=net,
                               int_mask=int_mask,
                               org_img=org_img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            if not args.no_save:
                output_file = args.output + 'output_' + img_files[i]
                result = mask_to_image(mask)
                result.save(output_file)

                logging.info("Mask saved to {}".format(output_file))

            if not args.no_viz:
                logging.info("Visualizing results for image {}, close to continue ...".format(img_files[i]))
                plot_img_and_mask(org_img, mask)
                # todo: replace background based on mask and provided bg.
                # todo: using openCV to display original image, predicted mask and new image with replaced bg.
        i += 1