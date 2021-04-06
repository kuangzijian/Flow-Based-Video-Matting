import argparse
import logging
import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

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
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input dataset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='model/model.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i',  default='dataset/unet_testing/', metavar='INPUT', nargs='+',
                        help='path of input dataset')

    parser.add_argument('--output', '-o', default='dataset/unet_testing_outputs/', metavar='INPUT', nargs='+',
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
                        default=0.5)

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
    path = args.input
    net = UNet(n_channels=3, n_classes=1)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")
    alphanum_key = lambda key: [int(re.split('_', key)[1].split('.')[0])]
    files = sorted(os.listdir(path), key=alphanum_key)
    i = 0

    while i < (len(files) - 1):
        logging.info("\nPredicting image {} ...".format(files[i]))
        print("\nPredicting image {} ...".format(files[i]))
        if 'png' in files[i] or 'jpg' in files[i]:
            img = Image.open(os.path.join(path, files[i]))

            mask = predict_img(net=net,
                               full_img=img,
                               scale_factor=args.scale,
                               out_threshold=args.mask_threshold,
                               device=device)

            if not args.no_save:
                output_file = args.output + 'output_' + files[i]
                result = mask_to_image(mask)
                result.save(output_file)

                logging.info("Mask saved to {}".format(output_file))

            if not args.no_viz:
                logging.info("Visualizing results for image {}, close to continue ...".format(files[i]))
                plot_img_and_mask(img, mask)
        i += 1