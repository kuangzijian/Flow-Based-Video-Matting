#!/usr/bin/env python
import torch
import math
import numpy
import PIL
import PIL.Image
import sys
import cv2
import numpy as np
import os, re
import argparse
import softsplat

try:
    from .correlation import correlation  # the custom cost volume layer
except:
    sys.path.insert(0, './correlation')
    import correlation  # you should consider upgrading python

backwarp_tenGrid = {}
backwarp_tenPartial = {}

assert (int(str('').join(torch.__version__.split('.')[0:2])) >= 13)  # requires at least pytorch version 1.3.0

# torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True  # make sure to use cudnn for computational performance

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(
            1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(
            1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([tenHor, tenVer], 1).cuda()

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones(
            [tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
    tenInput = torch.cat([tenInput, backwarp_tenPartial[str(tenFlow.shape)]], 1)

    tenOutput = torch.nn.functional.grid_sample(input=tenInput,
                                                grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1),
                                                mode='bilinear', padding_mode='zeros', align_corners=False)

    tenMask = tenOutput[:, -1:, :, :]
    tenMask[tenMask > 0.999] = 1.0
    tenMask[tenMask < 1.0] = 0.0

    return tenOutput[:, :-1, :, :] * tenMask

class PWCNet(torch.nn.Module):
    def __init__(self):
        super(PWCNet, self).__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super(Extractor, self).__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix]

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super(Decoder, self).__init__()

                intPrevious = \
                [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                    intLevel + 1]
                intCurrent = \
                [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][
                    intLevel + 0]

                if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4,
                                                                           stride=2, padding=1)
                if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(
                    in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2,
                    padding=1)
                if intLevel < 6: self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1,
                                    padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3,
                                    stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3,
                                    stride=1, padding=1)
                )

            def forward(self, tenFirst, tenSecond, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None

                    tenVolume = torch.nn.functional.leaky_relu(
                        input=correlation.FunctionCorrelation(tenFirst=tenFirst, tenSecond=tenSecond),
                        negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([tenVolume], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenFirst=tenFirst,
                                                                                                     tenSecond=backwarp(
                                                                                                         tenInput=tenSecond,
                                                                                                         tenFlow=tenFlow * self.fltBackwarp)),
                                                               negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([tenVolume, tenFirst, tenFlow, tenFeat], 1)

                tenFeat = torch.cat([self.netOne(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netTwo(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netThr(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFou(tenFeat), tenFeat], 1)
                tenFeat = torch.cat([self.netFiv(tenFeat), tenFeat], 1)

                tenFlow = self.netSix(tenFeat)

                return {
                    'tenFlow': tenFlow,
                    'tenFeat': tenFeat
                }

        class Refiner(torch.nn.Module):
            def __init__(self):
                super(Refiner, self).__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128,
                                    kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )

            def forward(self, tenInput):
                return self.netMain(tenInput)

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()
        self.load_state_dict({strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in
                              torch.hub.load_state_dict_from_url(
                                  url='http://content.sniklaus.com/github/pytorch-pwc/network-' + "default" + '.pytorch',
                                  file_name='pwc-' + "default").items()})

    def forward(self, tenFirst, tenSecond):
        tenFirst = self.netExtractor(tenFirst)
        tenSecond = self.netExtractor(tenSecond)

        objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
        objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
        objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
        objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
        objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)

        return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])

def estimate(tenFirst, tenSecond, netNetwork):
    assert (tenFirst.shape[2] == tenSecond.shape[2])
    assert (tenFirst.shape[3] == tenSecond.shape[3])

    intWidth = tenFirst.shape[3]
    intHeight = tenFirst.shape[2]

    tenPreprocessedFirst = tenFirst
    tenPreprocessedSecond = tenSecond
    # can be divided by 64
    intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
    intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

    tenPreprocessedFirst = torch.nn.functional.interpolate(input=tenPreprocessedFirst,
                                                           size=(intPreprocessedHeight, intPreprocessedWidth),
                                                           mode='bilinear', align_corners=False)
    tenPreprocessedSecond = torch.nn.functional.interpolate(input=tenPreprocessedSecond,
                                                            size=(intPreprocessedHeight, intPreprocessedWidth),
                                                            mode='bilinear', align_corners=False)

    tenFlow = 20.0 * torch.nn.functional.interpolate(input=netNetwork(tenPreprocessedFirst, tenPreprocessedSecond),
                                                     size=(intHeight, intWidth), mode='bilinear', align_corners=False)

    tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
    tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

    # print(tenFlow.shape)
    return tenFlow

def runPWC(arguments_strFirst, arguments_strSecond, netNetwork, sizes):
    img1 = PIL.Image.open(arguments_strFirst).resize(sizes)
    img2 = PIL.Image.open(arguments_strSecond).resize(sizes)


    I1 = (numpy.ascontiguousarray(
        numpy.array(img1)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                    1.0 / 255.0)))
    I2 = (numpy.ascontiguousarray(
        numpy.array(img2)[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) * (
                    1.0 / 255.0)))
    tenFirst = torch.FloatTensor([I1]).cuda()
    tenSecond = torch.FloatTensor([I2]).cuda()

    tenOutput = estimate(tenFirst, tenSecond, netNetwork)

    return tenOutput

def run_pwc_from_dir(path):
    if args.testing == False:
        output_path = 'dataset/intermediate_mask_training/input/'
    else:
        output_path = 'dataset/intermediate_mask_testing/input/'

    netNetwork = PWCNet().cuda().eval()
    alphanum_key = lambda key: [int(re.split('_', key)[1].split('.')[0])]
    files = sorted(os.listdir(path), key=alphanum_key)

    sizes = [PIL.Image.open(os.path.join(path, f), 'r').size for f in files]
    sizes = max(sizes)
    print("Max size:", sizes)

    for i in range((len(files) - 1)):
        w, h = sizes
        img1 = os.path.join(path, files[i])
        img2 = os.path.join(path, files[i + 1])
        print("Working on the optical flow between: {} and {}".format(files[i], files[i+1]))

        tenFlow_raw = runPWC(img1, img2, netNetwork, sizes)
        tenFlow = np.array(tenFlow_raw[0].detach().cpu().numpy(), np.float32)

        mag = tenFlow[0, :, :]
        # split the moving object and background using a threshold
        mag = [[0 if x < args.threshold else 255 for x in y] for y in mag]
        ang = tenFlow[1, :, :]
        hsv = np.zeros((h, w, 3), numpy.float32)
        #hsv[..., 1] = 255
        #hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = mag
        if i == 0:
            hsv[..., 2] = mag
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            temp = rgb

            # duplicate the first optical flow generated for the first frame
            cv2.imwrite(output_path + 'intmask_' + str(1) + '.png', rgb)
            cv2.imwrite(output_path + 'intmask_' + str(2) + '.png', rgb)

        else:
            hsv[..., 2] = mag
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            # get warpped using previous rgb and the current rgb
            warpped = softsplat_warp(torch.FloatTensor([temp.transpose(2, 0, 1)]).cuda(),
                           torch.FloatTensor([rgb.transpose(2, 0, 1)]).cuda(), tenFlow_raw)

            # assign the 0 valued element in magnitude channel with value of warpped
            for r_index, r in enumerate(mag):
                for c_index, c in enumerate(r):
                    if c < args.threshold:
                        mag[r_index][c_index] = warpped[0][0][r_index][c_index]
                    else:
                        mag[r_index][c_index] = 255
            hsv[..., 2] = mag
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            temp = rgb
            cv2.imwrite(output_path + 'intmask_' + str(i + 2) + '.png', rgb)
        i += 1

def estimate_optical_flow(org_img_path, mask_threshold = 0.5):
    print("Estimating optical flows for all input imgs...")
    alphanum_key = lambda key: [int(re.split('_', key)[1].split('.')[0])]
    img_files = sorted(os.listdir(org_img_path), key=alphanum_key)
    pwcNetwork = PWCNet().cuda().eval()
    int_masks = []
    for i in range(len(img_files)):
        if 'png' in img_files[i] or 'jpg' in img_files[i] or 'bmp' in img_files[i]:
            org_img = PIL.Image.open(os.path.join(org_img_path, img_files[i].split('.')[0] + '.jpg'))
            if i == 0:
                print("Working on the optical flow between: {} and {}".format(img_files[i], img_files[i]))
                # for the first frame, since there is no previous frame, we estimate the optical flow using it self
                previous_img = img = os.path.join(org_img_path, img_files[i])
            else:
                print("Working on the optical flow between: {} and {}".format(img_files[i - 1], img_files[i]))
                # we estimate the optical flow for each two frames
                previous_img = os.path.join(org_img_path, img_files[i - 1])
                img = os.path.join(org_img_path, img_files[i])

            # estimate the optical flow
            tenFlow_raw = runPWC(previous_img, img, pwcNetwork, org_img.size)
            tenFlow = np.array(tenFlow_raw[0].detach().cpu().numpy(), np.float32)
            w, h = org_img.size
            mag = tenFlow[0, :, :]

            # split the moving object and background using a mask threshold
            mag = [[0 if x < mask_threshold else 255 for x in y] for y in mag]
            hsv = np.zeros((h, w, 3), numpy.float32)
            hsv[..., 2] = mag
            int_mask = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            int_mask = PIL.Image.fromarray(np.uint8(int_mask))
        int_masks.append(int_mask)

    return int_masks

def softsplat_backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(
            tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(
            tenFlow.shape[0], -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([tenHorizontal, tenVertical], 1).cuda()

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)

    return torch.nn.functional.grid_sample(input=tenInput,
                                           grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1),
                                           mode='bilinear', padding_mode='zeros', align_corners=True)

def softsplat_warp(tenFirst, tenSecond, tenFlow, fltTime=1):
    tenMetric = torch.nn.functional.l1_loss(input=tenFirst, target=softsplat_backwarp(tenInput=tenSecond, tenFlow=tenFlow),
                                            reduction='none').mean(1, True)
    tenSoftmax = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow * fltTime, tenMetric=-20.0 * tenMetric,
                                             strType='softmax')

    return tenSoftmax

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", default='', help="Directory of the dataset.")
    parser.add_argument("-th", "--threshold", default=0.5, help="Threshold to split moving object and background")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Generate training dataset for UNet",
                        default=False)
    args = parser.parse_args()
    path = args.dataset
    if args.dataset == '':
        if args.testing == False:
            path = 'dataset/original_training'
        else:
            path = 'dataset/original_testing'
    if os.path.isdir(path):
        run_pwc_from_dir(path)
    else:
        print("Dataset path is invalid.")
