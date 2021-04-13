# Flow-based-Video-Matting-Algorithm

This is the repository to the paper "Flow based video matting" by Zijian Kuang and Xinran Tie.
We proposed to use optical flow and UNet to generate a refined mask for video matting purpose.

<a href="https://arxiv.org/abs/1709.02371" rel="Paper"><img src="https://github.com/kuangzijian/Flow-Based-Video-Matting/blob/master/readme_imgs/network.png" alt="Paper" width="100%"></a>

## Getting Started

You will need [Python 3.6](https://www.python.org/downloads) and the packages specified in _requirements.txt_.
We recommend setting up a [virtual environment with pip](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
and installing the packages there.
The correlation layer is implemented in CUDA using CuPy, which is why CuPy is a required dependency. It can be installed using pip install cupy or alternatively using one of the provided binary packages as outlined in the CuPy repository.

Install packages with:

```
$ pip install -r requirements.txt
```

Or install with for Windows as per [PyTorch official site](https://pytorch.org/get-started/locally/):

```
$ pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_s
table.html
$ pip install -r requirements.txt
```

## Dataset

We created our own video matting dataset. The dataset includes four online conference style green screen videos. We extracted the data from video and generated ground truth mask for each character, and then we applied virtual background to the frames as our training/testing dataset. You can download the dataset from <a href="https://drive.google.com/file/d/1HGUDS7oaYbBAJHfsQJBhc-r-9kRhh2cs/view?usp=sharing" rel="dataset"> this link <a/>. The data examples are shown as below:

 Input image 1:![1](https://github.com/kuangzijian/Flow-Based-Video-Matting/blob/master/readme_imgs/with_bg.jpg) | Input image 2: ![2](https://github.com/kuangzijian/Flow-Based-Video-Matting/blob/master/readme_imgs/with_bg2.jpg)
:-------------------------:|:-------------------------:
Ground truth 1:![3](https://github.com/kuangzijian/Flow-Based-Video-Matting/blob/master/readme_imgs/ground_truth.jpg) | Ground truth 2:![4](https://github.com/kuangzijian/Flow-Based-Video-Matting/blob/master/readme_imgs/ground_truth2.jpg)

To use our code to generate more video matting data and groudtruth, you can use the functions in _dataset_generator.py_


## Configure and Run the Code
To train our model: 
 1. Create folder structure like the example shows in the picture below, and then dump the training data into the _original_training_ folder, and dump the ground truth data into the _ground_truth_training_ folder:
 
  ![1](https://github.com/kuangzijian/Flow-Based-Video-Matting/blob/master/readme_imgs/dataset_structure.png)
  
 2. Run the training code:

``` 
python funet_train.py

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 6)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 1)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 20.0)
``` 
To predict using our model:
 1. Dump the testing data into the _original_testing_ folder, and dump the ground truth data into the _ground_truth_testing_ folder.
 2. Run the predicting code:

``` 
python funet_predict.py

optional arguments:
  -h, --help            show this help message and exit
  --model FILE, -m FILE
                        Specify the file in which the model is stored
                        (default: MODEL.pth)
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        filenames of input images (default: None)
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of ouput images (default: None)
  --viz, -v             Visualize the images as they are processed (default:
                        False)
  --no-save, -n         Do not save the output masks (default: False)
  --mask-threshold MASK_THRESHOLD, -t MASK_THRESHOLD
                        Minimum probability value to consider a mask pixel
                        white (default: 0.5)
  --scale SCALE, -s SCALE
                        Scale factor for the input images (default: 1)
``` 


## Citation
```
[1]  @inproceedings{Sun_CVPR_2018,
         author = {Deqing Sun and Xiaodong Yang and Ming-Yu Liu and Jan Kautz},
         title = {{PWC-Net}: {CNNs} for Optical Flow Using Pyramid, Warping, and Cost Volume},
         booktitle = {IEEE Conference on Computer Vision and Pattern Recognition},
         year = {2018}
     }
```

```
[2]  @misc{pytorch-pwc,
         author = {Simon Niklaus},
         title = {U-Net: Convolutional Networks for Biomedical Image Segmentation},
         year = {2015},
         howpublished = {\url{https://arxiv.org/abs/1505.04597}}
    }
``` 

```
[3]  @misc{U-Net,
         author = {Olaf Ronneberger, Philipp Fischer, Thomas Brox},
         title = {A Reimplementation of {PWC-Net} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-pwc}}
    }
```  

## License

This project is licensed under the MIT License.
