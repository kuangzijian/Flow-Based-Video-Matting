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

We created our own video matting dataset. The dataset includes four online conference style green screen videos. We extracted the data from video and generated ground truth mask for each character, and then we applied virtual background to the frames as our training/testing dataset. You can download the dataset from <a href="https://drive.google.com/file/d/1HGUDS7oaYbBAJHfsQJBhc-r-9kRhh2cs/view?usp=sharing" rel="dataset"> this link <a/>.

If you encounter GPU Out of Memory issue, you can reduce the neuron numbers in _config.py_
```
fc_internal = 1536 # number of neurons in hidden layers of s-t-networks
```

To start the training, just run _main.py_ as follows! If training on the dummy data does not lead to an AUROC of 1.0, something seems to be wrong.
Please report us if you have issues when using the code.

```
$ python main.py
```

## Configure and Run the Code
How to use Data extraction tool to extract data from video clips:
 1. Create folder structure like the example shows in the picture below.
 
  ![1](https://github.com/zerobox-ai/differnet/blob/zijian/dataset/data-generation/annotations/structure1.png)
  
 2. Dump the videos and annotations (rename them use 1.xml, 1.avi as one pair annotation and video) into the folders under data-generation folder.
 
  ![2](https://github.com/zerobox-ai/differnet/blob/zijian/dataset/data-generation/annotations/structure2.png)
  
 3. Modify the annotation files: Since the annotation uses label "defect" to indicate the defect area, while, both good and defective bottles are labeled as "bottle" which is confusing. To indicate which "bottle" is defective, we need to find the frames that labeled with defect, and then manully update the group's label from "bottle" to "defective" for the groups that falling in to those frames. 
  ![3](https://github.com/zerobox-ai/differnet/blob/zijian/dataset/data-generation/annotations/structure3.png)
  
 - For example: in the example image above, the frame 15 and 16 are labeled as "defect" which indicates those 2 frames has defect areas on the bottles. So we need to find the group that contains frame 15 and 16, and then manully update the label from "bottle" to "defective". and then delete the whole \<track\> group that labeled as "defect" (since we don't care about the defect area in data extraction).
 
 4. Modify the config.py, fill in appropriate value for num_videos, save_cropped_image_to and save_original_image_to
 
 5. run the data extraction: python data_extraction.py


The given dummy dataset shows how the implementation expects the construction of a dataset. Coincidentally, the [MVTec AD dataset](https://www.mvtec.com/de/unternehmen/forschung/datasets/mvtec-ad/) is constructed in this way.

Set the variables _dataset_path_ and _class_name_ in _config.py_ to run experiments on a dataset of your choice. The expected structure of the data is as follows:

``` 
train data:

        dataset_path/class_name/train/good/any_filename.png
        dataset_path/class_name/train/good/another_filename.tif
        dataset_path/class_name/train/good/xyz.png
        [...]

test data:

    'normal data' = non-anomalies

        dataset_path/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
        dataset_path/class_name/test/good/did_you_know_the_image_extension_webp?.png
        dataset_path/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
        dataset_path/class_name/test/good/dont_know_how_it_is_with_windows.png
        dataset_path/class_name/test/good/just_dont_use_windows_for_this.png
        [...]

    anomalies - assume there are anomaly classes 'crack' and 'curved'

        dataset_path/class_name/test/crack/dat_crack_damn.png
        dataset_path/class_name/test/crack/let_it_crack.png
        dataset_path/class_name/test/crack/writing_docs_is_fun.png
        [...]

        dataset_path/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
        dataset_path/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
        [...]
``` 

## Credits

Some code of the [FrEIA framework](https://github.com/VLL-HD/FrEIA) was used for the implementation of Normalizing Flows. Follow [their tutorial](https://github.com/VLL-HD/FrEIA) if you need more documentation about it.


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
         title = {A Reimplementation of {PWC-Net} Using {PyTorch}},
         year = {2018},
         howpublished = {\url{https://github.com/sniklaus/pytorch-pwc}}
    }
```  

## License

This project is licensed under the MIT License.
