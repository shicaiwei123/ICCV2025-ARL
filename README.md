# ICCV2025-ARL

Here is the official code for "Improving Multimodal Learning via Imbalanced Learning", which is a flexible framework to enhance the optimization process of multimodal learning. Please refer to our [ICCV 2025 paper](https://arxiv.org/pdf/2507.10203) for more details.


## Main Dependencies
+ Ubuntu 20.04
+ CUDA Version: 11.3
+ PyTorch 1.11
+ python 3.8.6
+ **note** The optimal hyperparameter ($\gamma$) for ARL may vary with the dependencies. If your equipment and software are different, you may need to adjust the hyperparameters accordingly.
+ **note** PyTorch 2.0 and the late version may lead to significant performance decline. The reason is unclear.


## Usage
### Data Preparation
Download Dataset：
[CREMA-D](https://pan.baidu.com/s/11ISqU53QK7MY3E8P2qXEyw?pwd=4isj), [Kinetics-Sounds](https://pan.baidu.com/s/1E9E7h1s5NfPYFXLa1INUJQ?pwd=rcts).
Here we provide the processed dataset directly. 

The original dataset can be seen in the following links,
[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D),
[Kinetics-Sounds](https://github.com/cvdfoundation/kinetics-dataset).
[VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/),

 And you need to process the dataset following the instruction below.

### Pre-processing

For CREMA-D and VGGSound dataset, we provide code to pre-process videos into RGB frames and audio wav files in the directory ```dataset/data/```.

#### CREMA-D 

As the original CREMA-D dataset has provided the original audio and video files, we simply extract the video frames by running the code:

```python dataset/data/CREMAD/video_preprecessing.py```

Note that, the relevant path/dir should be changed according your own env.  

## Data path

you should move the download dataset into the folder *train_test_data*, or make a soft link in this floder.


## Key code of ARL

- Modality analysis
  - determine the weight for ARL (see line 114 in main_arl_variance.py)

- Asymmetric Learning

  - The class of GradScale in backbone.py to modify the backward graident (Line 71 in models/backbone.py)

  - Using GradScale in the backbone (after modality encoder)  to modify the graident (Line 325 and 345 in models/backbone.py)

- Unimodal Regularization
  - Add unimodal loss to enhance the modality encoder via parameters-shared module. （see **_AUXI in models/fusion_modules.py）


## Train 

We provide bash file for a quick start.

For CREMA-D

```bash
bash cramed_arl.sh
```
For AVE 

```bash
bash ave_arl.sh
```

For KS 

```bash
bash ks_arl.sh
```


## Test

```python
python valid.py
```

## Contact us

If you have any detailed questions or suggestions, you can email us:
**shicaiwei@std.uestc.edu.cn**
