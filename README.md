# Can Shuffling Video Benefit Temporal Bias Problem for Temporal Grounding

Code for ECCV 2022 paper "Can Shuffling Video Benefit Temporal Bias Problem: A Novel Training Framework for Temporal Grounding"


## Installation
We provide the environment file for anaconda.

You can build the conda environment simply by,
```bash
conda env create -f environment.yml
```

## Dataset Preparation
#### Features and Pretrained Models
You can download our features for Charades-STA and ActivityNet Captions and the pretrained models of our method 
on re-divided splits by an anonymous box drive [link](https://app.box.com/s/t0e3uu8qhpvbpo70qheb7w7i1lj8frqc).

(For ActivityNet Captions, we extract the i3d features from the original videos 
using an open implementation of [I3D](https://github.com/piergiaj/pytorch-i3d), with stride 16 and fps 16.)

Please put the video feature files 'VID.npy' into the directories
`data/Charades/i3d_feature` and `data/ANet/i3d_feature`, respectively.

Please put the pretrianed models into the directories `grounding/ckp/charades_cd` and `grounding/ckp/anet_cd`, respectively.
#### Word Embeddings
For Charades-STA, we directly provide the word embeddings files in this github repositories. You don't need to do anything else.

For ActivityNet Captions, due to the limitation of the file size of github,
you need to download the word embeddings from the [link](https://app.box.com/s/t0e3uu8qhpvbpo70qheb7w7i1lj8frqc), 
and put the word embeddings into the directory `data/ANet/words`.


## Quick Start
```
conda activate HLTI
cd grounding
```


### Charades-CD

Train:
```
python train.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias one_name
```
The checkpoints and prediction results will be saved in `grounding/runs/DATASET/`

Evaluate:
```
python test.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias test
```

You can change the model to be evaluated in the corresponding config file. By default, test.py will use the pre-trained model provided by us.

### ActivityNet-CD

Train:
```
python train.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias one_name
```
Evaluate:
```
python test.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias test
```

### About Pretrained Models

We provide the corresponding prediction results, parameter setting, and evaluation result files
in `grounding/ckp` for both datasets.

## Baseline

We also provide the implementation of the baseline (QAVE).

### Charades-CD

Train:
```
python train_baseline.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias one_name
```
Evaluate:
```
python test_baseline.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias test
```

Please determine the model to be evaluated in the corresponding config file.

### ActivityNet-CD

Train:
```
python train_baseline.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias one_name
```
Evaluate:
```
python test_baseline.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias test
```