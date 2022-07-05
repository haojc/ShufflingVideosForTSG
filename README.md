# ToCVPR2022: Can Shuffling Video Benefit Temporal Bias Problem: A Novel Training Framework for Temporal Grounding

## Installation
We provide the environment file for anaconda.

You can build the conda environment simply by,
```bash
conda env create -f environment.yml
```

## Dataset Preparation
Sorry for that we cannot provide our features due to the space limitation of supplement materials and the double-blind policy.
We will open our features after th review process of CVPR22.

For Charades-STA, you can download the i3d features from the implementation of [VSLNet](https://github.com/IsaacChanghau/VSLNet).
But because they have modified their features recently, the downloaded features may do not fit to our pre-trained model.
You can re-train the model using these features.
 
For ActivityNet Captions, you need to extract the i3d features from the original videos using an open implementation of [I3D](https://github.com/piergiaj/pytorch-i3d), with stride 16 and fps 16.

All the features should be the format of 'VIDEO_ID.npy'.

Please put the features into the directory data/Charades/i3d_feature and data/ANet/i3d_feature.

## Quick Strat
cd grounding

###Charades-CD

Train:
```
python train.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias one_name
```
Evaluate:
```
python test.py --gpu_id=0 --cfg charades_cd_i3d.yml --alias test
```

You can change the model to be evaluated in the corresponding config file. By default, test.py will use the pre-trained model provided by us.

###ActivityNet-CD

Train:
```
python train.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias one_name
```
Evaluate:
```
python test.py --gpu_id=0 --cfg anet_cd_i3d.yml --alias test
```

##Models

Due the space limitation, we only provide the pre-trained model for Charades-CD in ckp/charades_cd/***.ckp.
You can find the corresponding prediction results, parameter settting, and training/evaluation log files in this path for both datasets.
