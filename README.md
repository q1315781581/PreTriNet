# PreTriNet: A Hybrid Framework for Efficient and Accurate Semi-Supervised Volumetric Medical Image Segmentation

## 1. Environment

First, create a new environment and install the requirements:
```shell
conda create -n pretrinet python=3.8
conda activate pretrinet
cd PreTriNet/
pip install -r requirements.txt
```

Then, before running the code, set the `PYTHONPATH` to `pwd`:
```shell
export PYTHONPATH=$(pwd)/code:$PYTHONPATH
```

## 2. Data Preparation

First, download the datasets and put them under the `Datasets` folder:
- **LASeg dataset** for SSL: download the preprocessed data from https://github.com/yulequan/UA-MT/tree/master/data. 

- **MMWHS dataset** for UDA: download according to https://github.com/cchen-cc/SIFA#readme. **Or download the preprocessed data via [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hwanggr_connect_ust_hk/Evzk4w-LpoVFgKwa9dwl38EBR_szwDKITwJE0nOue1pLvw?e=joo4ei).**

The file structure should be: 
```shell
.
├── Datasets
│   ├── LASeg
│   │   ├── 2018LA_Seg_Training Set
│   │   │   ├── 0RZDK210BSMWAA6467LU
│   │   │   │   ├── mri_norm2.h5
│   │   │   ├── 1D7CUD1955YZPGK8XHJX
│   │   │   └── ...
│   │   ├── test.list
│   │   └── train.list
│   ├── MMWHS
│   │   ├── CT
│   │   │   ├── imagesTr
│   │   │   │   ├── ct_train_1001_image.nii.gz
│   │   │   │   └── ...
│   │   │   └── labelsTr
│   │   │   │   ├── ct_train_1001_label.nii.gz
│   │   │   │   └── ...
│   │   └── MR
│   │       ├── imagesTr
│   │       └── labelsTr
```



### 2.1 Pre-process LASeg dataset
Run `python ./code/data/preprocess_la.py` to:
- convert `.h5` files to `.npy`.
- generate the labeled/unlabeled splits

### 2.2 Pre-process MMWHS dataset
Run `python ./code/data/preprocess_mmwhs.py` to:
- reorient to the same orientation, RAI;
- convert to continuous labels;
- crop centering at the heart region; 
- for each 3D cropped image top 2/% of its intensity histogram was cut off for alleviating artifacts;
- resize and convert to `.npy`;
- generate the train/validation/test splits.

For all the pre-processing, you can comment out the functions corresponding to splits and use our pre-split files.


Finally, you will get a file structure as follow:
```shell
.
├── LA_data
│   └── ...
└── MMWHS_data
    └── ...
```



## 3. Teacher Model Pre-training

**This codebase allows pre-training teacher models (GSG weights) for all tasks using one single script.** 


Run the following command to pre-train a teacher model:

```shell
python code/train_teacher_two_stage.py \
  --task <task_name> \
  --split_labeled <labeled_split_path> \
  --split_eval <eval_split_path> \
  --teacher_ckpt anatomix.pth \
  --exp train_teacher_two_stage/<exp_name> \
  --gpu 0
```

Parameters:
`--task`: task name, e.g., la, mmwhs_ct2mr, or mmwhs_mr2ct
`--split_labeled`: path to labeled split file, e.g., ./LA_data/split_txts/labeled_0.05, ./LA_data/split_txts/labeled_0.1, ./MMWHS_data/split_txts/train_ct2mr_labeled, or ./MMWHS_data/split_txts/train_mr2ct_labeled
`--split_eval`: path to evaluation split file, e.g., ./LA_data/split_txts/eval_0.05, ./LA_data/split_txts/eval_0.1, ./MMWHS_data/split_txts/eval_ct2mr, or ./MMWHS_data/split_txts/eval_mr2ct
`--teacher_ckpt`: path to the pre-trained base model
`--exp`: experiment name and save directory
`--gpu`: use which gpu to train



## 4. Training & Testing & Evaluating

**This codebase allows train, test, and evaluate on all the four settings using one single bash file.** 

Run the following commands for training, testing and evaluating.

```shell
bash train.sh -c 0 -e diffusion -t <task> -i '' -l 1e-2 -w 10 -n 300 -d true -k ./logs/train_teacher_two_stage/<exp_name>/ckpts/best_teacher_final.pth --g false
```

Parameters:

`-c`: use which gpu to train

`-e`: use which training script, can be `diffusion` for `train_diffusion.py`

`-t`: switch to different tasks:  
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; For SSL on `5%` labeled LA dataset: `la_0.05`   
&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; For UDA on MMWHS dataset: `mmwhs_ct2mr` for labeled CT and unlabeled MR, `mmwhs_mr2ct` in opposite    

`-i`: name of current experiment, can be whatever you like

`-l`: learning rate

`-w`: weight of unsupervised loss

`-n`: max epochs

`-d`: whether to train, if `true`, training -> testing -> evaluating; if `false`, testing -> evaluating

`-k`: path to pretrained teacher model checkpoint

`-g`: enable debug mode for development


