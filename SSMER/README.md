# SSMER

## Anaconda virtual env setup
```
# Install conda env
conda create -n ssmer python=3.10
# press y :)

# Install pachkages for the SSMER
pip install -r requirements.txt
```

## dataset directory structure
- frame representation path
  - /your dataset directory path/frame
- voxel representation path
  - /your dataset directory path/voxel
- timesurface representation path
  - /your dataset directory path/timesurface
- move train_val_test_split_multilabel.json & train_val_test_split_multilabel_mod.json to your dataset path

## Train
options
- -r: event representation
  - f: frame
  - v: voxel
  - t: timesurface
- --epoch: training epochs
- -lr: learning rate
- --gpu: gpu number
- - -b: batch size
- --save-folder: saving path
- dataset folder: change the path to your dataset path
- --pretrained: path for the pretrained weight
```
# train code for the SSMER
python ssmer_split.py -r frame,voxel,timesurface --epochs 200 --lr 0.05 --gpu 0 -b 64 --save-folder ./checkpoint/v2e_max_hrnet/frame_voxel_timesurface_0.05 /mnt/ssd1/data/celebvhq_scene_v2e
```

## Acknowledgements
This repository borrows or partially modifies the models from [SimSiam](https://github.com/facebookresearch/simsiam)
