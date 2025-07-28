# CMFA : Cross Modal Fusion Attention

## Installation
```
# Install conda env
conda create -n cmfa python=3.9 -y

# Install pachkages for the CMFA
pip install -r requirements.txt
```

## Train
Please change the dataset path in `./Config/defaults.py`
```
python train.py
```
## Test
Please change the pre-trained weight path in `./Config/defaults.py`

## E-SIE Dataset
- Download link : [E-SIE](https://drive.google.com/file/d/1qpd5KjZfV-gfz2qC23xpIp2pZR3hJg9N/view?usp=sharing)

## Pretrained Model
- Pretrained weight :

```
python test.py
```

## Acknowledgements
This repository borrows or partially modifies the models from [SLPT](https://github.com/Jiahao-UTS/SLPT_Training).
