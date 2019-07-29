# Deep Meta Metric Learning (DMML)
This repo contains PyTorch code for ICCV19' paper: Deep Meta Metric Learning, including person re-identification experiments on Market-1501 and DukeMTMC-reID datasets.

## Requirements
- Python 3.6+
- PyTorch 0.4
- tensorboardX 1.6

To install all python packages, please run the following command:
```
pip install -r requirements.txt
```

## Usage
### Training
After adding dataset directory in `demo.sh`, simply run the following command to train DMML on Market-1501:
```
bash demo.sh
```
Usage instructions of all training parameters can be found in `config.py`.
### Evaluation
To evaluate the performance of a trained model, run
```
python eval.py
```
which will output Rank-1, Rank-5, Rank-10 and mAP scores.

## Citation
=======
