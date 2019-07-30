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
## Datasets
- Market-1501 dataset can be downloaded from [here](http://www.liangzheng.org/Project/project_reid.html).
- DukeMTMC-reID dataset can be downloaded from [here](http://vision.cs.duke.edu/DukeMTMC/).
--
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

### Citation
Please use the citation provided below if it is useful to your research:

Guangyi Chen, Tianren Zhang, Jiwen Lu, and Jie Zhou, Deep Meta Metric Learning, ICCV, 2019.
```bash
@inproceedings{chen2019deep,
  title={Deep Meta Metric Learning},
  author={Chen, Guangyi and Zhang, Tianren and Lu, Jiwen and Zhou, Jie},
  booktitle={ICCV},
  year={2019}
}
```
