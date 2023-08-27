# SAMPath-plus: A Segment Anything Model for Semantic Segmentation in Digital Pathology
A Pytorch implementation for the SAM-Path framework. More details about the research can be found in the paper [SAM-Path: A Segment Anything Model for Semantic Segmentation in Digital Pathology](https://link.springer.com/chapter/10.1007/978-3-031-47401-9_16) and [arxiv](https://arxiv.org/abs/2307.09570).  
<div>
  <img src="imgs/overview.png" width="100%"  alt="The overview of our framework."/>
</div>

## Installation
Follow the instructions for installing [Anaconda/miniconda](https://www.anaconda.com/products/distribution).  
Additionally, install the dependencies of [SAM](https://github.com/facebookresearch/segment-anything). Please refrain from installing the original SAM as we have made some modifications.

Then Install the required packages:
```
  $ pip install monai torchmetrics==0.11.4 pytorch_lightning==2.0.2 albumentations box wandb
```
## Data organization
Detailed structure of our dataset organization is provided. Download our preprocessed dataset from the following: link [https://drive.google.com/drive/folders/1BUPZz3nB52J5zRs1ZcEvNK03zw18BeLN?usp=sharing](https://drive.google.com/drive/folders/1BUPZz3nB52J5zRs1ZcEvNK03zw18BeLN?usp=sharing).

## Training
Use ```train.py``` to train and evaluate our framework. 

For example:
```
python main.py --config configs.BCSS --devices 0 --project sampath --name bcss_run0
python main.py --config configs.CRAG --devices 1 --project sampath --name crag_run0
```
The pretrained models can be downloaded from [SAM](https://github.com/facebookresearch/segment-anything#model-checkpoints) and [HIPT](https://github.com/mahmoodlab/HIPT#pre-reqs--installation).

## Contact
For any questions or concerns, feel free to report issues or send a direct message to the new repository owner.

## Citation
If this project contributes to your research, please cite using the following BibTeX entry, adjusting the authors to fit your reference style.  
```
@article{zhang2023sam,
  title={SAM-Path: A Segment Anything Model for Semantic Segmentation in Digital Pathology},
  author={Zhang, Jingwei and Ma, Ke and Kapse, Saarthak and Saltz, Joel and Vakalopoulou, Maria and Prasanna, Prateek and Samaras, Dimitris},
  journal={arXiv preprint arXiv:2307.09570},
  year={2023}
}
```