<div align="center">

# COVID-19 Recognition

[![python](https://img.shields.io/badge/-Python-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

<br>

## Description
The COVID-19 pandemic has highlighted the importance of early detection of the disease. In many cases, chest X-rays can play a crucial role in identifying the presence of COVID-19, as well as differentiating between COVID-19, normal and pneumonia cases. This project aims to develop a deep learning model for recognizing COVID-19 positive cases from X-Ray chest images using the ConvNeXt architecture. The purpose of this project is to provide a tool for early detection of COVID-19 using chest X-rays. The model will be trained on a dataset of X-Ray images and will be evaluated based on its accuracy and other performance metrics. This tool can be used by healthcare professionals to make informed decisions in the management of COVID-19 patients. This project also aims to test the performance of ConvNeXt architecture on chest X-Ray image datasets.

<br>

## Dataset
The dataset used for training and testing the model consists of X-Ray chest images of COVID-19 positive, normal, and pneumonia cases. The dataset was obtained from the following research papers:
- Covid Image Data Collection by Joseph Paul Cohen et al. [arXiv](https://arxiv.org/pdf/2003.11597.pdf)
- COVID-19 Image Data Collection: Prospective Predictions are the Future by Joseph Paul Cohen et al. [arXiv](https://arxiv.org/pdf/2006.11988v3.pdf)

They also provide a GitHub link that can be found [here](https://github.com/ieee8023/covid-chestxray-dataset). Dataset that is already preprocessed also can be downloaded from [kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset).

<br>

## Architecture
![](https://github.com/adhiiisetiawan/covid19-recognition/blob/main/convnext.png)

This project using ConvNeXt architecture, ConvNeXt is a modification of the ResNet architecture for deep ConvNets. It introduces the concept of "grouped convolutions" which allow for a more efficient utilization of computation resources such as memory and computation power. The main features of ConvNeXt include:

1. **Grouped Convolutions:** The main difference from ResNet is that ConvNeXt uses grouped convolutions instead of standard convolutions in the residual blocks. Grouped convolutions split the input channels into groups, and apply a separate set of filters to each group. This allows for more computation to be done with a smaller number of parameters, which helps reduce overfitting.

2. **Bottleneck Design:** ConvNeXt uses a bottleneck design, where the number of filters in the 3x3 convolution is reduced, and then increased again using a 1x1 convolution. This design helps reduce the number of parameters and computational costs.

3. **Stem and Head:** ConvNeXt has a stem and head component, which includes several convolutional layers to extract features from the input image and several fully connected layers to make the final prediction.

4. **Dense Connections:** ConvNeXt uses dense connections, which means that every residual block is connected to every other block in the network. This helps reduce the vanishing gradient problem that can occur in deep ConvNets.

Overall, the ConvNeXt architecture aims to balance efficiency and performance, and has been shown to achieve state-of-the-art results on various computer vision tasks such as image classification and object detection.

<br>

## Main Technologies

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

[Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

<br>

## Project Structure

The directory structure of new project looks like this:

```
├── configs                   <- Hydra configuration files
│   ├── callbacks                <- Callbacks configs
│   ├── data                     <- Data configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── extras                   <- Extra utilities configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── hydra                    <- Hydra configs
│   ├── local                    <- Local configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── paths                    <- Project paths configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── eval.yaml             <- Main config for evaluation
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by hydra and lightning loggers
│
├── notebooks              <- Jupyter notebooks for the project
│
├── src                    <- Source code
│   ├── data                     <- Lightning datamodules
│   ├── models                   <- Lightning models
│   └── utils                    <- Utility scripts
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── requirements.txt          <- File for installing python dependencies
├── README.md
├── eval.py                   <- Run evaluation
└── train.py                  <- Run training
```

<br>

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/adhiiisetiawan/covid19-recognition
cd covid19-recognition

# [OPTIONAL] create python environment
python3 -m venv [env-name]

# activate environment
source [env-name]/bin/activate

# install requirements
pip install -r requirements.txt
```

**Note:** Before run training, you must download the dataset first from [kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset) and put in the `data` folder. You also need to extract the dataset. For another approach, you need to register a kaggle API to download the folder with code. But, I recomended to download manually since using kaggle API need some effort to configure. 

Train model with default configuration, default configuration using ConvNeXt Base

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with specific architecture

```bash
# train on CPU
python src/train.py trainer=cpu model=convnext_small

# train on GPU
python src/train.py trainer=gpu model=convnext_tiny
```

Train model using wandb logger

```bash
# train on CPU
python src/train.py trainer=cpu model=convnext_small logger=wandb

# train on GPU
python src/train.py trainer=gpu model=convnext_tiny logger=wandb
```

<br>

## Performance and Results
Here's the performance of the model using Covid-19 X-Ray Dataset. I just provide three ConvNeXt models (tiny, small, base) because of limited resources. Details about graphic accuracy and loss can be found on [wandb report](https://wandb.ai/adhiisetiawan/covid19-recognition).<br>
![](https://github.com/adhiiisetiawan/covid19-recognition/blob/main/wandb.png)

| name | acc | #params | model |
|:---:|:---:|:---:|:---:|
| ConvNeXt-T | 96.9 | 28M | [model](https://drive.google.com/file/d/1Z7Q-dv-iIkkjt2tM-wravQQ7E0TFY0hw/view?usp=share_link) |
| ConvNeXt-S | 95.4 | 50M | [model](https://drive.google.com/file/d/1-0_-XMVs0NSvdEGk6Iv8EJdyRyU0ZsGL/view?usp=share_link) |
| ConvNeXt-B | 93.8 | 89M | [model](https://drive.google.com/file/d/1-010Cz8Cl6HpiQfB9MNn4oUtJpfdXv_X/view?usp=share_link) |

<br>

## Acknowledgement
- This project is built using the [Lightning Hydra](https://github.com/ashleve/lightning-hydra-template) template.
- Thanks to Joseph Paul Cohen et al. for providing the dataset. [Paper 1](https://arxiv.org/pdf/2003.11597.pdf) | [Paper 2](https://arxiv.org/pdf/2006.11988v3.pdf) | [GitHub](https://github.com/ieee8023/covid-chestxray-dataset)
- Thanks to `pranavraikokte` for providing the preprocessed dataset in [Kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

<br>

## Reference
```
@article{cohen2020covid,
  title={COVID-19 image data collection},
  author={Joseph Paul Cohen and Paul Morrison and Lan Dao},
  journal={arXiv 2003.11597},
  url={https://github.com/ieee8023/covid-chestxray-dataset},
  year={2020}
}
```

```
@article{cohen2020covidProspective,
  title={COVID-19 Image Data Collection: Prospective Predictions Are the Future},
  author={Joseph Paul Cohen and Paul Morrison and Lan Dao and Karsten Roth and Tim Q Duong and Marzyeh Ghassemi},
  journal={arXiv 2006.11988},
  url={https://github.com/ieee8023/covid-chestxray-dataset},
  year={2020}
}
```

```
@Article{liu2022convnet,
  author  = {Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  title   = {A ConvNet for the 2020s},
  journal = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year    = {2022},
}
```
