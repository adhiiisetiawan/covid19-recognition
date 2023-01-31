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
The current COVID-19 pandemic has highlighted the importance of early detection of the disease. In many cases, chest X-rays can play a crucial role in identifying the presence of COVID-19, as well as differentiating between COVID-19, normal and pneumonia cases. This project aims to develop a deep learning model for recognizing COVID-19 positive cases from X-Ray chest images using the ConvNeXt architecture. The purpose of this project is to provide a tool for early detection of COVID-19 using chest X-rays. The model will be trained on a dataset of X-Ray images and will be evaluated based on its accuracy and other performance metrics. This tool can be used by healthcare professionals to make informed decisions in the management of COVID-19 patients. This project also aims to test the performance of ConvNeXt architecture on chest X-Ray image datasets.

## Dataset
The dataset used for training and testing the model consists of X-Ray chest images of COVID-19 positive, normal, and pneumonia cases. The dataset was obtained from the following research papers:
- Covid Image Data Collection by Joseph Paul Cohen et al. [arXiv](https://arxiv.org/pdf/2003.11597.pdf)
- COVID-19 Image Data Collection: Prospective Predictions are the Future by Joseph Paul Cohen et al. [arXiv](https://arxiv.org/pdf/2006.11988v3.pdf)

They are also provide GitHub link that can be found [here](https://github.com/ieee8023/covid-chestxray-dataset). Dataset that already preprocessed also can be download from [kaggle](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset). 

## Architecture
![](https://github.com/adhiiisetiawan/covid19-recognition/blob/main/convnext.png)

This project using ConvNeXt architecture, ConvNeXt is a modification of the ResNet architecture for deep ConvNets. It introduces the concept of "grouped convolutions" which allow for a more efficient utilization of computation resources such as memory and computation power. The main features of ConvNeXt include:

1. Grouped Convolutions: The main difference from ResNet is that ConvNeXt uses grouped convolutions instead of standard convolutions in the residual blocks. Grouped convolutions split the input channels into groups, and apply a separate set of filters to each group. This allows for more computation to be done with a smaller number of parameters, which helps reduce overfitting.

2. Bottleneck Design: ConvNeXt uses a bottleneck design, where the number of filters in the 3x3 convolution is reduced, and then increased again using a 1x1 convolution. This design helps reduce the number of parameters and computational costs.

3. Stem and Head: ConvNeXt has a stem and head component, which includes several convolutional layers to extract features from the input image and several fully connected layers to make the final prediction.

4. Dense Connections: ConvNeXt uses dense connections, which means that every residual block is connected to every other block in the network. This helps reduce the vanishing gradient problem that can occur in deep ConvNets.

Overall, the ConvNeXt architecture aims to balance efficiency and performance, and has been shown to achieve state-of-the-art results on various computer vision tasks such as image classification and object detection.

## Main Technologies

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - a lightweight PyTorch wrapper for high-performance AI research. Think of it as a framework for organizing your PyTorch code.

[Hydra](https://github.com/facebookresearch/hydra) - a framework for elegantly configuring complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line.

<br>

## Main Ideas

- [**Predefined Structure**](#project-structure): clean and scalable so that work can easily be extended
- [**Rapid Experimentation**](#your-superpowers): thanks to hydra command line superpowers
- [**Little Boilerplate**](#how-it-works): thanks to automating pipelines with config instantiation
- [**Main Configs**](#main-config): allow to specify default training configuration
- [**Experiment Configs**](#experiment-config): allow to override chosen hyperparameters
- [**Workflow**](#workflow): comes down to 4 simple steps
- [**Experiment Tracking**](#experiment-tracking): Tensorboard, W&B, Neptune, Comet, MLFlow and CSVLogger
- [**Logs**](#logs): all logs (checkpoints, configs, etc.) are stored in a dynamically generated folder structure
- [**Hyperparameter Search**](#hyperparameter-search): made easier with Hydra plugins like Optuna Sweeper
- [**Tests**](#tests): generic, easy-to-adapt tests for speeding up the development
- [**Continuous Integration**](#continuous-integration): automatically test your repo with Github Actions
- [**Best Practices**](#best-practices): a couple of recommended tools, practices and standards

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
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── data                     <- Lightning datamodules
│   ├── models                   <- Lightning models
│   └── utils                    <- Utility scripts
│
├── .env.example              <- Example of file for storing private environment variables
├── .gitignore                <- List of files ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── pyproject.toml            <- Configuration options for testing and linting
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
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
