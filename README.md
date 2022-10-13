# FedAudio: A Guideline Benchmark for Audio Tasks with FL

## Table of Contents
* Overview
* Installation
* Supported Dataset
* Usage
* Contact

## Overview
Federated Learning has gained considerable interest in enabling multiple clients holding sensitive data to collaboratively train machine learning models without centralizing data. However, while many works have addressed the application of audio tasks with FL, few realistic FL datasets exist as benchmarks for algorithmic research. 
Therefore, we present FedAudio, a benchmark guideline suite for evaluating federated learning methods on audio tasks. 


The FLamby package contains:
* Data loaders with various types of data preprocessing methods and automatically partitioning.
* FL feature managers to inject noise on top of the raw signal to simulate the realstic FL challenges.
* Baseline evaluation results and code with popular aggregation functions.

<div align="center">
 <img src="FedAudio.png" width="600px">
</div>

## Installation
In our repo, we include an unoffical offline version of FedML as an example to implement the FedAudio. Users could directly clone our repo and run the code without installing any other federated learning frameworks.

To begin with, please clone this repo and install the conda environment:
```
git clone https://github.com/zhang-tuo-pdf/FedSpeech22.git
cd FedSpeech22
conda env create -f fedspeech.yml
conda activate fedml
```
To make sure the fully functioning for the data downloading and processing, please install the below packages as well:
```
apt install sox
brew install git-lfs
```

## Supported Dataset
Currently, FedAudio encompasses four audio datasets (Google Speech Command, IEMOCAP, CREMA-D, Urban Sound) with client partition, covering three common tasks, and accompanied by baseline training code and results. We would continue enriching the package with datasets and baseline from other applications to support more research scenarios. Please leave an message to us if you want to contribute or have some suggestions for the package development.

