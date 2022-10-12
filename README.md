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

