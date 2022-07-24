# Dataloading framework

raw data -> data split -> adding fl feature -> preprocess -> torch dataloader

default data split for all dataset: 

train data: [key, wav_command, word_id]

for example: ['004ae714_nohash_1_zero', '../data/speech_commands/audio/zero/004ae714_nohash_1.wav',  34]

test data: [key, wav_command, word_id]

for example: ['ffb86d3c_nohash_2_stop', '../data/speech_commands/audio/stop/ffb86d3c_nohash_2.wav', 26]

# FedSpeech22
step 1: load the dataset
```
cd data_loading/data_loader
python gcommand_loader.py
```
step 2: run the fl training
```
cd experiment
sh run_distributed.sh gcommand 2118 10 8 30 1 16 0.1 0 8
# dataset name, total client number, sampled client number, gpu number, round number, local epoch number, batch size, lr, start gpu, total gpu
```

# Dataset
Google speech command: 2118 clients
https://arxiv.org/abs/1804.03209

