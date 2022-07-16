# Dataloading framework

raw data -> data split -> adding fl feature -> preprocess -> torch dataloader

data split for gcommand: 

train data: [key, wav_command, [seg_ini, seg_end], word_id]

for example: ['004ae714_nohash_1_zero', '../data/speech_commands/audio/zero/004ae714_nohash_1.wav', ['0', '0.7895'], 34]

test data: [key, wav_command, word_id]

for example: ['ffb86d3c_nohash_2_stop', '../data/speech_commands/audio/stop/ffb86d3c_nohash_2.wav', 26]

# FedSpeech22
```
cd experiment
sh run_distributed.sh 2118 10 30 1 16 0.1 0 8
#total client number, sampled client number, round number, local epoch number, batch size, lr, start gpu, total gpu
```

# Dataset
Google speech command: 2118 clients
https://arxiv.org/abs/1804.03209

