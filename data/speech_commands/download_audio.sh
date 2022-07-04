#!/bin/bash

audio_path=audio

mkdir -p $audio_path

n=`find $audio_path -name '*.wav' | wc -l`

pwd=`pwd`

if (( $n < 105835 )); then
  wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
  cd $audio_path
  tar -xvf $pwd/speech_commands_v0.02.tar.gz
  cd $pwd
fi