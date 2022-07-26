#!/bin/bash
wget -N http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
#wget http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Features.Models.tar.gz
echo =========Begin Extracting MELD.RAW.tar.gz==================
tar -x --skip-old-files -f MELD.Raw.tar.gz
echo =========Done Extracting MELD.RAW.tar.gz===================

echo =========Begin Extracting MELD.Raw/train.tar.gz============
tar -x --skip-old-files -f MELD.Raw/train.tar.gz
echo =========Done Extracting MELD.Raw/train.tar.gz=============

echo =========Begin Extracting MELD.Raw/dev.tar.gz==============
tar -x --skip-old-files -f MELD.Raw/dev.tar.gz
echo =========Done Extracting MELD.Raw/dev.tar.gz================

echo =========Begin Extracting MELD.Raw/test.tar.gz==============
tar -x --skip-old-files -f MELD.Raw/test.tar.gz
echo =========Done Extracting MELD.Raw/test.tar.gz===============

cd train_splits
NUM_FILES=$(find ./ -name "*.mp4" | wc -l)
mkdir -p "waves"
for i in *.mp4; do
    ffmpeg -hide_banner -loglevel error -y -i "$i" "./waves/$(basename "$i" .mp4).wav"
  echo "$i"
done | pv -l -s "$NUM_FILES" >/dev/null

cd ../output_repeated_splits_test
NUM_FILES=$(find ./ -name "*.mp4" | wc -l)
mkdir -p "waves"
for i in *.mp4; do
    ffmpeg -hide_banner -loglevel error -y -i "$i" "./waves/$(basename "$i" .mp4).wav"
  echo "$i"
done | pv -l -s "$NUM_FILES" >/dev/null

cd ../dev_splits_complete
NUM_FILES=$(find ./ -name "*.mp4" | wc -l)
mkdir -p "waves"
for i in *.mp4; do
    ffmpeg -hide_banner -loglevel error -y -i "$i" "./waves/$(basename "$i" .mp4).wav"
  echo "$i"
done | pv -l -s "$NUM_FILES" >/dev/null