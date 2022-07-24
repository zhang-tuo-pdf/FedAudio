# IEMOCAP Loaders
```
python3 iemocap_loader.py --raw_data_path iemocap_data_path \
                          --output_data_path output_data_path \
                          --process_method pretrain/raw/opensmile \
                          --feature_type apc/tera/emobase \
                          --test_session Session1/Session2/Session3/Session4/Session5

```
Output data format:
[key, wav_iemocap, emotion_id, data]
