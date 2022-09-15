# for i in {1..10}
#    do
#    # process the data to the fold
#    echo "taskset 100 python3 urbansound_loader.py --raw_data_path /media/data/public-data/SoundEvent/UrbanSound8K/ --setup centralized --fl_feature False --test_fold" $i
#    taskset 100 python3 urbansound_loader.py --raw_data_path /media/data/public-data/SoundEvent/UrbanSound8K/ --setup centralized --fl_feature False --test_fold $i
# done

# taskset 100 python3 gcommand_loader.py --raw_data_path ../../data/speech_commands --setup centralized --fl_feature False
# for i in {1..5}
#    do
    # process the data to the fold
#    echo "taskset 100 python3 crema_d_loader.py --raw_data_path /media/data/public-data/SER/crema-d/ --setup centralized --fl_feature False --test_fold "$i
#    taskset 100 python3 crema_d_loader.py --raw_data_path ../../data/crema_d/CREMA-D --setup centralized --fl_feature False --test_fold $i
# done

for i in {1..5}
    do
    # process the data to the fold
    echo "taskset 100 python3 iemocap_loader.py --raw_data_path /media/data/sail-data/iemocap/ --setup centralized --fl_feature False --test_session Session"$i
    taskset 100 python3 iemocap_loader.py --raw_data_path /media/data/sail-data/iemocap/ --setup centralized --fl_feature False --test_session Session$i
done

