import numpy as np
import json
import os

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

if os.path.exists('./data') == False:
    os.mkdir("./data/")

if os.path.exists('./data/audio_16k') == False:
    os.mkdir('./data/audio_16k/')
    audio_path = "../../../data/NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train"
    audio_files = get_immediate_files(audio_path)

    base_dir ="./data"
    for audio in audio_files:
        print('sox ' + audio_path +"/"+ audio + ' -r 16000 ' + base_dir + '/audio_16k/' + audio)
        os.system('sox ' + audio_path +"/"+ audio + ' -r 16000 ' + base_dir + '/audio_16k/' + audio)

if os.path.exists('./data/class_labels_indices.csv') == False:
    original_label = np.loadtxt('../../../data/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/nips4b_birdchallenge_espece_list.csv', delimiter=',', dtype='str')
    # class number,class name,English_name,Scientific_name,type

    new_header = 'index,mid,display_name'
    with open ('./data/class_labels_indices.csv', 'w') as f:
        f.write(new_header + '\n')
        for i in range(1, len(original_label)):
            new_class_num = str(i-1)
            new_class_name = original_label[i][1]
            new_mid = '/m/nips' + str(i).zfill(2)
            f.write(new_class_num + ',' + new_mid + ',' + new_class_name + '\n')

if os.path.exists('./data/datafiles') == False:
    os.mkdir('./data/datafiles')

    label_set = np.loadtxt('./data/class_labels_indices.csv', delimiter=',', dtype='str')
    label_map = []
    for i in range(1, len(label_set)):
        label_map.append(label_set[i][1])
    print("Categories: ",label_map)


    # load my split of val tra tes
    with open ("../../../split/test.txt", "r") as f:
        tes_split = [line.strip().replace("txt", "wav").replace("cepst_conc_cepst_nips4b_birds_", "nips4b_birds_") for line in f.readlines()]
    with open ("../../../split/val.txt", "r") as f:
        val_split = [line.strip().replace("txt", "wav").replace("cepst_conc_cepst_nips4b_birds_", "nips4b_birds_") for line in f.readlines()]   
    with open ("../../../split/train.txt", "r") as f:
        tra_split = [line.strip().replace("txt", "wav").replace("cepst_conc_cepst_nips4b_birds_", "nips4b_birds_") for line in f.readlines()]

    train_wav_list = []
    eval_wav_list = []
    test_wav_list = []
    dir = "./data/audio_16k/"

    train_set_labels = np.loadtxt('../../../data/NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/nips4b_birdchallenge_train_labels.csv', delimiter=',', dtype='str')
    print("Number of datas: ",len(train_set_labels))
    for i in range(0, len(train_set_labels)):
        # if not end with .wav, pass
        if train_set_labels[i][0].endswith('.wav') == False:
            continue

        cur_wav_file = train_set_labels[i][0]
        if train_set_labels[i][1] == '0':
            cur_labels = label_map[0]
        else:
            counter = []
            for j in range(4, len(train_set_labels[i])):
                #print(train_set_labels[i][j])
                if train_set_labels[i][j] == "1":
                    counter.append(label_map[j-3])
            cur_labels = ",".join(counter)
        print(cur_labels)

        if train_set_labels[i][0] in tes_split:
            cur_dict = {"wav": dir+cur_wav_file, "labels": cur_labels}
            test_wav_list.append(cur_dict)

        elif train_set_labels[i][0] in val_split:
            cur_dict = {"wav": dir+cur_wav_file, "labels": cur_labels}
            eval_wav_list.append(cur_dict)
        
        elif train_set_labels[i][0] in tra_split:
            cur_dict = {"wav": dir+cur_wav_file, "labels": cur_labels}
            train_wav_list.append(cur_dict)

    with open('./data/datafiles/train.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)
    with open('./data/datafiles/eval.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)
    with open('./data/datafiles/test.json', 'w') as f:
        json.dump({'data': test_wav_list}, f, indent=1)

print("Done!")

