import pickle
import torch
import numpy as np
import dataloader
from torch.utils.data import WeightedRandomSampler
import models
from utilities import *
import sklearn

def evaluate(path):
    model_path = path + '/models/best_audio_model.pth'
    optimizer_path = path + '/models/best_audio_optimizer.pth'
    with open (path+"/args.pkl", "rb") as f:
        args = pickle.load(f)

    print('now train a audio spectrogram transformer model')

    audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
                  'noise':args.noise}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False}

    audio_model = models.ASTModel(label_dim=args.n_class, fstride=args.fstride, tstride=args.tstride, input_fdim=128,
                                  input_tdim=args.audio_length, imagenet_pretrain=args.imagenet_pretrain,
                                  audioset_pretrain=args.audioset_pretrain, model_size='base384')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(model_path, map_location=device)
    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd)

    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        audio_model = nn.DataParallel(audio_model)
    audio_model = audio_model.to(device)
    # switch to evaluate mode
    audio_model.eval()

    if args.data_train is not None:
        if args.bal == 'bal':
            print('balanced sampler is being used')
            samples_weight = np.loadtxt(args.data_train[:-5]+'_weight.csv', delimiter=',')
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

            train_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
                batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
        else:
            print('balanced sampler is not used')
            train_loader = torch.utils.data.DataLoader(
                dataloader.AudiosetDataset(args.data_train, label_csv=args.label_csv, audio_conf=audio_conf),
                batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # get prediction results
        train_results = []
        train_labels = []
        for i, (audio_input, labels) in enumerate(train_loader):
            audio_input = audio_input.to(device)
            train_labels.append(labels.cpu().numpy())
            labels = labels.to(device)
            with torch.no_grad():
                audio_output = audio_model(audio_input)
                audio_output = torch.sigmoid(audio_output)

                # print(audio_output.shape)
                # print(audio_output)

            audio_output = audio_output.cpu().numpy()
            sum = np.sum(audio_output, axis=1)
            # print(sum.shape)
            # mean
            audio_output = audio_output / sum[:, np.newaxis]
            # print(audio_output.shape)
            # print(audio_output)
            train_results.append(audio_output)
        train_results = np.concatenate(train_results, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        print(train_results.shape)

        train_auc = sklearn.metrics.roc_auc_score(train_labels.ravel(),train_results.ravel())
        print('train auc: ', train_auc)
    else:
        print('no training data is provided')

    if args.data_val is not None:
            
        val_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        
        # get prediction results
        val_results = []
        val_labels = []
        for i, (audio_input, labels) in enumerate(val_loader):
            audio_input = audio_input.to(device)
            val_labels.append(labels.cpu().numpy())
            labels = labels.to(device)
            with torch.no_grad():
                audio_output = audio_model(audio_input)
                audio_output = torch.sigmoid(audio_output)

                # print(audio_output.shape)
                # print(audio_output)

            audio_output = audio_output.cpu().numpy()
            sum = np.sum(audio_output, axis=1)
            # print(sum.shape)
            # mean
            audio_output = audio_output / sum[:, np.newaxis]
            # print(audio_output.shape)
            # print(audio_output)
            val_results.append(audio_output)
        val_results = np.concatenate(val_results, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        print(val_results.shape)

        val_auc = sklearn.metrics.roc_auc_score(val_labels.ravel(),val_results.ravel())
        print('val auc: ', val_auc)
    else:
        print('no validation data is provided')

    if args.data_eval is not None:

        eval_loader = torch.utils.data.DataLoader(
            dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
            batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        # get prediction results
        eval_results = []
        eval_labels = []
        for i, (audio_input, labels) in enumerate(eval_loader):
            audio_input = audio_input.to(device)
            eval_labels.append(labels.cpu().numpy())
            labels = labels.to(device)
            with torch.no_grad():
                audio_output = audio_model(audio_input)
                audio_output = torch.sigmoid(audio_output)

                # print(audio_output.shape)
                # print(audio_output)

            audio_output = audio_output.cpu().numpy()
            sum = np.sum(audio_output, axis=1)
            # print(sum.shape)
            # mean
            audio_output = audio_output / sum[:, np.newaxis]
            # print(audio_output.shape)
            # print(audio_output)
            eval_results.append(audio_output)
        eval_results = np.concatenate(eval_results, axis=0)
        eval_labels = np.concatenate(eval_labels, axis=0)
        print(eval_results.shape)

        eval_auc = sklearn.metrics.roc_auc_score(eval_labels.ravel(),eval_results.ravel())
        print('eval auc: ', eval_auc)
    else:
        print('no evaluation data is provided')

    

if __name__ == '__main__':
    evaluate('exp/ast-attempt4-weighted')


    
    