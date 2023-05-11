import pickle
import torch
import numpy as np
import dataloader
from torch.utils.data import WeightedRandomSampler
import models
from utilities import *

def evaluate(path):
    model_path = path + '/models/best_audio_model.pth'
    optimizer_path = path + '/models/best_audio_optimizer.pth'
    with open (path+"/args.pkl", "rb") as f:
        args = pickle.load(f)

    print('now train a audio spectrogram transformer model')

    audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem, 'mixup': args.mixup, 'dataset': args.dataset, 'mode':'train', 'mean':args.dataset_mean, 'std':args.dataset_std,
                  'noise':args.noise}
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': args.dataset, 'mode':'evaluation', 'mean':args.dataset_mean, 'std':args.dataset_std, 'noise':False}

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

    val_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_val, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    eval_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(args.data_eval, label_csv=args.label_csv, audio_conf=val_audio_conf),
        batch_size=args.batch_size*2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

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

    # get prediction results
    train_results = []
    for i, (audio_input, labels) in enumerate(train_loader):
        audio_input = audio_input.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            audio_output = audio_model(audio_input)
            audio_output = torch.sigmoid(audio_output)
        train_results.append(audio_output.cpu().numpy())
    train_results = np.concatenate(train_results, axis=0)
    print(train_results.shape)

if __name__ == '__main__':
    evaluate('exp/ast-attempt7-acc-mean-std')


    
    