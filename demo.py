import pickle
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from Functions import *
import matplotlib.pyplot as plt
from Network import *
from Functions import *
from tqdm import tqdm
from sklearn.model_selection import KFold
from ranger import Ranger
import argparse
from sklearn.metrics import mean_squared_error
from accelerate import Accelerator
import time
import json
from RNAdegformer_functions import *
from accelerate import Accelerator

tokens = 'ACGU().BEHIMSX'
# eterna,'nupack','rnastructure','vienna_2','contrafold_2',


class RNADataset(Dataset):
    def __init__(self, seqs, labels, ids, ew, bpp_path, transform=None, training=True, pad=False, k=5):
        self.transform = transform
        self.seqs = seqs  # .transpose(1,0,2,3)
        # print(self.data.shape)
        self.data = []
        self.labels = labels.astype('float32')
        self.bpp_path = bpp_path
        self.ids = ids
        self.training = training
        self.bpps = []
        # .reshape(1,bpps.shape[-1],bpps.shape[-1])
        dm = get_distance_mask(len(seqs[-1]))
        self.dms = np.asarray([dm for i in range(12)])
        # print(dm.shape)
        # exit()
        self.lengths = []
        for i, id in tqdm(enumerate(self.ids)):
            bpps = np.load(os.path.join(
                self.bpp_path, 'train_test_bpps', id+'_bpp.npy'))
            if pad:
                bpps = np.pad(
                    bpps, ([0, 0], [0, 130-bpps.shape[1]], [0, 130-bpps.shape[2]]), constant_values=0)
            # print(bpps.shape)
            # exit()
            # dms=np.asarray([dm for i in range(bpps.shape[0])])
            # bpps=np.concatenate([bpps.reshape(bpps.shape[0],1,bpps.shape[1],bpps.shape[2]),dms],1)

            with open(os.path.join(self.bpp_path, 'train_test_bpps', id+'_struc.p'), 'rb') as f:
                structures = pickle.load(f)
            with open(os.path.join(self.bpp_path, 'train_test_bpps', id+'_loop.p'), 'rb') as f:
                loops = pickle.load(f)
            seq = self.seqs[i]
            self.lengths.append(len(seq))
            # print(seq)
            # exit()
            input = []

            for j in range(bpps.shape[0]):
                input_seq = np.asarray([tokens.index(s) for s in seq])
                # input_seq=np.pad(input_seq,[0,130-bpps.shape[1]])
                input_structure = np.asarray(
                    [tokens.index(s) for s in structures[j]])
                input_loop = np.asarray([tokens.index(s) for s in loops[j]])
                input.append(
                    np.stack([input_seq, input_structure, input_loop], -1))
            input = np.asarray(input).astype('int')
            # print(input.shape)
            if pad:
                input = np.pad(
                    input, ([0, 0], [0, 130-input.shape[1]], [0, 0]), constant_values=14)
            # print(input.shape)
            # exit()
            # print(input.shape)
            self.data.append(input)
            # exit()
            # print(np.stack([input_seq,input_structure,input_loop],-1).shape)
            # exit()
            # plt.subplot(1,4,1)
            # for _ in range(4):
            #     plt.subplot(1,4,_+1)
            #     plt.imshow(bpps[0,_])
            # plt.show()
            # exit()
            self.bpps.append(np.clip(bpps, 0, 1).astype('float32'))
            # if i >200:
            #     break
        self.data = np.asarray(self.data)
        self.lengths = np.asarray(self.lengths)
        # print(self.lengths)
        # exit()
        self.ew = ew
        self.k = k
        # if pad:
        self.src_masks = [self.generate_src_mask(
            self.lengths[i], self.data.shape[-2], self.k) for i in range(len(self.data))]
        self.pad = pad

    def generate_src_mask(self, L1, L2, k):
        mask = np.ones((k, L2), dtype='int8')
        for i in range(k):
            mask[i, L1+i+1-k:] = -100
        return mask

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample = {'data': self.data[idx], 'labels': self.labels[idx]}
        if self.training:
            bpp_selection = np.random.randint(self.bpps[idx].shape[0])
            bpps = self.bpps[idx][bpp_selection]
            bpps = np.concatenate(
                [bpps.reshape(1, bpps.shape[0], bpps.shape[1]), self.dms[0]], 0)
            bpps = bpps.astype('float32')
            # print(self.bpps[idx].shape[0])
            # src_mask=
            # print(src_mask.shape)
            # exit()
            # if self.pad:
            sample = {'data': self.data[idx][bpp_selection], 'labels': self.labels[idx], 'bpp': bpps,
                      'ew': self.ew[idx], 'id': self.ids[idx], 'src_mask': self.src_masks[idx]}
            # else:
            #     sample = {'data': self.data[idx][bpp_selection], 'labels': self.labels[idx], 'bpp': self.bpps[idx][bpp_selection],
            #     'ew': self.ew[idx],'id':self.ids[idx]}
        else:
            bpps = self.bpps[idx]
            bpps = np.concatenate(
                [bpps.reshape(bpps.shape[0], 1, bpps.shape[1], bpps.shape[2]), self.dms], 1)
            bpps = bpps.astype('float32')
            sample = {'data': self.data[idx], 'labels': self.labels[idx], 'bpp': bpps,
                      'ew': self.ew[idx], 'id': self.ids[idx]}
        # if self.transform:
        #     sample=self.transform(sample)
        return sample


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/MPR.yaml")
    args = parser.parse_args()
    config = load_config_from_yaml(args.config_path)

    json_path = os.path.join(config.path, 'train.json')
    json = pd.read_json(json_path, lines=True)
    json = json[json.signal_to_noise > config.noise_filter]
    ids = np.asarray(json.id.to_list())

    error_weights = get_errors(json)
    error_weights = config.error_alpha+np.exp(-error_weights*config.error_beta)
    train_indices, val_indices = get_train_val_indices(json, config.fold, SEED=2020, nfolds=config.nfolds)

    _, labels = get_data(json)
    sequences = np.asarray(json.sequence)
    train_seqs = sequences[train_indices]
    val_seqs = sequences[val_indices]
    train_labels = labels[train_indices]
    val_labels = labels[val_indices]
    train_ids = ids[train_indices]
    val_ids = ids[val_indices]
    train_ew = error_weights[train_indices]
    val_ew = error_weights[val_indices]

    dataset = RNADataset(train_seqs, train_labels, train_ids, train_ew, config.path)
    val_dataset = RNADataset(val_seqs, val_labels, val_ids, val_ew, config.path)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RibonanzaNet(config).to(device)
    model.load_state_dict(torch.load(f"models/model0.pt", map_location='cpu'))
    model = nn.DataParallel(model)

    optimizer = Ranger(model.parameters(),weight_decay=config.weight_decay, lr=config.learning_rate)
    criterion=weighted_MCRMSE

    for epoch in range(config.epochs):
        model.train(True)
        total_loss = 0
        optimizer.zero_grad()
        step = 0
        for data in train_loader:
            step+=1
            src=data['data'].to(device)
            src = src[:, :, 0]
            bpps=data['bpp'].to(device)
            labels=data['labels'].to(device)
            ew=data['ew'].to(device)
            output=model(src,bpps)
            loss=criterion(output[:,:68],labels,ew).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            total_loss+=loss

        train_loss=total_loss/(step+1)
        torch.cuda.empty_cache()

        if (epoch+1)%config.val_freq==0:
            val_loss=validate(model,device,val_dataloader,batch_size=config.batch_size)
            print(f'Validation loss: {val_loss}') 

        if (epoch+1)%config.save_freq==0:
            save_weights(model,optimizer,epoch,checkpoints_folder)

    print("finished!!")

if __name__ == "__main__":
    train()