import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class SequenceDataset(Dataset):
    def __init__(self, folder_paths, root_path, is_trainset = False):
        self.seq_samples = []
        self.label_samples = []
        self.length_samples = []
        if is_trainset == True:
            label_file1 = os.path.join(root_path, 'train_videos.npy')
            anno1 = np.load(label_file1,allow_pickle=True)
            for i in range(anno1.shape[0]):
                vid_name = str(int(anno1[i][0]))
                label = np.array([int(anno1[i][4]), int(anno1[i][4])+int(anno1[i][5])])
                seq_file = os.path.join(folder_paths, vid_name, 'score.npy')
                '''
                test_file = os.path.join(folder_paths, vid_name, 'kpt.npy')
                seqs = np.load(test_file,allow_pickle=True)
                seq = seqs[0].cpu().numpy()
                for ele in range(1, len(seqs)):
                    seq = np.concatenate((seq, seqs[ele].cpu().numpy()))
                print(seq.shape)
                test_file = os.path.join(folder_paths, vid_name, 'feat.npy')
                seqs = np.load(test_file,allow_pickle=True)
                seq = seqs[0].cpu().numpy()
                for ele in range(1, len(seqs)):
                    seq = np.concatenate((seq, seqs[ele].cpu().numpy()))
                print(seq.shape)
                '''

                seqs = np.load(seq_file,allow_pickle=True)
                seq = seqs[0]
                for ele in range(1, len(seqs)):
                    seq = np.concatenate((seq, seqs[ele]))

                self.seq_samples.append(torch.from_numpy(np.array(seq)))
                self.label_samples.append(label)
                self.length_samples.append(seq.shape[0])

            label_file2 = os.path.join(root_path, 'rest_videos.npy')
            anno2 = np.load(label_file2,allow_pickle=True)
            for i in range(anno2.shape[0]):
                vid_name = str(int(anno2[i][0]))
                label = np.array([int(anno2[i][4]), int(anno2[i][4])+int(anno2[i][5])])
                seq_file = os.path.join(folder_paths, vid_name, 'score.npy')
                seqs = np.load(seq_file,allow_pickle=True)
                seq = seqs[0]
                for ele in range(1, len(seqs)):
                    seq = np.concatenate((seq, seqs[ele]))

                self.seq_samples.append(torch.from_numpy(np.array(seq)))
                self.label_samples.append(label)
                self.length_samples.append(seq.shape[0])

        else: 
            label_file = os.path.join(root_path, 'test_videos.npy')
            anno = np.load(label_file,allow_pickle=True)
            for i in range(anno.shape[0]):
                vid_name = str(int(anno[i][0]))
                label = np.array([int(anno[i][4]), int(anno[i][4])+int(anno[i][5])])
                seq_file = os.path.join(folder_paths, vid_name, 'score.npy')
                seqs = np.load(seq_file,allow_pickle=True)
                seq = seqs[0]
                for ele in range(1, len(seqs)):
                    seq = np.concatenate((seq, seqs[ele]))

                self.seq_samples.append(torch.from_numpy(np.array(seq)))
                self.label_samples.append(label)
                self.length_samples.append(seq.shape[0])

    def __len__(self):
        return len(self.label_samples) 

    def __getitem__(self, idx):
        # Load sequence data and label pair
        seq = self.seq_samples[idx]
        label = self.label_samples[idx]
        length = self.length_samples[idx]
        return seq, label, length


def collate_fn(batch):
    # Pad signals to max length in batch
    seqs, labels, seq_lengths = zip(*batch)
    lengths = [seq.shape[0] for seq in seqs]
    max_length = max(lengths)
    padded_seqs = []
    for seq in seqs:
        if seq.shape[0] < max_length:
            padded_seq = torch.cat([seq, torch.zeros(max_length - seq.shape[0], seq.shape[1])], dim=0)
        else:
            padded_seq = seq
        padded_seqs.append(padded_seq)
    padded_seqs = torch.stack(padded_seqs)
    return padded_seqs, labels, seq_lengths