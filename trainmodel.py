import csv
import torch
import os
import torchaudio
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

sample_rate = 44100
model_name = 'audioCNNModel500'
path = './pathologicalVoice/TrainingDataset'
batch_size = 256
num_epochs = 400
# train_data = pd.read_csv(path + '/TrainingDatalist.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AudioDataset(Dataset):
    def __init__(self, csv_file, sample_rate=44100):
        self.data = []
        self.sample_rate = sample_rate
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                audio_file = os.path.join(path + '/training_voice_data/' + row['ID'] + '.wav')
                label = int(row['Disease category']) - 1  # Convert label to 0-based index
                self.data.append((audio_file, label))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        sample_rate = self.sample_rate
        n_fft = int(16/1000 * sample_rate) #
        hop_length = int(8/1000 * sample_rate)
        audio_file, label = self.data[idx]
        # Preprocess waveform (e.g. resample, normalize, augment)
        waveform, sample_rate = torchaudio.load(audio_file)
        # if waveform.shape[0] > 1:
            # waveform = torch.mean(waveform, dim=0, keepdim=True)
        # waveform = torchaudio.functional.compute_deltas(waveform)
        waveform = self.ProcessAudio(waveform).to(device)
        # waveform = self.transformation(waveform)
        mel_specgram = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=33).to(device)
        waveform = mel_specgram(waveform)
        return waveform.reshape(3 ,-1, 126), label # waveform重塑為(3, 10, 126)的tensor，label為int
    
    def ProcessAudio(self, waveform):
        waveform_length = self.sample_rate
        if waveform.shape[1] < waveform_length:
            waveform = torch.nn.functional.pad(waveform, (0, waveform_length - waveform.shape[0]), "constant")
        elif waveform.shape[1] > waveform_length:
            waveform = waveform[ :, :waveform_length]
        return waveform
aD = AudioDataset(path + '/TrainingDatalist.csv')
# print(len(aD))
# for i in range(len(aD)):
#     print(f"{aD[i][0].shape}, {aD[i][1]}")
# class CNNModel(nn.Module):
#     def __init__(self, num_classes=5):
#         super(CNNModel, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=2),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2)
#         )
#         self.flatten = nn.Flatten()
#         self.linear = nn.Linear( 8*5*9, 128)
#         self.linear2 = nn.Linear( 128, num_classes)
#         self.softmax = nn.Softmax(dim=1)
        
#     def forward(self, input_data):
#         x = self.conv1(input_data)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.flatten(x)
#         logits = self.linear(x)
#         logits = self.linear2(logits)
#         predictions = self.softmax(logits)
#         return predictions
def AudioDataloader(csv_file, batch_size, shuffle=True):
    dataset = AudioDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
def train_model(model, dataloaders, loss_fn, optimizer, num_epochs):
    for epoch in range (num_epochs):
        total_correct = 0
        for x_batch, y_batch in dataloaders:
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(y_pred, dim=1)
            total_correct += torch.sum(preds == y_batch).item()
        print(f"Epoch: {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {total_correct / len(aD):.4f}")
# model = CNNModel().to(device)
model = models.efficientnet_b0(weights=None, num_classes=5)
model.load_state_dict(torch.load('audioCNNModel.pth')) #transfer learning
data_loader = AudioDataloader(path + '/TrainingDatalist.csv', batch_size)
loss_fn = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(model, data_loader, loss_fn, optimiser, num_epochs)
torch.save(model.state_dict(), f"{model_name}.pth")