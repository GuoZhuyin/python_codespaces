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
num_epochs = 100
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
        return waveform.reshape(3 ,-1, 126), label
    
    def ProcessAudio(self, waveform):
        waveform_length = self.sample_rate
        if waveform.shape[1] < waveform_length:
            waveform = torch.nn.functional.pad(waveform, (0, waveform_length - waveform.shape[0]), "constant")
        elif waveform.shape[1] > waveform_length:
            waveform = waveform[ :, :waveform_length]
        return waveform
class CNNModel(nn.Module):
    def __init__(self, num_classes=5):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear( 8*5*9, 128)
        self.linear2 = nn.Linear( 128, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        logits = self.linear(x)
        logits = self.linear2(logits)
        predictions = self.softmax(logits)
        return predictions
def AudioDataloader(csv_file, batch_size, shuffle=True):
    dataset = AudioDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        predictions = model(input)
        # Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        predicted_index = predictions[0].argmax(0)
        predicted = class_mapping[predicted_index]
        expected = class_mapping[target]
    return predicted, expected
class_mapping = [
    "1",
    "2",
    "3",
    "4",
    "5"
]
# pred_model = CNNModel().to(device)
pred_model = models.efficientnet_b0(weights=None, num_classes=5)
pred_model.load_state_dict(torch.load(f"{model_name}.pth"))
pred_model.eval()
# print(pred_model)
aD = AudioDataset(path + '/TrainingDatalist.csv')
for i in range(len(aD)):
    input = aD[i][0].unsqueeze(0).to(device)
    target = aD[i][1]
    predicted, expected = predict(pred_model, input, target, class_mapping)
    print(f"Predicted: '{predicted}', Expected: '{expected}'")