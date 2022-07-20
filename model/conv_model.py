#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import pdb
emo_dict = {0: 'neu', 1: 'hap', 2: 'sad', 3: 'ang'}
class_dict = {'emotion': 4, 'affect': 3, 'gender': 2}

class conv_classifier(nn.Module):
    def __init__(self, pred, audio_size, txt_size, hidden_size, att=None):
        super(conv_classifier, self).__init__()
        self.dropout_p = 0.2
        self.test_conf = None
        self.rnn_cell = nn.GRU
        
        num_class = class_dict[pred]
        
        hidden_size = 64
        self.emo_classifier = nn.Sequential(
            nn.Linear(hidden_size*2*2, 128),
            nn.ReLU(),
            nn.Linear(128, num_class)
        )
        
        self.arousal_pred = nn.Sequential(
            nn.Linear(hidden_size*2*2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.valence_pred = nn.Sequential(
            nn.Linear(hidden_size*2*2, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.audio_conv = nn.Sequential(
            nn.Conv1d(audio_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),

            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(self.dropout_p),
        )
        
        self.text_conv = nn.Sequential(
            nn.Conv1d(txt_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),

            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
        )
        
        self.audio_rnn = self.rnn_cell(input_size=128, hidden_size=hidden_size,
                                       num_layers=2, batch_first=True,
                                       dropout=self.dropout_p, bidirectional=True)
        self.txt_rnn = self.rnn_cell(input_size=128, hidden_size=hidden_size,
                                     num_layers=2, batch_first=True,
                                     dropout=self.dropout_p, bidirectional=True)
        
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)

    def forward(self, audio, txt_embedding):

        audio, txt_embedding = audio.float(), txt_embedding.float()
        audio = audio.permute(0, 2, 1)
        audio = self.audio_conv(audio)
        audio = audio.permute(0, 2, 1)
        
        txt_embedding = txt_embedding.permute(0, 2, 1)
        txt_embedding = self.text_conv(txt_embedding)
        txt_embedding = txt_embedding.permute(0, 2, 1)
        
        audio, h_state = self.audio_rnn(audio)
        txt_embedding, h_state = self.txt_rnn(txt_embedding)
        
        audio = torch.mean(audio, dim=1)
        txt_embedding = torch.mean(txt_embedding, dim=1)
    
        final_feat = torch.cat((audio, txt_embedding), 1)
        preds = self.emo_classifier(final_feat)
        arousal = self.arousal_pred(final_feat)
        valence = self.valence_pred(final_feat)
        return preds, arousal, valence


class audio_conv(nn.Module):
    def __init__(self, pred, audio_size, dropout):
        super(audio_conv, self).__init__()
        self.dropout_p = dropout
        
        self.pred_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
        
        self.conv = nn.Sequential(
            nn.Conv1d(audio_size, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),
            
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(self.dropout_p),

            nn.Conv1d(96, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(self.dropout_p),
        )
        
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)

    def forward(self, audio):

        audio = audio.float()
        audio = audio.permute(0, 2, 1)
        audio = self.conv(audio)
        audio = audio.permute(0, 2, 1)
        audio = torch.mean(audio, dim=1)
        
        preds = self.pred_layer(audio)
        preds = torch.log_softmax(preds, dim=1)
        return preds
    

class audio_conv_rnn(nn.Module):
    def __init__(self, pred, audio_size, dropout, label_size=4):
        super(audio_conv_rnn, self).__init__()
        self.dropout_p = dropout
        self.rnn_cell = nn.GRU
        
        self.pred_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, label_size)
        )
        
        self.conv = nn.Sequential(
            nn.Conv1d(audio_size, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(self.dropout_p),
        )
        
        self.rnn = self.rnn_cell(input_size=64, hidden_size=64, 
                                 num_layers=1, batch_first=True, 
                                 dropout=self.dropout_p, bidirectional=True).cuda()
        
        self.init_weight()

    def init_weight(self):
        for m in self._modules:
            if type(m) == nn.Linear:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)
            if type(m) == nn.Conv1d:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                m.bias.data.fill_(0.01)

    def forward(self, audio, lengths=None):

        # conv modul
        audio = self.conv(audio.float().permute(0, 2, 1))
        audio = audio.permute(0, 2, 1)
        if lengths is None:
            # output
            x_output, _ = self.rnn(audio)
            z = torch.mean(x_output, dim=1)
        else:
            # rnn modul
            audio_packed = pack_padded_sequence(audio, lengths.cpu(), batch_first=True, enforce_sorted=False)
            output_packed, h_state = self.rnn(audio_packed)
            x_output, _ = pad_packed_sequence(output_packed, True, total_length=audio.size(1))
            
            # output
            z = torch.sum(x_output, dim=1) / torch.unsqueeze(lengths, 1)
        preds = self.pred_layer(z)
        preds = torch.log_softmax(preds, dim=1)
        return preds