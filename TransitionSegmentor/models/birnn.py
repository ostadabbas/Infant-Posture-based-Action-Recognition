import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

class BiRNN(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.5, is_cuda=True):
        super(BiRNN, self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_size
        self.num_layers = num_layers

        self.hidden = None
        self.biLSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size // 2, num_layers=num_layers, bidirectional=True,dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() and is_cuda else "cpu")
        print(self.device)
        self.w = nn.Parameter(torch.zeros(1, hidden_size, requires_grad=True))
        #self.score = torch.zeros(x.size(0),x.size(1), requires_grad=True)
        #self.output = torch.zeros(x.size(0),2, requires_grad=True)

        print("Model runs on : {}, num_of_layers {}".format(self.device,self.num_layers))
        # self.init_hidden()
        self.to(self.device)

    def init_hidden(self, inputs):
        # if is_cuda:
        self.hidden = (
            Variable(torch.zeros(2 * self.num_layers, inputs.size(0), self.hidden_dim // 2)),
            Variable(torch.zeros(2 * self.num_layers, inputs.size(0), self.hidden_dim // 2)))
        # else:
        #     self.hidden = (
        #         torch.zeros(2, 1, self.hidden_dim // 2),
        #         torch.zeros(2, 1, self.hidden_dim // 2))

    def forward(self, inputs, seq_lengths):
        """
        run the lstm
        :param sequence: sound features sequence
        :return: phi for every time frame
        """
        self.init_hidden(inputs)

        # Pack the sequence into a PackedSequence object
        packed_inputs = nn.utils.rnn.pack_padded_sequence(inputs, seq_lengths, batch_first=True)

        # Pass the packed sequence through the LSTM layer
        packed_outputs, self.hidden = self.biLSTM(packed_inputs, self.hidden)
        
        # Unpack the PackedSequence object into a tensor
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)

        lstm_out = self.dropout(lstm_out)

        w = self.w.view(1,self.w.size(0), self.w.size(1))
        x_T = lstm_out.transpose(1,2)
        score = (w * lstm_out).sum(dim=-1)
        print('resulttttttttttttttttttttttttttt')
        print(score.size())
        #value, self.output = torch.topk(score, 2)
        #print(self.output)
        return score
        #return lstm_out
