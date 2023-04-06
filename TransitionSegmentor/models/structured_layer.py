import torch.nn as nn
import torch


__author__ = 'YosiShrem'


class structured_layer(nn.Module):
    def __init__(self, dim=64,is_cuda=True,no_tagging=True):
        super(structured_layer, self).__init__()
        self.dim = dim
        self.w_neg = nn.Parameter(torch.zeros(1, dim, requires_grad=True))
        # self.w1 = nn.Parameter(torch.zeros(1, dim, requires_grad=True))

        self.MIN_GAP = 5
        self.MIN_SIZE = 5
        self.MAX_SIZE =2000
        self.support_tagging = not no_tagging
        self.device = torch.device("cuda" if torch.cuda.is_available() and is_cuda else "cpu")

        print("Structured runs on : {}".format(self.device))
        self.to(self.device)


    def forward(self, x):
        score = torch.zeros(x.size(0),x.size(1), requires_grad=True)
        x = x.to(self.device)
        '''
        for i in range(x.size(0)):
            x_i = x[i].t()
            x_i = torch.mm(self.w_neg, x_i)
            score[i] = x_i.clone().squeeze()
        '''
        w = self.w_neg.view(1,self.w_neg.size(0), self.w_neg.size(1))
        x_T = x.transpose(1,2)
        score = (w * x).sum(dim=-1)
        return score

    def predict(self, inputs):
        """
        go over all possible segmentations and choose the one with the highest score
        :param input: the score for each time frame. w*phi(x,y_i)
        :return: score, onset,offset
        """
        score_list = []
        onset_list = []
        offset_list = []
        vot_list = []
        
        
        for id in range(len(inputs)):
            input = inputs[id]

            #print(input)
            input=input.view(-1).cpu()
            #print(input.shape)

            onset = 1
            offset = onset+self.MIN_SIZE
            score = input[onset] + input[offset]
            for i in range(self.MIN_GAP, input.shape[0]):
                max_length = min(i+self.MAX_SIZE , input.shape[0])
                for j in range(i + self.MIN_SIZE, max_length):
                    tmp = input[i] + input[j]
                    if tmp > score:
                        score = tmp
                        onset = i
                        offset = j            
            #print(onset, offset)
            score_list.append(score)
            onset_list.append(onset)
            offset_list.append(offset)
            vot_list.append(offset-onset)

        #return score,vot,onset,ofset
        return score_list, vot_list , onset_list, offset_list