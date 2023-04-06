import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, epsilon=0, min_gap=5, min_size=5, max_size=float('inf')):
        super(CustomLoss, self).__init__()
        self.eps = epsilon
        self.onset = 0
        self.offset = 0
        self.MIN_GAP = min_gap
        self.MIN_SIZE = min_size
        self.MAX_SIZE = max_size

        # Define output and input as trainable parameters
        #self.output = nn.Parameter(torch.zeros(1, requires_grad=True))
        #self.input = nn.Parameter(torch.zeros(1, requires_grad=True))
        #self.onset = nn.Parameter(torch.zeros(1, requires_grad=True))
        #self.offset = nn.Parameter(torch.zeros(1, requires_grad=True))


    def forward(self, w_phi, y, eps=5):
        batch_size = y.size(0)

        loss = 0
        for id in range(batch_size):
            self.onset = 1
            self.offset = self.onset + self.MIN_SIZE
            # Update input to be the current batch element
            input = nn.Parameter(w_phi[id].t())

            # Compute output for current batch element
            self.output = input[self.onset] + input[self.offset] + (max(0, abs(y[id][0] - self.onset) - eps) + max(0, abs(y[id][1] - self.offset) - eps)) / 2

            for i in range(self.MIN_GAP, input.size(0)):
                for j in range(i + self.MIN_SIZE, min(i + self.MAX_SIZE, input.size(0))):
                    tmp = input[i] + input[j] + (max(0, abs(y[id][0] - i) - eps) + max(0, abs(y[id][1] - j) - eps)) / 2
                    if tmp > self.output:
                        self.output = tmp
                        self.onset = i
                        self.offset = j
      
            loss += self.output
            
        return loss / batch_size

    def predict(self, w_phi):
        batch_size = w_phi.size(0)

        loss = 0
        for id in range(batch_size):
            onset = 1
            offset = onset + self.MIN_SIZE
            # Update input to be the current batch element
            input = w_phi[id].t()

            # Compute output for current batch element
            output = input[onset] + input[offset]
            for i in range(self.MIN_GAP, input.size(0)):
                for j in range(i + self.MIN_SIZE, min(i + self.MAX_SIZE, input.size(0))):
                    tmp = input[i] + input[j]
                    if tmp > output:
                        output = tmp
                        onset = i
                        offset = j            
        return output, onset, offset

