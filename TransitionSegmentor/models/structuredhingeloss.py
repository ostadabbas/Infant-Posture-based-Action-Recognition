import torch
import torch.nn as nn

class StructuredHingeLoss(nn.Module):
    def __init__(self, epsilon=0, min_gap=5, min_size=5, max_size=float('inf')):
        super(StructuredHingeLoss, self).__init__()
        self.eps = epsilon
        self.onset = 0
        self.offset = 0
        self.MIN_GAP = min_gap
        self.MIN_SIZE = min_size
        self.MAX_SIZE = max_size

    def forward(self, inputs):
        outputs = torch.zeros(inputs.size(0))
        onsets = torch.zeros(inputs.size(0), requires_grad=True)
        offsets = torch.zeros(inputs.size(0), requires_grad=True)

        for id in range(len(inputs)):

            self.onset = 1
            self.offset = self.onset+self.MIN_SIZE

            input = inputs[id].t()
            #target = targets[id]
            print(input.size())
            output = input[self.onset] + input[self.offset]

            for i in range(self.MIN_GAP, input.size(0)):
                for j in range(i + self.MIN_SIZE, min(i + self.MAX_SIZE, input.size(0))):
                    tmp = input[i] + input[j]
                    if tmp > output:
                        output = tmp
                        self.onset = i
                        self.offset = j
            outputs[id] = output
            onsets[id] = self.onset
            offsets[id] = self.offset
        return outputs, onsets, offsets