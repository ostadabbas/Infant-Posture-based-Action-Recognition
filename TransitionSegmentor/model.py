import torch
from torch import nn

from models import cnnlstm, birnn
def generate_model(opt, device):
    if opt.model == 'cnnlstm':
        model = cnnlstm.CNNLSTM(num_classes=opt.n_classes)
    if opt.model == 'birnn':
        model = birnn.BiRNN()
        #structured = structured_layer.structured_layer()
        #hingeloss = structuredhingeloss.StructuredHingeLoss()

    return model.to(device)
