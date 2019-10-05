import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class FM(nn.Module):
    """
    2-way pairwise

    Variables ----------
    n: length of transaction
    k: latent vec dim
    w_0: global bias
    w: bias for each items
    V: latent vec
    --------------------
    """
    def __init__(self, n, k):
        super(FM, self).__init__()
        self.w_0 = torch.randn(1, dtype=torch.float32, requires_grad=True)
        self.w = nn.Parameter(torch.rand(n, 1))
        self.V = nn.Parameter(torch.rand(n, k))

    def forward(self, x):
        """
        x is input and has to be one-hot vector
        """
        linear = torch.mm(self.w, x)
        inter1 = torch.mm(x, self.V).pow(2)
        inter2 = torch.mm(x.pow(2), self.V.pow(2))
        intersection = 0.5*(inter1-inter2).sum()
        
        y = w_0 + linear + intersection
        return y

if __name__=="__main__":

    # criterion = nn.CrossEntropyLoss()

    # \alpha*||w||_2 is L2 reguralization
    # weight_decay option is for reguralization
    # weight_decay number is \alpha
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
