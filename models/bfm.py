import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class BFM(nn.Module):
    """
    2-way pairwise BASKET-SENSITIVE FACTORIZATION MACHINE(BFM)

    Variables ----------
    p: length of a transaction(n+2m)
    n: # number of users
    m: # number of items
    k: latent vec dim
    gamma: gamma = to select relation term
    w_0: global bias (\mu_0, 1st term in eq7)
    bias_u: bias for users (\mu_i, 2st term in eq7)
    bias_i: bias for target items
    bias_b: bias for basket items
    u_V: latent vec for user (k*n matrix)
    t_V: latent vec for target item (k*m matrix)
    b_V: latent vec for basket items (k*m matrix)
    --------------------
    """
    def __init__(self, n, m, k, gamma=[1,1,1,1]):
        super(FM, self).__init__()
        # biases
        self.w_0 = torch.randn(1, dtype=torch.float32, requires_grad=True)
        self.w_bias = nn.Parameter(torch.rand(n+2*m, 1))

        # input (input must one-hot vector)
        self.u_V = nn.Parameter(torch.rand(k, n))
        self.t_V = nn.Parameter(torch.rand(k, m))
        self.b_V = nn.Parameter(torch.rand(k, m))

    def forward(self, x):
        # bias for each users and items(target & basket)
        bias = torch.mm(self.w_bias, x)

        # latent vectors
        # x = x.view((1, x.shape[0]))                           # maybe have to extend dim
        u_vec = torch.mm(self.u_V, x[0:n-1]).view(-1)           # user latent vec
        t_vec = torch.mm(self.t_V, x[n:n+m-1])).view(-1)        # target item latent vec
        b_vecs = torch.mm(self.b_V, x[n+m:n+2*m-1]).view(-1)    # basket items latent vecs
        n_b = b_vecs.shape[1]                                   # number of basket items

        # user & target item relation
        u_t = torch.dot(u_vec, t_vec)

        # target item & basket items relation
        for i in range(n_b):
            t_b += torch.dot(t_vec, b_vecs[i])

        # among basket items relation
        for i in range(n_b):
            for j in range(i+1, n_b):
                bs += torch.dot(b_vecs[i], b_vecs[j])

        # user & basket items relation
        for i in range(n_b):
            u_b += torch.dot(u_vec, b_vecs[i])

        # output
        y = self.w_0 + bias + gamma[0]*u_t + gamma[1]*t_b + gamma[2]*bs + gamma[3]*u_b
        
        return y

        def pmi(self, t_1, t_2):
            pass


if __name__=="__main__":

    # criterion = nn.CrossEntropyLoss()

    # \alpha*||w||_2 is L2 reguralization
    # weight_decay option is for reguralization
    # weight_decay number is \alpha
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)
