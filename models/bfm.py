import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# for reproducibility
def seed_everything(seed=1234):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed=1234
seed_everything(seed)

class BFM(nn.Module):
    """
    2-way pairwise BASKET-SENSITIVE FACTORIZATION MACHINE(BFM)

    Variables ----------
    p:      length of a transaction(n+2m)
    n:      # number of users
    m:      # number of items
    k:      latent vec dim
    gamma:  gamma = to select relation term
    w_0:    global bias (\mu_0, 1st term in eq7)
    bias_u: bias for users (\mu_i, 2st term in eq7)
    bias_i: bias for target items
    bias_b: bias for basket items
    u_V:    latent vec for user (k*n matrix)
    t_V:    latent vec for target item (k*m matrix)
    b_V:    latent vec for basket items (k*m matrix)
    alpha:  alpha in eq11. If alpha is 0, model is BFM.
    --------------------
    """
    def __init__(self, n, m, k, gamma=[1,1,1,1], alpha=0.0):
        super(BFM, self).__init__()
        # biases
        self.w_0 = torch.randn(1, dtype=torch.float32, requires_grad=True)
        self.w_bias = nn.Parameter(torch.rand(n+2*m, 1))

        # input (input must one-hot vector)
        self.u_V = nn.Parameter(torch.normal(0, 1, size=(n, k)))
        self.t_V = nn.Parameter(torch.normal(0, 1, size=(m, k)))
        self.b_V = nn.Parameter(torch.normal(0, 1, size=(m, k)))

        # sigmoid
        self.sigmoid = nn.Sigmoid()

        # hyper parameter
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, delta, pmi):
        """
        Input data has to be 1 transaction.
        x[0:n+2*m-1] is target transaction t,
        x[n+2*m:] is t^m transaction of t
        pmi has to be calculated in preprocessing.
        """
        # transaction
        t = x[:n+2*m]
        # t^m transaction
        tm = x[n+2*m:]
        y = self.fm(t)
        y = self.sigmoid(y*delta.float())

        # final equation
        # y += self.alpha*pmi*self.constrain(t, tm)

        return y

    def fm(self, x):
        x = x.view((1, x.shape[0]))                             # maybe have to extend dim
        # Bias for each users and items(target & basket)
        bias = torch.mm(x, self.w_bias)

        # Latent vectors
        # x = x[:n+2*m]
        # x = x.view((1, x.shape[0]))                           # maybe have to extend dim

        # User latent vec
        u_vec = torch.mm(x[:,:n], self.u_V)

        # Target item latent vec
        t_vec = torch.mm(x[:,n:n+m], self.t_V)

        # Basket items latent vecs
        # Basket items indices
        index = (x[0,n+m:n+2*m]==1).nonzero()
        b_vecs = self.b_V[index,:]
        # The number of basket items
        n_b = list(index.shape)[0]

        # User & target item relation
        u_t = torch.dot(u_vec.view(-1), t_vec.view(-1))

        # Target item & basket items relation
        t_b = None
        for i in range(n_b):
            if t_b is None:
                t_b = torch.mm(t_vec, torch.t(b_vecs[i]))
            else:
                t_b += torch.mm(t_vec, torch.t(b_vecs[i]))

        # Among basket items relation
        bs = None
        for i in range(n_b):
            for j in range(i+1, n_b):
                if bs is None:
                    bs = torch.mm(b_vecs[i], torch.t(b_vecs[j]))
                else:
                    bs += torch.mm(b_vecs[i], torch.t(b_vecs[j]))

        # User & basket items relation
        u_b = None
        for i in range(n_b):
            if u_b is None:
                u_b = torch.mm(u_vec, torch.t(b_vecs[i]))
            else:
                u_b += torch.mm(u_vec, torch.t(b_vecs[i]))

        # Output
        y = self.w_0 + bias + \
            self.gamma[0]*u_t + \
            self.gamma[1]*t_b + \
            self.gamma[2]*bs + \
            self.gamma[3]*u_b

        return y

        def constrain(self, t, tm):
            cons = 0.5*(self.fm(t)-self.fm(tm)).pow(2)
            return cons

        def get_vec(self, x, type="u"):
            """
            type will be u(user), i(item)
            """
            pass


if __name__=="__main__":
    import sys
    sys.path.append("../")
    from tmp_dataloader import Data

    ds = Data()
    train, test, valid = ds.get_data()

    """
    n:      # users
    m:      # items
    k:      latent vec dim
    """
    n = len(ds.usrset)
    m = len(ds.itemset)
    k = 32

    model = BFM(n, m, k, gamma=[1,1,1,1], alpha=0.0)
    criterion = nn.BCELoss()
    # criterion = nn.NLLLoss()
    # \alpha*||w||_2 is L2 reguralization
    # weight_decay option is for reguralization
    # weight_decay number is \alpha
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.01)

    cnt = 0
    for x in train:
        optimizer.zero_grad()
        x, label = x[0], x[1]
        y = model(x, delta=label, pmi=1)
        if label.item()==-1:
            loss = criterion(y, torch.tensor([[0.]]))
        else:
            loss = criterion(y, label)
        loss.backward()
        optimizer.step()
        if cnt%2500==0:
            print(f"Loss : {loss}")
        cnt+=1
    print(cnt) # => 957264


