import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import bfm

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

class ABFM(bfm.BFM):
    def __init__(self, n, m, k, gamma=[1,1,1,1], alpha=0.0):
        super(ABFM, self).__init__(n, m, k, gamma=[1,1,1,1], alpha=0.0)

        # add softmax functin
        self.softmax = nn.Softmax(dim=-1)

    def fm(self, x):
        # transaction vec whose elements are only 0 or 1.
        x = x.view((1, x.shape[0]))

        # Bias for each users and items(target & basket)
        bias = torch.mm(x, self.w_bias)# .view(-1)

        # User latent vec
        u_vec = torch.mm(x[:,:self.n], self.u_V)

        # Target item latent vec
        t_vec = torch.mm(x[:,self.n:self.n+self.m], self.t_V)

        # Basket items latent vecs and Basket items indices
        index = (x[0,self.n+self.m:self.n+2*self.m]==1).nonzero()
        b_vecs = self.b_V[index,:]
        # The number of basket items
        n_b = list(index.shape)[0]

        # User & target item relation
        u_t = torch.mm(u_vec, t_vec.t())

        # Target item & basket items relation with attention
        b_vecs = b_vecs.squeeze()
        t_b = torch.mm(t_vec, b_vecs.t())
        a_t_b = self.softmax(t_b)
        t_b = torch.mm(a_t_b, t_b.t()).sum(dim=-1, keepdim=True)

        # Among basket items relation
        # faster (maybe 2x faster)
        # At this point range(n_b) makes error because the last for loop i+1==n_b,
        # so b_vecs[i+1:n:b] is size 0 vec. That size 0 vec cause error
        # like : https://github.com/pytorch/pytorch/issues/20006
        bs = None
        for i in range(n_b-1):
            if bs is None:
                bs = torch.mm(b_vecs[i].view(1,-1), b_vecs[i+1:n_b].t()).sum(dim=-1, keepdim=True)
            else:
                bs += torch.mm(b_vecs[i].view(1,-1), b_vecs[i+1:n_b].t()).sum(dim=-1, keepdim=True)

        # User & basket items relation with attention
        u_b = torch.mm(u_vec, b_vecs.t())
        a_u_b = self.softmax(u_b)
        u_b = torch.mm(a_u_b, u_b.t()).sum(dim=-1, keepdim=True)

        # Output
        y = self.w_0 + \
            bias + \
            self.gamma[0]*u_t + \
            self.gamma[1]*t_b + \
            self.gamma[2]*bs + \
            self.gamma[3]*u_b

        # print(f"u_t : {u_t}\n" \
        #       f"t_b : {t_b}\n" \
        #       f"bs  : {bs}\n" \
        #       f"u_b : {u_b}\n" \
        #       f"w_0 : {self.w_0}\n"
        #       f"bias: {bias}\n" \
        #       f"y   : {y}")
        # print(f"u_t : {u_t.shape}\n" \
        #       f"t_b : {t_b.shape}\n" \
        #       f"bs  : {bs.shape}\n" \
        #       f"u_b : {u_b.shape}\n" \
        #       f"w_0 : {self.w_0.shape}\n"
        #       f"bias: {bias.shape}\n" \
        #       f"y   : {y.shape}")

        return y

def main():
    import sys
    sys.path.append("../")
    import datetime
    from torch.utils.data import DataLoader

    from dataloader import Data

    ds = Data()
    train, test, valid = ds.get_data()

    """
    n: # users
    m: # items
    k: latent vec dim
    """
    n = len(ds.usrset)
    m = len(ds.itemset)
    k = 32

    gamma=[1,1,1,1]
    alpha=0.0

    lr=0.0001
    momentum=0
    weight_decay=0.01

    model = ABFM(n, m, k, gamma, alpha)
    # \alpha*||w||_2 is L2 reguralization
    # weight_decay option is for reguralization
    # weight_decay number is \alpha
    optimizer = optim.SGD(model.parameters(), \
                          lr=lr, \
                          momentum=momentum, \
                          weight_decay=weight_decay)

    today = datetime.date.today()
    # Saved directory
    os.makedirs(f"../trained/abfm/{today}", exist_ok=True)
    model_name = "ABFM"
    epochs = 21

    # Load trained parameters
    loaded = False
    if loaded:
        model_path = "../trained/2019-11-08/ABFM_4.pt"
        model.load_state_dict(torch.load(model_path))
        epochs = 5


    # Print Information
    print("{:-^60}".format("Data stat"))
    print(f"# User        : {n}\n" \
          f"# Item        : {m}")
    print("{:-^60}".format("Optim status"))
    print(f"Optimizer     : {optimizer}\n" \
          f"Learning rate : {lr}\n" \
          f"Momentum      : {momentum}\n" \
          f"Weight decay  : {weight_decay}")
    print("{:-^60}".format("Model/Learning status"))
    print(f"Model name    : {model_name}\n" \
          f"Mid dim       : {k}\n" \
          f"Gamma         : {gamma}\n" \
          f"Alpha         : {alpha}\n" \
          f"Epochs        : {epochs}\n" \
          f"Loaded        : {loaded}")
    if loaded:
        print(f"Learned model : {model_path}")
    print("{:-^60}".format(""))


    for e in range(epochs):
        print("{:-^60}".format(f"epoch {e}"))
        cnt = 0
        ave_loss = 0
        train, _, _ = ds.get_data()
        for x in train:
            optimizer.zero_grad()
            x, label = x[0], x[1]
            loss = model(x, delta=label, pmi=1)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ave_loss += loss.item()
            cnt+=1
            if cnt%2500==0:
                print(f"Last loss : {loss.item():3.6f} at {cnt:6d}, " \
                      f"Label : {label.item():2.0f},   " \
                      f"# basket item : {x.sum().item()-2:3.0f}    " \
                      f"Average loss so far : {ave_loss/cnt:3.6f}")
        print(cnt) # => 957264
        torch.save(model.state_dict(), f"../trained/abfm/{today}/{model_name}_{e}.pt")
        print("{:-^60}".format("end"))

if __name__=="__main__":
    main()
