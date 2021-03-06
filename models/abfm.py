import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Bad solution
# TODO: Fix this
import sys
sys.path.append("../")
from models.bfm import BFM

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

class ABFM(BFM):
    def __init__(self, n_usr, n_itm, k, gamma=[1,1,1,1], alpha=0.0):
        super(ABFM, self).__init__(n_usr, n_itm, k, gamma=[1,1,1,1], alpha=0.0)

        # add softmax function for attention
        self.softmax = nn.Softmax(dim=-1)

    def fm(self, x, debug=False):
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

        b_vecs = b_vecs.squeeze()

        # Target item & basket items relation with attention
        a_t_b = self.softmax(torch.mm(t_vec, b_vecs.t()))
        t_b = b_vecs * a_t_b.t()
        t_b = torch.mm(t_b, t_vec.t()).sum(dim=0, keepdim=True)

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
        a_u_b = self.softmax(torch.mm(u_vec, b_vecs.t()))
        u_b = b_vecs * a_u_b.t()
        u_b = torch.mm(u_b, t_vec.t()).sum(dim=0, keepdim=True)

        # Output
        y = self.w_0 + \
            bias + \
            self.gamma[0]*u_t + \
            self.gamma[1]*t_b + \
            self.gamma[2]*bs + \
            self.gamma[3]*u_b
        # y = self.gamma[0]*u_t + \
        #     self.gamma[1]*t_b + \
        #     self.gamma[2]*bs + \
        #     self.gamma[3]*u_b

        if debug:
            print(f"a_t_b : {a_t_b}\n" \
                  f"a_u_b : {a_u_b}\n" \
                  f"u_t   : {u_t.item():>8.5f}\n" \
                  f"t_b   : {t_b.item():>8.5f}\n" \
                  f"bs    : {bs.item():>8.5f}\n" \
                  f"u_b   : {u_b.item():>8.5f}\n" \
                  f"bias  : {bias.item():>8.5f}\n" \
                  f"w_0   : {self.w_0.item():>8.5f}\n"
                  f"n_b   : {n_b}")
            # print(f"u_t : {u_t.shape}\n" \
            #       f"t_b : {t_b.shape}\n" \
            #       f"bs  : {bs.shape}\n" \
            #       f"u_b : {u_b.shape}\n" \
            #       f"w_0 : {self.w_0.shape}\n"
            #       f"bias: {bias.shape}\n" \
            #       f"y   : {y.shape}")

        return y

    def rank_list(self, x):

        # set 1 to all target items
        x[self.n:self.n+self.m] = 1
        # Bias for each users
        u_bias = torch.mm(x[:self.n].view(1,-1), self.w_bias[:self.n])
        # bias for target items
        t_bias = self.w_bias[self.n:self.n+self.m]
        # bias for basket items
        b_bias = torch.mm(x[self.n+self.m:self.n+2*self.m].view(1,-1), self.w_bias[self.n+self.m:self.n+2*self.m])
        # all bias
        bias = (u_bias + t_bias + b_bias).squeeze()

        # maybe have to extend dim
        x = x.view((1, x.shape[0]))

        # User latent vec
        u_vec = torch.mm(x[:,:self.n], self.u_V)

        # Target item latent vec
        t_vec = self.t_V.view(self.t_V.shape[0], 1, self.t_V.shape[1])

        # Basket items latent vecs and Basket items indices
        index = (x[0,self.n+self.m:self.n+2*self.m]==1).nonzero()
        b_vecs = torch.squeeze(self.b_V[index,:])
        # The number of basket items
        n_b = list(index.shape)[0]

        # User & target item relation
        u_t = (u_vec * t_vec).sum(-1).sum(-1)

        # Target item & basket items relation with attention
        t_b = torch.sum(t_vec * b_vecs, dim=-1)
        a_t_b = self.softmax(t_b).unsqueeze(-1) * b_vecs
        t_b = (a_t_b * t_vec).sum(-1).sum(-1)

        # Among basket items relation
        bs = None
        for i in range(n_b):
            if bs is None:
                bs = torch.mm(b_vecs[i].view(1,-1), b_vecs[i+1:n_b].t()).sum()
            else:
                bs += torch.mm(b_vecs[i].view(1,-1), b_vecs[i+1:n_b].t()).sum()

        # User & basket items relation with attention
        u_b = torch.sum(u_vec * b_vecs, dim=-1)
        a_u_b = self.softmax(u_b).unsqueeze(-1) * b_vecs
        u_b = (a_u_b * t_vec).sum(-1).sum(-1)


        # Output
        y = self.w_0 + \
            bias + \
            self.gamma[0]*u_t + \
            self.gamma[1]*t_b + \
            self.gamma[2]*bs + \
            self.gamma[3]*u_b

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

    epochs = 21
    neg = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ABFM(n, m, k, gamma, alpha).to(device=device)

    # \alpha*||w||_2 is L2 reguralization
    # weight_decay option is for reguralization
    # weight_decay number is \alpha
    optimizer = optim.SGD(model.parameters(), \
                          lr=lr, \
                          momentum=momentum, \
                          weight_decay=weight_decay)

    # Saved directory
    today = datetime.date.today()
    c_time = datetime.datetime.now().strftime("%H-%M-%S")
    save_dir = f"../trained/abfm/{today}/{c_time}"
    os.makedirs(save_dir, exist_ok=True)
    model_name = "ABFM"

    # Load trained parameters
    loaded = False
    if loaded:
        model_path = "../trained/2019-11-08/ABFM_4.pt"
        model.load_state_dict(torch.load(model_path))
        epochs = 5


    # Print Information
    print("{:-^60}".format("Data stat"))
    print(f"# User        : {n}\n" \
          f"# Item        : {m}\n" \
          f"Neg sample    : {neg}\n")
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
    print("{:-^60}".format("Description"))
    print("Without nagative samples")
    print("{:-^60}".format(""))


    for e in range(epochs):
        print("{:-^60}".format(f"epoch {e}"))
        cnt = 0
        ave_loss = 0
        train, _, _ = ds.get_data(neg=neg)
        for x in train:
            optimizer.zero_grad()
            x, label = x[0].to(device), x[1].to(device)
            loss = model(x, delta=label, pmi=1)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ave_loss += loss.item()
            cnt+=1
            if cnt%2500==0:
                print(f"Last loss : {loss.item():>9.6f} at {cnt:6d}, " \
                      f"Label : {label.item():2.0f},   " \
                      f"# basket item : {x.sum().item()-2:3.0f},   " \
                      f"Average loss so far : {ave_loss/cnt:>9.6f}")
        # print(cnt) # => 957264
        torch.save(model.state_dict(), f"{save_dir}/{model_name}_{e}.pt")
        print("{:-^60}".format("end"))

if __name__=="__main__":
    main()
