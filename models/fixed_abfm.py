import os
import random
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
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

torch.set_default_dtype(torch.float64)
class FABFM(BFM):
    def __init__(self, n_usr, n_itm, k, d, h=1, gamma=[1,1,1,1], alpha=0.0):
        super(FABFM, self).__init__(n_usr, n_itm, k, gamma=[1,1,1,1], alpha=0.0)
        """
        d : dim of query, value and key
        h : the number of heads
        """

        # Softmax function
        self.softmax = nn.Softmax(dim=-1)

        # Query, key, value matrices
        self.WQ = nn.Parameter(torch.randn(h, k, d))
        self.WK = nn.Parameter(torch.randn(h, k, d))
        self.WV = nn.Parameter(torch.randn(h, k, d))
        self.sqrt_d = np.sqrt(d)
        self.h = h

        # Matrix to convert multi-head attention to one matrix
        self.O = nn.Parameter(torch.randn(h*d, k))

    def fm(self, x, debug=False):
        # transaction vec whose elements are only 0 or 1.
        x = x.view((1, x.shape[0])).double()

        # Bias for each users and items(target & basket)
        bias = torch.mm(x, self.w_bias)# .view(-1)

        # User latent vec
        u_idx = (x[0,:self.n]==1).nonzero()
        u_vec = self.u_V[u_idx].view(1,-1)

        # Target item latent vec
        t_idx = (x[0,self.n:self.n+self.m]==1).nonzero()
        t_vec = self.t_V[t_idx].view(1,-1)

        # Basket items latent vecs and Basket items indices
        b_idx = (x[0,self.n+self.m:self.n+2*self.m]==1).nonzero()
        b_vecs = self.b_V[b_idx,:].squeeze()
        # The number of basket items
        n_b = list(b_idx.shape)[0]

        # User & target item relation
        u_t = torch.mm(u_vec, t_vec.t())


        # Basic idea ===========================
        # Basket items are input seq words in this context.
        # Target item is a output word
        # In other word, basket items are encoder hidden states
        # and target item is decoder hidden state.
        # ======================================

        # Cal query, key and value for attention
        duplicated_bvecs = b_vecs.expand(self.h,-1,-1)
        duplicated_tvec = t_vec.expand(self.h,-1,-1)
        # TODO: Why is matrix multiplication used to get Q, K and V?
        # bmm is slow?
        # https://github.com/pytorch/pytorch/issues/889
        Q = torch.bmm(duplicated_tvec, self.WQ)
        K = torch.bmm(duplicated_bvecs, self.WK)
        V = torch.bmm(duplicated_bvecs, self.WV)
        # print("Q")
        # print(Q.shape)
        # print(Q)
        # print("K")
        # print(K.shape)
        # print(K)
        # print("V")
        # print(V.shape)
        # print(V)
        # print("\n")

        # Dot product with each Q row and K row
        attn = (Q*K).sum(-1)/self.sqrt_d
        # print("---Q*K---")
        # print(attn.shape)
        # print(attn)
        # softmax based on each vector
        attn = self.softmax(attn).view(self.h,n_b,1)
        # print("---softmax---")
        # print(attn.shape)
        # print(attn)
        # multiply attention and corresponding V row vector
        attn_V = attn*V
        # print("---basket V---")
        # print(b_vecs.shape)
        # print(b_vecs)
        # print("---attn*V---")
        # print("(head, # basket, latent dim)")
        # print(attn.shape)
        # print(attn)

        # TODO
        # Now we got basket vectors multiplied by attention.
        # Should we just take dot product with basket vectors and a target item?
        # If so, we have to convert target item vector
        # to the same dim vectors of basket item vectors.
        # How do I do that?
        #
        # TODO: Attention doesn't seem to work...
        #       Have to solve


        # Target item & basket items relation with attention
        # Just multiply attn and target item vector
        # print("---- t_vec ----")
        # print(t_vec.shape)
        # print(t_vec)
        # Experimental ------------
        attn_V = attn_V.sum(1).view(1,-1)
        attn_V = torch.mm(attn_V, self.O)
        t_b = torch.mm(t_vec, attn_V.t())
        # print("---- t_b ----")
        # print(t_b.shape)
        # print(t_b)
        # Original -----------------
        # We cannot use FFNN to convert target basket dot product vectors
        # because the number of basket items are dynamic.
        # How do I convert to scalar? addition? multiplication?
        # t_b = (attn_V*t_vec).sum(-1)
        # t_b = t_b.sum()/(n_b*self.h) # just addition to convert scalar


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
        # Experimental ------------
        u_b = torch.mm(u_vec, attn_V.t())
        # Original ------------
        # Just multiply attn and target item vector
        # u_b = (attn_V*u_vec).sum(-1)
        # We cannot use FFNN to convert target basket dot product vectors
        # because the number of basket items are dynamic.
        # How do I convert to scalar? addition? multiplication?
        # u_b = u_b.sum()/(n_b*self.h) # just addition to convert scalar

        # Output
        y = self.w_0 + \
            bias + \
            self.gamma[0]*u_t + \
            self.gamma[1]*t_b + \
            self.gamma[2]*bs + \
            self.gamma[3]*u_b

        if debug:
            print(f"attn  : {attn}\n" \
                  f"WQ    : {self.WQ}\n" \
                  f"WK    : {self.WK}\n" \
                  # f"u_t   : {u_t.item():>8.5f}\n" \
                  # f"t_b   : {t_b.item():>8.5f}\n" \
                  # f"bs    : {bs.item():>8.5f}\n" \
                  # f"u_b   : {u_b.item():>8.5f}\n" \
                  f"u_t   : {u_t.item()}\n" \
                  f"t_b   : {t_b.item()}\n" \
                  f"bs    : {bs.item()}\n" \
                  f"u_b   : {u_b.item()}\n" \
                  f"bias  : {bias.item()}\n" \
                  f"w_0   : {self.w_0.item():>8.5f}\n" \
                  f"n_b   : {n_b}")
            # print(f"u_vec : {u_vec}\n"\
            #       f"t_vec : {t_vec}\n"\
            #       f"b_vecs: {b_vecs}\n"\
            #       f"bias_V: {self.w_bias}")
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

        duplicated_bvecs = b_vecs.expand(self.h,-1,-1)

        Q = torch.bmm(duplicated_bvecs, self.WQ)
        K = torch.bmm(duplicated_bvecs, self.WK)
        V = torch.bmm(duplicated_bvecs, self.WV)
        attn = (Q*K).sum(-1)/self.sqrt_d
        attn = self.softmax(attn).view(self.h,n_b,1)
        attn_V = (attn*V).unsqueeze(dim=1)
        # Target item & basket items relation with attention
        # TODO: Fix stupid sum
        t_b = (attn_V*t_vec).sum(-1).sum(-1).sum(0)
        t_b/=(n_b*self.h) # just addition to convert scalar
        # User & basket items relation with attention
        u_b = (attn_V*u_vec).sum(-1)
        u_b = u_b.sum()/(n_b*self.h) # just addition to convert scalar


        # Among basket items relation
        bs = None
        for i in range(n_b):
            if bs is None:
                bs = torch.mm(b_vecs[i].view(1,-1), b_vecs[i+1:n_b].t()).sum()
            else:
                bs += torch.mm(b_vecs[i].view(1,-1), b_vecs[i+1:n_b].t()).sum()

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

    seed=1234
    seed_everything(seed)

    ds = Data()
    _, _, _ = ds.get_data()

    """
    n: # users
    m: # items
    k: latent vec dim
    """
    n_usr = len(ds.usrset)
    n_itm = len(ds.itemset)
    k = 16

    gamma=[1,1,1,1]
    alpha=0.0
    d=k
    h=2

    lr=0.0001
    momentum=0
    weight_decay=0.01

    epochs=21
    neg=2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = FABFM(n_usr, n_itm, k, d, h, gamma, alpha).to(device=device)

    # \alpha*||w||_2 is L2 reguralization
    # weight_decay option is for reguralization
    # weight_decay number is \alpha
    optimizer = optim.SGD(model.parameters(), \
                          lr=lr, \
                          momentum=momentum, \
                          weight_decay=weight_decay)
    # optimizer = optim.Adam(model.parameters(), \
    #                       lr=lr, \
    #                       weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    # Saved directory
    today = datetime.date.today()
    c_time = datetime.datetime.now().strftime("%H-%M-%S")
    save_dir = f"../trained/fixed_abfm/{today}/{c_time}"
    os.makedirs(save_dir, exist_ok=True)
    model_name = "FABFM"

    # Load trained parameters
    loaded = False
    if loaded:
        model_path = "../trained/"
        model.load_state_dict(torch.load(model_path))
        epochs = 5


    # Print Information
    print("{:-^60}".format("Data stat"))
    print(f"# User        : {n_usr}\n" \
          f"# Item        : {n_itm}\n" \
          f"Neg sample    : {neg}")
    print("{:-^60}".format("Optim status"))
    print(f"Optimizer     : {optimizer}\n" \
          f"Criterion     : {criterion}\n" \
          f"Learning rate : {lr}\n" \
          f"Momentum      : {momentum}\n" \
          f"Weight decay  : {weight_decay}")
    print("{:-^60}".format("Model/Learning status"))
    print(f"Model name    : {model_name}\n" \
          f"Mid dim k     : {k}\n" \
          f"Q,K,M dim d   : {d}\n" \
          f"Head          : {h}\n" \
          f"Gamma         : {gamma}\n" \
          f"Alpha         : {alpha}\n" \
          f"Epochs        : {epochs}\n" \
          f"Loaded        : {loaded}")
    if loaded:
        print(f"Learned model : {model_path}")
    print("{:-^60}".format("Description"))
    print("Addition and normalization with h*n_b.\n"\
          "Attention is applied to only t_b and u_b.\n"\
          "Use same vector for target and basket.\n"\
          "Changed random seed to get train data.\n"\
          "Use double type for all layers.")
    print("{:-^60}".format(""), flush=True)


    for e in range(epochs):
        cnt = 0
        ave_loss = 0
        random.seed(seed)
        seed = random.randint(0, 9999)
        print("{:-^60}".format(f"epoch {e}, seed={seed}"))

        train, _, _ = ds.get_data(neg=neg, seed=seed)
        for x in train:
            optimizer.zero_grad()
            x, label = x[0].to(device).double(), x[1].to(device).double()
            loss = model(x, delta=label, pmi=1)
            if label==-1:
                loss = criterion(loss, label+1)
            else:
                loss = criterion(loss, label)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ave_loss += loss.item()
            cnt+=1
            if cnt%2500==0:
                print(f"Last loss : {loss.item():>10.6f} at {cnt:6d}, " \
                      f"Label : {label.item():2.0f},   " \
                      f"# basket item : {x.sum().item()-2:3.0f},   " \
                      f"Average loss so far : {ave_loss/cnt:>9.6f}", flush=True)
        # print(cnt) # => 957264
        # torch.save(model.state_dict(), f"{save_dir}/{model_name}_{e}.pt")
        # Better way to save model?
        state = {"name": model_name, "epoch": e, "state_dict": model.state_dict(),
                 "neg":neg, "optimizer": optimizer.state_dict(), "k": k,
                 "d": d, "h": h, "gamma": gamma, "alpha": alpha}
        torch.save(state, f"{save_dir}/{model_name}_{e}.pt")

        """
        # Load func is like below
        def load_model(filename, model_name=FABFM):
            if os.path.isfile(filename):
                # Load file
                checkpoint = torch.load(filename)

                # Load dataset setting
                neg = checkpoint["neg"]
                ds = Data(root_dir="./data/ta_feng/")
                _, _, _= ds.get_data(neg=neg)
                n_usr = len(ds.usrset)
                n_itm = len(ds.itemset)

                # Load network
                optimizer = checkpoint["optimizer"]
                k = checkpoint["k"]
                gamma = checkpoint["gamma"]
                alpha = checkpoint["alpha"]
                norm = checkpoint["norm"]
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                if model_name=="FABFM":
                    d = checkpoint["d"]
                    h = checkpoint["h"]
                    from models.fixed_abfm import FABFM
                    model = FABFM(n_usr, n_itm, k, d, h, gamma, alpha).to(device=device)
                elif model_name=="ABFM":
                    from models.abfm import ABFM
                    model = ABFM(n_usr, n_itm, k, gamma, alpha).to(device=device)
                elif model_name=="BFM":
                    from models.bfm import BFM
                    model = BFM(n_usr, n_itm, k, gamma, alpha).to(device=device)
                model.load_state_dict(checkpoint["state_dict"])
                return model
            else:
                print(f"There is not {filename}")
                return None
        """


        print("{:-^60}".format("end"))

if __name__=="__main__":
    main()
