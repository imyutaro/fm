import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

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
    def __init__(self, n_usr, n_itm, k, gamma=[1,1,1,1], alpha=0.0, norm=False):
        super(BFM, self).__init__()
        # parameters
        self.n = n_usr
        self.m = n_itm
        self.k = k

        # biases
        self.w_0 = nn.Parameter(torch.randn(1))
        self.w_bias = nn.Parameter(torch.randn(n_usr+2*n_itm, 1))

        # latent vectors (input must one-hot vector)
        # self.u_V = nn.Parameter(torch.normal(0, 1, size=(n, k)))
        # self.t_V = nn.Parameter(torch.normal(0, 1, size=(m, k)))
        # self.b_V = nn.Parameter(torch.normal(0, 1, size=(m, k)))
        self.u_V = nn.Parameter(torch.randn(n_usr, k))
        # self.t_V = nn.Parameter(torch.randn(n_itm, k))
        self.b_V = nn.Parameter(torch.randn(n_itm, k))
        self.t_V = self.b_V

        # sigmoid
        # self.sigmoid = nn.Sigmoid()
        self.lnsigmoid = nn.LogSigmoid()

        # hyper parameter
        self.alpha = alpha
        self.gamma = gamma
        self.norm = norm

    def forward(self, x, delta, pmi):
        """
        Input data has to be 1 transaction.
        x[0:n+2*m] is target transaction t,
        x[n+2*m:] is t^m transaction of t
        pmi has to be calculated in preprocessing.
        """
        # transaction
        t = x[:self.n+2*self.m]
        # t^m transaction
        tm = x[self.n+2*self.m:]
        y = self.fm(t)
        # y = self.lnsigmoid(y*delta.float())
        # y *= -1

        # final equation
        # y += self.alpha*pmi*self.constrain(t, tm)

        return y

    def fm(self, x, debug=False):
        # transaction vec whose elements are only 0 or 1.
        x = x.view((1, x.shape[0])).double()

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

        # Target item & basket items relation
        b_vecs = b_vecs.squeeze()
        t_b = torch.mm(t_vec, b_vecs.t()).sum(dim=-1, keepdim=True)
        """
        # for jit compiling
        t_b = torch.mm(t_vec, torch.t(b_vecs[0]))
        for i in range(n_b):
            if i>0:
                # t_b += torch.mm(t_vec, torch.t(b_vecs[i]))
                # b_vec = torch.mm(t_vec, torch.t(b_vecs[i]))
                t_b = torch.addmm(t_b, t_vec, torch.t(b_vecs[i]))
        t_b /= n_b
        """

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
        """
        # for jit compiling
        bs = torch.mm(b_vecs[0], torch.t(b_vecs[1]))
        for i in range(n_b):
            for j in range(i+1, n_b):
                if i!=0 and j!=1:
                    bs += torch.mm(b_vecs[i], torch.t(b_vecs[j]))
        bs /= n_b
        """

        # User & basket items relation
        u_b = torch.mm(u_vec, b_vecs.t()).sum(dim=-1, keepdim=True)
        """
        # for jit compiling
        u_b = torch.mm(u_vec, torch.t(b_vecs[0]))
        for i in range(n_b):
            if i>0:
                u_b += torch.mm(u_vec, torch.t(b_vecs[i]))
        u_b /= n_b
        """

        # Normalize
        if self.norm:
            t_b /= n_b
            bs /= n_b
            u_b /= n_b

        # Output
        y = self.w_0 + \
            bias + \
            self.gamma[0]*u_t + \
            self.gamma[1]*t_b + \
            self.gamma[2]*bs + \
            self.gamma[3]*u_b

        if debug:
            print(f"u_t   : {u_t.item():>8.5f}\n" \
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

    '''
    def constrain(self, t, tm):
        cons = 0.5*(self.fm(t)-self.fm(tm)).pow(2)
        return cons

    def get_vec(self, x, type="u"):
        """
        type will be u(user), i(item)
        """
        pass
    '''

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
        # u_t = (u_vec.view(1, -1) * t_vec).sum(dim=-1).sum(-1)
        u_t = (u_vec * t_vec).sum(-1).sum(-1)

        # Target item & basket items relation
        t_b = (t_vec * b_vecs).sum(-1).sum(-1)

        # Among basket items relation
        bs = None
        for i in range(n_b):
            if bs is None:
                bs = torch.mm(b_vecs[i].view(1,-1), b_vecs[i+1:n_b].t()).sum()
            else:
                bs += torch.mm(b_vecs[i].view(1,-1), b_vecs[i+1:n_b].t()).sum()

        # User & basket items relation
        u_b = torch.mm(u_vec, b_vecs.t()).sum()

        if self.norm:
            t_b /= n_b
            bs /= n_b
            u_b /= n_b

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
    train, test, valid = ds.get_data()

    """
    n: # users
    m: # items
    k: latent vec dim
    """
    n = len(ds.usrset)
    m = len(ds.itemset)
    k = 16

    gamma=[1,1,1,1]
    alpha=0.0
    norm=False

    lr=0.0001
    momentum=0
    weight_decay=0.01

    epochs=21
    neg=0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BFM(n, m, k, gamma, alpha).to(device=device)

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

    # traced = torch.jit.script(model)
    # x = next(train)
    # x, label = x[0], x[1]
    # traced = torch.jit.trace(model, example_inputs=(x, label, torch.tensor([1.])))
    # print(traced.code)

    """
    cnt = 0
    for x in train:
        optimizer.zero_grad()
        x, label = x[0], x[1]
        delta = torch.tensor([[1.]])
        y = model(x, delta=delta, pmi=1)
        if label.item()==-1:
            loss = criterion(y, torch.tensor([[0.]]))
        else:
            loss = criterion(y, label)
        loss.backward()
        optimizer.step()
        if cnt%2500==0:
            print(f"Loss : {loss:.6f} at {cnt:6d}, " \
                  f"Label : {label.item():2.0f},   " \
                  f"# basket item : {x.sum().item()-2:3.0f}")
        cnt+=1
    print(cnt) # => 957264
    os.makedirs("../trained", exist_ok=True)
    torch.save(model.state_dict(), "../trained/BFM_alldelta1.pt")
    """

    # Saved directory
    today = datetime.date.today()
    c_time = datetime.datetime.now().strftime("%H-%M-%S")
    save_dir = f"../trained/bfm/{today}/{c_time}"
    os.makedirs(save_dir, exist_ok=True)
    model_name = "BFM"

    # Load trained parameters
    loaded = False
    if loaded:
        model_path = "../trained/2019-11-08/BFM_4.pt"
        model.load_state_dict(torch.load(model_path))
        epochs = 5


    # Print Information
    print("{:-^60}".format("Data stat"))
    print(f"# User        : {n}\n" \
          f"# Item        : {m}\n" \
          f"Neg sample    : {neg}")
    print("{:-^60}".format("Optim status"))
    print(f"Optimizer     : {optimizer}\n" \
          f"Criterion     : {criterion}\n" \
          f"Learning rate : {lr}\n" \
          f"Momentum      : {momentum}\n" \
          f"Weight decay  : {weight_decay}")
    print("{:-^60}".format("Model/Learning status"))
    print(f"Model name    : {model_name}\n" \
          f"Mid dim       : {k}\n" \
          f"Gamma         : {gamma}\n" \
          f"Alpha         : {alpha}\n" \
          f"Normalize     : {norm}\n" \
          f"Epochs        : {epochs}\n" \
          f"Loaded        : {loaded}")
    if loaded:
        print(f"Learned model : {model_path}")
    print("{:-^60}".format("Description"))
    print("Changed random seed to get train data\n"\
         "Use double type for all layers.")
    print("{:-^60}".format("Path"))
    print(f"{save_dir}")
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
            x, label = x[0].to(device), x[1].to(device)
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
        # torch.save(model.state_dict(), f"{save_dir}/{model_name}_{e}.pt")
        # Better way to save model?
        state = {"name": model_name, "epoch": e, "state_dict": model.state_dict(),
                 "neg":neg, "optimizer": optimizer.state_dict(),
                 "k": k, "gamma": gamma, "alpha": alpha, "norm": norm}
        torch.save(state, f"{save_dir}/{model_name}_{e}.pt")

        print("{:-^60}".format("end"))

if __name__=="__main__":
    main()
