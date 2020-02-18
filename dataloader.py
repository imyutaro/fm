import os
import random
import pickle
import torch
import torch.nn.functional as F

# for reproducibility
seed = 1234
def seed_everything(seed=1234):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
seed_everything(seed)

class Data():
    def __init__(self, filename="transaction.dat", root_dir=None):
        if not root_dir:
            from hydra import utils
            self.root_dir = os.path.dirname(utils.get_original_cwd())
            self.root_dir = os.path.join(self.root_dir, "data/ta_feng/")
        else:
            self.root_dir = root_dir

        f = open(self.root_dir+filename, "rb")
        data = pickle.load(f)

        self.usrset, self.itemset = self._uiset(data)

        self.o_usr = self._usr2vec(self.usrset)
        self.o_item = self._item2vec(self.itemset)

        self.n_neg = {}

    def _uiset(self, data):
        """Make user and item set from data"""
        usrset = set()
        itemset = set()
        for d in data:
            usrset |= {d[0]}
            itemset |= d[1]
        return list(usrset), list(itemset)

    def _usr2vec(self, usrset):
        """Convert user list to one-hot vector"""
        usrset.sort()
        o_usr = F.one_hot(torch.arange(0, len(usrset))).float()
        return o_usr

    def _item2vec(self, itemset):
        """Convert item list to one-hot vector"""
        itemset.sort()
        o_item = F.one_hot(torch.arange(0, len(itemset))).float()
        return o_item

    def _train(self):
        f = open(self.root_dir+"train.pkl","rb")
        train = pickle.load(f)

        if type(train) is dict:
            train = [(u, list(t), 1) for u, ts in train.items() for t in ts]
        self.n_train = len(train)

        return train

    def _neg_train(self, neg):

        assert neg==1 or neg==2, "neg option has to be 1 or 2"
        if neg==2:
            f = open(self.root_dir+"negative_sample.pkl","rb")
        elif neg==1:
            f = open(self.root_dir+"negative_sample_few.pkl","rb")

        neg_train = pickle.load(f)
        if type(neg_train) is dict:
            neg_train = [(u, list(t), -1) for u, ts in neg_train.items() for t in ts]
        self.n_neg["train"] = len(neg_train)

        return neg_train

    def _neg_load(self, src, neg):

        assert neg==1 or neg==2, "neg option has to be 1 or 2"
        if neg==2:
            f = open(self.root_dir+f"/neg/neg_{src}_2.pkl","rb")
        elif neg==1:
            f = open(self.root_dir+f"/neg/neg_{src}_1.pkl","rb")

        neg_data = pickle.load(f)
        if type(neg_data) is dict:
            neg_data = [(u, list(t), -1) for u, ts in neg_data.items() for t in ts]
        self.n_neg[f"{src}"] = len(neg_data)

        return neg_data

    def _test(self):
        f = open(self.root_dir+"test.pkl","rb")
        test = pickle.load(f)

        assert type(test) is dict, "pickle data must be dict"

        # test = [(u, list(t), 1) for u, ts in test.items() for t in ts]
        test_list = []
        self.n_test = 0

        for u, ts in test.items():
            for t in ts:
                test_list.append((u, list(t), 1))
                self.n_test += len(t)

        return test_list

    def _valid(self):
        f = open(self.root_dir+"validation.pkl","rb")
        valid = pickle.load(f)

        if type(valid) is dict:
            valid = [(u, list(t), 1) for u, ts in valid.items() for t in ts]
        self.n_valid = len(valid)

        return valid

    def _convert(self, base):

        for d in base:
            usr = d[0]
            trans = d[1]
            label = torch.tensor([[d[2]]], dtype=torch.float32)

            usr = self.o_usr[self.usrset.index(usr)]

            basket = 0
            for item in trans:
                basket += self.o_item[self.itemset.index(item)]

            # print(trans)
            # print("items:", len(trans), basket.sum())
            # print("general basket\n", (basket==1).nonzero())
            for item in trans:
                # print(item)
                target = self.o_item[self.itemset.index(item)]
                t = basket-target
                # print("target \n", (target==1).nonzero())
                # print("t nonzero\n", (t==1).nonzero())
                t = torch.cat((target, t))
                # print(target.sum(), t[:target.shape[0]].sum(), t[target.shape[0]:].sum())
                t = torch.cat((usr, t))
                yield (t, label)

    def get_data(self, neg=2, test_neg=False, seed=1234):
        """neg : you can set how larger negative sample you use"""
        train = self._train()
        if neg:
            neg_train = self._neg_train(neg)
            train += neg_train

        # Shuffle train data
        for _ in range(22):
            train = random.Random(seed).sample(train, len(train))
        train = self._convert(train)

        test = self._test()
        if test_neg:
            neg_test = self._neg_load(src="test", neg=neg)
            test += neg_test
        test = self._convert(test)

        valid = self._convert(self._valid())

        return train, test, valid

def main():
    import time

    ds = Data(root_dir="./data/ta_feng/")
    train, test, valid = ds.get_data()

    # t1 = next(test)
    # index = (t1[0]==1).nonzero()
    # print(index)

    # tmp = t1[0]
    # print(tmp)

    # basket_items = index[2:]
    # basket_items = set(basket_items.view(-1).tolist())
    # print(basket_items)

    # tmp[index[2:]] = 0
    # # index = (t1[0]==1).nonzero()
    # # print(index)
    # # print(type(index))

    # new_idx = index[2:]
    # print(new_idx+200)
    # tmp[new_idx] = 1

    #load network
    from models import bfm
    n = len(ds.usrset)
    m = len(ds.itemset)
    k = 32
    model = bfm.BFM(n, m, k)
    # choose 0--11
    path = ["./trained/bfm/2019-11-08/BFM_2.pt", \
            "./trained/bfm/2019-11-12/BFM_17.pt", \
            "./trained/bfm/2019-11-14/BFM_no_l2_2.pt", \
            "./trained/bfm/2019-11-14/BFM_no_l2_4.pt", \
            "./trained/bfm/2019-11-15/BFM_norm_2.pt", \
            "./trained/bfm/2019-11-15/BFM_norm_4.pt"]

    model_path = path[7]
    # model_path = path[6]
    print(f"{model_path:-^60}")
    model.load_state_dict(torch.load(model_path))
    # print("output :", model(t1[0], t1[1], pmi=1))

    cnt = 0
    from scipy.stats import rankdata
    for i in test:
        if cnt==50: break
        with torch.no_grad():
            print(f"\nLabel     : {i[1]}")
            print("fm output :", model.fm(i[0]))
            print("Loss      :", model(i[0], i[1], pmi=1))
            x, _ = i[0], i[1]
            target_idx = (x[n:n+m]==1).nonzero()
            y = model.rank_list(x).numpy()
            rank = rankdata(-y, method="min")[target_idx]
            print("Rank      :", rank)
        cnt+=1

if __name__=="__main__":
    main()
