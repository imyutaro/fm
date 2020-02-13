import numpy as np
from tqdm import tqdm
import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F

def sigmoid(x):
    sigmoid_range = 34.538776394910684

    if x <= -sigmoid_range:
        return 1e-15
    if x >= sigmoid_range:
        return 1.0 - 1e-15

    return 1/(1+np.exp(-x))

def predict(x, idx, model):
    x[idx] = 1
    with torch.no_grad():
        y = model.fm(x)
    return y.item()

def wrapper_func(args):
    return predict(x=args[0], idx=args[1], model=args[2])

def multipro_hlu(test, n_test, model, n_usr, m, C=100, beta=5):
    """
    target itemでfor文を回してscoreが高い順に並べる必要がある.
    one-hot ベースでやる(x[n:n+m]ベースで)
    x[:n]        : user
    x[n:n+m]     : target item
    x[n+m:n+2*m] : basket items

    Variables -----
    test    : Test data
    n_test  : The number of test data
    model   : Trained model
    itemset : A set of items

    C       : Scaling parameter
    beta    : half-life parameter (Idk what that is...)
    ---------------
    """

    beta -= 1
    rank_sum = 0
    with torch.no_grad():
        for x in tqdm(test, total=n_test):
            x, _ = x[0], x[1]
            target_idx = (x[n_usr:n_usr+n_itm]==1).nonzero()+n_usr
            x[n_usr:n_usr+n_itm] = 0

            # multi-processing
            args = []
            for idx in range(n_usr, n_usr+n_itm):
                args.append((x, idx, model))
            processes = max(1, multiprocessing.cpu_count()-1)
            p = multiprocessing.Pool(processes)
            result_multi = p.imap(wrapper_func, args)
            rank_dict = {arg[1]: result for arg, result in zip(args, result_multi)}
            p.close()
            p.terminate()

            # Sort
            sorted_rank = sorted(rank_dict.items(), key=lambda x:x[1])
            # rank
            rank = [i for i, v in enumerate(sorted_rank) if v[0]==target_idx][0]

            # Summation
            rank_sum += 2**((1-rank)/beta)

    hlu = (C*rank_sum)/n_test
    return hlu

def evaluate(test, n_test, model, n_usr, n_itm, device, C=100, beta=5, n_rank=10, fAUC=False):
    from sklearn import metrics
    from scipy.stats import rankdata
    from sklearn.preprocessing import minmax_scale

    beta -= 1
    rank_sum = 0
    cnt = 0
    r_at_n = 0
    diversity = 0
    prd = []
    ans = []
    loss = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        # for x in tqdm(test, total=n_test):
        for x in test:
            x, label = x[0].to(device).double(), x[1].to(device).double()

            if label==-1:
                loss += criterion(model(x, label, pmi=1), label+1).item()
            else:
                loss += criterion(model(x, label, pmi=1), label).item()

            target_idx = (x[n_usr:n_usr+n_itm]==1).nonzero()
            # Predict
            y = model.rank_list(x).view(-1).cpu().numpy()

            """debug
            cnt += 1
            print(y[target_idx])
            # _ = [print(i) for i in y if i>y[target_idx]]
            num=0
            for i in y:
                if i==y[target_idx]:
                    num+=1
            print(num)
            if cnt == 10:
                exit()
            """

            if fAUC:
                # AUC can't be calculated if data has only positive data
                # prd.append(sigmoid(y[0][target_idx]))
                prd.append(sigmoid(y[target_idx]))
                ans.append(label.item())

            if label==1:
                # If label is pos, the greater value of y is better.
                # If label is neg, the lower value of y is better.

                # Normalize 0 to 1000
                # rank = minmax_scale(y[0], feature_range=(0,1000))
                rank = rankdata(y, method="min")
                diversity += len(set(rank))
                rank = rank[target_idx]

                # Summation for HLU
                rank_sum += 2**((1-rank)/beta)

                # R@N
                if rank<n_rank+1:
                    r_at_n+=1
                cnt+=1

    hlu = (C*rank_sum)/cnt
    r_at_n/=cnt
    diversity/=cnt
    loss/=n_test
    print(f"Loss      : {loss}")

    if fAUC:
        fpr, tpr, thresholds = metrics.roc_curve(ans, prd, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        result = (hlu, r_at_n, diversity, auc)
    else:
        result = (hlu, r_at_n, diversity)

    return result


def main():
    import csv

    from dataloader import Data
    from models.bfm import BFM
    from models.abfm import ABFM

    # model paths
    with open("./path") as f:
         paths = [row[0] for row in csv.reader(f, delimiter="\n")]
    # paths = [paths[15]]

    ds = Data(root_dir="./data/ta_feng/")
    itemset = ds.itemset

    #load network
    n_usr = len(ds.usrset)
    n_itm = len(ds.itemset)
    k = 32

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ABFM(n_usr, n_itm, k).to(device=device)
    # model = BFM(n_usr, n_itm, k).to(device=device)


    for path in paths:
        # reset test dataset
        _ , test, _ = ds.get_data()
        n_test = ds.n_test
        print(f"{path:-^60}")
        model.load_state_dict(torch.load(path))

        result, r_result, diversity = evaluate(test, n_test, model, n_usr, n_itm, device)
        # result = r_at_n(test, n_test, model, n, m)
        print(f"HLU       : {result}")
        print(f"R@10      : {r_result}")
        print(f"Diversity : {diversity}")
        print("{:-^60}".format(""))

if __name__=="__main__":
    main()
