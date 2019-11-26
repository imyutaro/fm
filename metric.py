import torch
import numpy as np
from tqdm import tqdm
import multiprocessing

def predict(x, idx, model):
    x[idx] = 1
    with torch.no_grad():
        y = model.fm(x)
    return y.item()

def wrapper_func(args):
    return predict(x=args[0], idx=args[1], model=args[2])

def multipro_hlu(test, n_test, model, n, m, C=100, beta=5):
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
            target_idx = (x[n:n+m]==1).nonzero()+n
            x[n:n+m] = 0

            # multi-processing
            args = []
            for idx in range(n, n+m):
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

def hlu(test, n_test, model, n, m, device, C=100, beta=5):
    from scipy.stats import rankdata

    beta -= 1
    rank_sum = 0
    cnt = 0
    r_result = 0
    with torch.no_grad():
        # for x in tqdm(test, total=n_test):
        for x in test:
            x, _ = x[0].to(device), x[1].to(device)
            target_idx = (x[n:n+m]==1).nonzero()
            # Predict
            y = model.rank_list(x).cpu().numpy()

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

            rank = rankdata(-y, method="min")[target_idx]

            # Summation
            rank_sum += 2**((1-rank)/beta)

            if rank<11:
                r_result+=1

    result = (C*rank_sum)/n_test
    r_result/=n_test

    return result, r_result

def r_at_n(test, n_test, model, n, m, rank=10):
    from scipy.stats import rankdata

    result = 0.0
    with torch.no_grad():
        # for x in tqdm(test, total=n_test):
        for x in test:
            x, _ = x[0], x[1]
            target_idx = (x[n:n+m]==1).nonzero()
            # Predict
            y = model.rank_list(x).numpy()

            rank = rankdata(-y, method="min")[target_idx]
            if rank<11:
                result+=1

    result/=n_test

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
    n = len(ds.usrset)
    m = len(ds.itemset)
    k = 32

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ABFM(n, m, k).to(device=device)
    # model = BFM(n, m, k).to(device=device)


    for path in paths:
        # reset test dataset
        _ , test, _ = ds.get_data()
        n_test = ds.n_test
        print(f"{path:-^60}")
        model.load_state_dict(torch.load(path))

        result, r_result = hlu(test, n_test, model, n, m, device)
        # result = r_at_n(test, n_test, model, n, m)
        print(f"HLU  : {result}")
        print(f"R@10 : {r_result}")
        print("{:-^60}".format(""))

if __name__=="__main__":
    main()
