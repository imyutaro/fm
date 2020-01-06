import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# for reproducibility
def seed_everything(seed=1234):
    # np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Load func is like below
def load_model(filename, device, model_name="FABFM"):
    if os.path.isfile(filename):
        from dataloader import Data
        # Load file
        checkpoint = torch.load(filename)

        # Load dataset setting
        neg = checkpoint["neg"]
        ds = Data(root_dir="./data/ta_feng/")
        train, test, _= ds.get_data(neg=neg, test_neg=True)
        n_usr = len(ds.usrset)
        n_itm = len(ds.itemset)

        # Load network
        # model_name = checkpoint["name"]
        optimizer = checkpoint["optimizer"]
        k = checkpoint["k"]
        gamma = checkpoint["gamma"]
        alpha = checkpoint["alpha"]
        norm = checkpoint["norm"]
        if model_name=="FABFM":
            d = checkpoint["d"]
            h = checkpoint["h"]
            from models.fixed_abfm import FABFM
            # model = FABFM(n_usr, n_itm, k, d, h, gamma, alpha).to(device=device)
            model = FABFM(n_usr, n_itm, k, d, h, gamma, alpha, norm=norm).to(device=device)
        elif model_name=="ABFM":
            from models.abfm import ABFM
            model = ABFM(n_usr, n_itm, k, gamma, alpha).to(device=device)
        elif model_name=="BFM":
            from models.bfm import BFM
            model = BFM(n_usr, n_itm, k, gamma, alpha).to(device=device)
        model.load_state_dict(checkpoint["state_dict"])

        print("{:-^60}".format("Data stat"))
        print(f"# User        : {n_usr}\n" \
              f"# Item        : {n_itm}\n" \
              f"Neg sample    : {neg}")
        print("{:-^60}".format("Optim status"))
        # print(f"lr            : {optimizer['param_groups'][0]['lr']}\n"\
        #       f"Momentum      : {optimizer['param_groups'][0]['momentum']}\n"\
        #       f"Dampening     : {optimizer['param_groups'][0]['dampening']}\n"\
        #       f"Weight_decay  : {optimizer['param_groups'][0]['weight_decay']}\n"\
        #       f"Nesterov      : {optimizer['param_groups'][0]['nesterov']}")
        print("{:-^60}".format("Model/Learning status"))
        print(f"Mid dim       : {k}\n" \
              f"Gamma         : {gamma}\n" \
              f"Alpha         : {alpha}")
        print("{:-^60}".format(""), flush=True)

        return model, train, test, ds.n_train, ds.n_test, n_usr, n_itm
    else:
        print(f"There is not {filename}")
        return None

def get_name(path):
    import os
    return os.path.splitext(os.path.basename(path))[0].split("_")[0]

def main():
    import csv
    import time
    from scipy.stats import rankdata

    from models.bfm import BFM
    from models.abfm import ABFM
    from models.fixed_abfm import FABFM
    from dataloader import Data

    # ds = Data(root_dir="./data/ta_feng/")
    # # train, test, valid = ds.get_data(neg=0)
    # train, test, valid = ds.get_data(neg=2)

    # # Load network
    # n_usr = len(ds.usrset)
    # n_itm = len(ds.itemset)
    # k = 32
    # d=k
    # h=2
    # gamma=[1,1,1,1]
    # alpha=0.0

    # lr=0.0001
    # momentum=0
    # weight_decay=0.01

    # epochs=21
    # neg=2

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # model = ABFM(n_usr, n_itm, k).to(device=device)
    # model = FABFM(n_usr, n_itm, k, d, h, gamma, alpha).to(device=device)

    with open("./path_for_metric") as f:
        paths = [row[0] for row in csv.reader(f, delimiter="\n")]
    paths = paths[19:]
    # paths = paths[42:]
    # paths = paths[69:70]


    cnt = 0
    for i, path in enumerate(paths):
        print(f"{path:-^60}")
        name = get_name(path)
        model, train, test, l_train, l_test, n_usr, n_itm = load_model(path, device, model_name=name)

        # choice= "train"
        choice= "test"
        metric=False
        if metric:
            from metric import evaluate

            if choice=="train":
                result, r_result, diversity, auc = evaluate(train, l_train, model, n_usr, n_itm, \
                                                        device, C=100, beta=5, n_rank=10, fAUC=True)
                print(f"Data      : {choice}\n" \
                      f"HLU       : {result}\n" \
                      f"R@10      : {r_result}\n" \
                      f"Diversity : {diversity}\n" \
                      f"AUC       : {auc}\n" \
                       "{:-^60}".format(""), flush=True)
            elif choice=="test":
                result, r_result, diversity, auc = evaluate(test, l_test, model, n_usr, n_itm, \
                                                       device, C=100, beta=5, n_rank=10, fAUC=True)
                print(f"Data      : {choice}\n" \
                      f"HLU       : {result}\n" \
                      f"R@10      : {r_result}\n" \
                      f"Diversity : {diversity}\n" \
                      f"AUC       : {auc}\n" \
                       "{:-^60}".format(""), flush=True)
        else:
            criterion = nn.BCEWithLogitsLoss()
            if choice=="train":
                data = train
            elif choice=="test":
                data = test
            cnt = 0
            for i in data:
                if cnt==50: break
                with torch.no_grad():
                    x, label = i[0].to(device), i[1].to(device)
                    print(f"\nLabel : {label.item()}")
                    # print(f"fm output : {model.fm(x, debug=True).item():>8.5f}")
                    print(f"fm output : {model.fm(x, debug=True).item()}")
                    print(f"sig out   : {torch.sigmoid(model(x, label, pmi=1)).item()}")
                    if label==-1:
                        # print(f"Loss      : {criterion(torch.sigmoid(model(x, label, pmi=1)), label+1).item():>8.5f}")
                        print(f"Loss      : {criterion(torch.sigmoid(model(x, label, pmi=1)), label+1).item()}")
                    else:
                        # print(f"Loss      : {criterion(torch.sigmoid(model(x, label, pmi=1)), label).item():>8.5f}")
                        print(f"Loss      : {criterion(torch.sigmoid(model(x, label, pmi=1)), label).item()}")
                    # target_idx = (x[n:n+m]==1).nonzero()
                    # y = model.rank_list(x).cpu().numpy()
                    # rank = rankdata(-y, method="min")[target_idx]
                    # print("Rank      :", rank)
                cnt+=1

if __name__=="__main__":
    seed = 1234
    seed_everything(seed)
    main()
