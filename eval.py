# from dis import dis
# from pprint import pprint
import os
import argparse
import sys
# import time
import random
import yaml
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

from robustbench.utils import load_model, clean_accuracy
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2

from tqdm import tqdm

from autoattack import AutoAttack

import matplotlib.pyplot as plt
import pandas as pd

def parse_args(args: list) -> argparse.Namespace:
    """Parse command line parameters.

    :param args: command line parameters as list of strings (for example
        ``["--help"]``).
    :return: command line parameters namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the models for this experiment."
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dataset-path",
        default="/home-local2/jongn2.extra.nobkp/data",
        help="the path to the dataset",
        type=str,
    )
    parser.add_argument(
        "--cpus-per-trial",
        default=1,
        help="the number of CPU cores to use per trial",
        type=int,
    )
    parser.add_argument(
        "--project-name",
        help="the name of the Weights and Biases project to save the results",
        # required=True,
        type=str,
    )
    parser.add_argument(
        "--dataset",
        help="dataset used",
        required=False,
        type=str,
    )
    parser.add_argument(
        "--debug-strategy",
        help="the strategy to use in debug mode",
        default="Random",
        type=str,
    )

    return parser.parse_args(args)


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

def run_trial(
    config: dict, params: dict, args: argparse.Namespace, num_gpus: int = 0
) -> None:
    """Train a single model according to the configuration provided.

    :param config: The trial and model configuration.
    :param params: The hyperparameters.
    :param args: The program arguments.
    """

    #
    norm_thread = params['norm_thread']
    model_name = params['model_name']
    root= params['results_root_path']
    resultsDirName = f'{root}/{model_name}_{norm_thread}'
    if not os.path.exists(resultsDirName):
        os.makedirs(resultsDirName)
        print("Results directory ", resultsDirName,  " Created ")
 
    set_seeds(params['seed'])
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using GPU: {use_cuda}')

    """ MODEL """
    #Load Model
    model = load_model(model_name=params['model_name'], dataset=params['dataset_name'], threat_model=params['norm_thread'])
    model = model.to(device)
    model.eval()
    print("Model Loaded")
    print(model)

    """# Dataset"""

    #@title cifar10
    # Normalize the images by the imagenet mean/std since the nets are pretrained
    transform = transforms.Compose([transforms.ToTensor(),])
        #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)

    # valid_size = 5000
    # train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-valid_size, valid_size])

    # # train_loader = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'],
    # #                                         shuffle=True, num_workers=2)

    # val_loader = torch.utils.data.DataLoader(val_set, batch_size=params['batch_size'],
    #                                         shuffle=True, num_workers=1)


    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    # batch=1
    # indices=range(len(test_set))
    # batches= [indices[:5000],indices[5000:]]
    # test_set = torch.utils.data.dataset.Subset(test_set,batches[batch])

    adv0=torch.load("results/Wang2023Better_WRN-28-10_L2/adverserial0.pt")
    adv1=torch.load("results/Wang2023Better_WRN-28-10_L2/adverserial1.pt")
    adv=torch.cat([adv0,adv1])

    original= torch.stack([transform(img) for img in test_set.data])
    targets = torch.Tensor(test_set.targets)
    print(original.shape, adv.shape)
    my_dataset = torch.utils.data.dataset.TensorDataset(original, adv, targets)
    test_loader = torch.utils.data.DataLoader(my_dataset, batch_size=params['batch_size'],
                                            shuffle=False, num_workers=1)

    acc, adv_acc, adv_failure, df  = eval(model, test_loader, device)

    df["(Acc,Adv Acc, Adv)"]=df[["Acc", "Adv Acc","Adv"]].apply(tuple, axis=1)
    df.to_csv(os.path.join(resultsDirName,f'results.csv'))
    import plotly.express as px

    fig=px.scatter(df,x="Entropy", y="CW norm", color="(Acc,Adv Acc, Adv)")
    fig.write_html(os.path.join(resultsDirName,f'CW_vs_ENtropy.html'))

    # sns.scatter(entropies.numpy(),norms.numpy(), alpha=.7)
    # plt.xlabel("CW norms")
    # plt.ylabel("Entropy")

    # plt.savefig(os.path.join(resultsDirName,f'CW_vs_ENtropy.png'), dpi=600)
    df_adv=df[df["Adv"]==False]
    spearman=df_adv[["Entropy","CW norm"]].corr("spearman")
    print(spearman)
    print(spearman.iloc[0,1])
    print(f'adv acc: {100*adv_acc:.2f}%')
    with open(os.path.join(resultsDirName,f'result_all.txt'), "w") as wf:
        wf.write(f"Accuracy: {acc} \n")
        wf.write(f"Adverserial accuracy: {adv_acc}\n")
        wf.write(f"Adverserial failure: {adv_failure}\n")
        wf.write(f"Spearman correlation Entropy vs CW norm (adverserial only): {spearman.iloc[0,1]}\n")
        wf.close()

def eval(model, loader, device):
    correct=0
    correct_adv=0
    constant=0
    # hs=[]
    # norms=[]
    total=0
    df_list=[]
    for images, adv_images, target in tqdm(loader):
        images, adv_images, target = images.to(device), adv_images.to(device), target.to(device)
        with torch.no_grad():
            out = model(images)
            out_adv = model(adv_images)
            entropy=torch.special.entr(torch.softmax(out,1)).sum(1)
            norm = torch.linalg.vector_norm(images.flatten(1) - adv_images.flatten(1),ord=2,dim=1)
            # print(entropy.shape)
            # print(norm.shape)
            # hs.append(entropy.cpu())
            # norms.append(norm.cpu())
            _, y = torch.max(out.data, 1)
            _, y_adv = torch.max(out_adv.data, 1)

            correct_adv += sum(y_adv == target ).item()
            constant += sum(y_adv == y ).item()
            correct += sum(y == target ).item()
            total += len(y_adv) 
            acc =(y == target).cpu().numpy()
            adv_acc =(y_adv == target).cpu().numpy()
            adv = (y != y).cpu().numpy()
            df_list+=zip(entropy.cpu().numpy(), norm.cpu().numpy(), acc, adv_acc ,adv )
    df=pd.DataFrame(df_list, columns=["Entropy", "CW norm", "Acc", "Adv Acc", "Adv"])
    return correct/total, correct_adv/total, constant/total, df

def run_experiment(params: dict, args: argparse.Namespace) -> None:
    """Run the experiment using Ray Tune.

    :param params: The hyperparameters.
    :param args: The program arguments.
    """
    config = {}

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpus_per_trial = 1 if use_cuda else 0

    run_trial(config=config, params=params, args=args, num_gpus=gpus_per_trial)



def main(args: list) -> None:
    """Parse command line args, load training params, and initiate training.

    :param args: command line parameters as list of strings.
    """
    args = parse_args(args)
    paramsfilename = f'./params.yaml'
    with open(paramsfilename, 'r') as param_file:
        params = yaml.load(param_file, Loader=yaml.SafeLoader)
    run_experiment(params, args)


def run() -> None:
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`
    This function can be used as entry point to create console scripts.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
