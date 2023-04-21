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

def run_trial_empty(
    config: dict, params: dict, args: argparse.Namespace, num_gpus: int = 0
) -> None:
    print("DO NOTHING AND EXIT")


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

    """# Dataset"""

    #@title cifar10
    # Normalize the images by the imagenet mean/std since the nets are pretrained
    transform = transforms.Compose([transforms.ToTensor(),])
        #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    valid_size = 5000
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset)-valid_size, valid_size])

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=params['batch_size'],
    #                                         shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=params['batch_size'],
                                            shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=params['batch_size'],
                                            shuffle=False, num_workers=2)

    # x_test, y_test = load_cifar10(n_examples=10000)

    #Load Model
    model = load_model(model_name=params['model_name'], dataset=params['dataset_name'], threat_model=params['norm_thread'])
    model = model.to(device)
    model.eval()

    ## 
    if params['attack'] =='cw':
        adv_acc = cw_attack(model, test_loader, device)
    else:
        adv_acc = autoattack(model, test_loader, params['norm_thread'], device)

    print(f'adv acc: {100*adv_acc:.2f}%')
    with open(resultsDirName+'/result.txt') as f:
        f.write(adv_acc)
        f.close()

def cw_attack(model, test_loader, device):
    correct = 0
    total = 0
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        x_adv = carlini_wagner_l2(model, images, n_classes=10, targeted=False).detach()
        with torch.no_grad():
            out_adv = model(x_adv)
            _, y_adv = torch.max(out_adv.data, 1)
            correct += sum(y_adv == labels).item()
            total += len(y_adv)
    return correct / total


def autoattack(model, test_loader, norm_thread, device):
    x_test, y_test = [], []
    i = 0
    for x, y in test_loader:
        if i == 0:
            x_test = x
            y_test = y
            i += 1
        else:
            x_test = torch.cat((x_test, x), 0)
            y_test = torch.cat((y_test, y), 0)
    #
    adversary = AutoAttack(model, norm=norm_thread, eps=8/255, version='standard')
    x_adv, y_adv = adversary.run_standard_evaluation(x_test.to(device), y_test.to(device), return_labels=True)
    return clean_accuracy(y_test, y_adv)


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
