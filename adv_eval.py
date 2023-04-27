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

from robustbench.utils import load_model
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent as pgd
from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from art.estimators.classification.pytorch import PyTorchClassifier
from art.metrics.metrics import clever_u

from tqdm import tqdm

from autoattack import AutoAttack
from ray import tune
from ray.tune import CLIReporter

from pretrained.resnet import resnet18, resnet50


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
        default='evalrobustness',
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

    """ MODEL """
    #Load Model
    if model_name.lower() == 'resnet18':
        model = resnet18(pretrained=True)
    elif model_name.lower() == 'resnet50':
        model = resnet50(pretrained=True)
    else:
        model = load_model(model_name=params['model_name'], dataset=params['dataset_name'], threat_model=params['norm_thread'])
    model = model.to(device)
    model.eval()
    print("Model Loaded")
    """# Dataset"""

    #@title cifar10
    # Normalize the images by the imagenet mean/std since the nets are pretrained
    if model_name.lower().startswith('resnet'):
         data_normalize = transforms.Normalize(mean = [0.4914, 0.4822, 0.4465], std = [0.2471, 0.2435, 0.2616])
         transform = transforms.Compose([transforms.ToTensor(), data_normalize])
         zeros = torch.zeros((3,32,32))
         ones = torch.ones_like(zeros)
         minpixel = data_normalize(zeros).min().item()
         maxpixel = data_normalize(ones).max().item() 
    else:
        transform = transforms.Compose([transforms.ToTensor(),])
        minpixel = 0.
        maxpixel = 1.

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
    save_tag = ''
    if config.get('batch_id'):
        batch_id=config['batch_id']
        id_to_run=params['id_to_run']
        size = len(test_set)//params['n_batches'] + 1 # math.ceil
        indices=np.arange(len(test_set))
        if id_to_run < 0: #run all mode
            batch_indices = indices[batch_id*size:(batch_id+1)*size]
            save_tag = f'{batch_id}'
        else:
            size1 = size//params['n_batches'] + 1
            batch_indices = indices[:size][batch_id*size1:(batch_id+1)*size1]
            save_tag = f'{id_to_run}_{batch_id}'
        test_set = torch.utils.data.dataset.Subset(test_set,batch_indices)
        print(f'batch {batch_id} from {batch_indices[0]} to {batch_indices[-1]}...')


    test_loader = torch.utils.data.DataLoader(test_set, batch_size=params['batch_size'],
                                            shuffle=False, num_workers=1)




    ## 
    if params['attack'] =='cw':
        adv_acc, adversarial = cw_attack(model, test_loader, device,minpixel=minpixel, maxpixel=maxpixel)
        torch.save(adversarial, os.path.join(resultsDirName,f'adverserial{save_tag}.pt'))

    elif params['attack'] =='fab':
        adv_acc, adversarial, y_adversarial = fab_attack(model, test_loader, norm_thread, device)
        torch.save(adversarial, os.path.join(resultsDirName,f'fab_adverserial{save_tag}.pt'))
        torch.save(adversarial, os.path.join(resultsDirName,f'fab_y_adverserial{save_tag}.pt'))
    elif params['attack'] == 'clever':
        clever_scores =   clever_scores(model, test_loader,  norm_thread, device, minpixel=minpixel, maxpixel=maxpixel)
        torch.save(clever_scores, os.path.join(resultsDirName,f'clever_scores{save_tag}.pt'))
    else:
        # adv_acc = autoattack(model, test_loader, params['norm_thread'], device)
        raise NotImplementedError


    print(f'adv acc: {100*adv_acc:.2f}%')
    with open(os.path.join(resultsDirName,f'result{save_tag}.txt'), "w") as f:
        f.write(f"Adverserial accuracy (batch {save_tag}): {adv_acc}")
        f.close()

def cw_attack(model, test_loader, device, minpixel=0., maxpixel=1.0):
    correct = 0
    total = 0
    adv_list = []    
    for images, labels in tqdm(test_loader):
        images, labels = images.to(device), labels.to(device)
        x_adv = carlini_wagner_l2(model, images, n_classes=10, targeted=False,clip_min=minpixel, clip_max=maxpixel).detach()
        adv_list.append(x_adv.cpu())
        with torch.no_grad():
            out_adv = model(x_adv)
            _, y_adv = torch.max(out_adv.data, 1)
            correct += sum(y_adv == labels).item()
            total += len(y_adv)
    adversarial=torch.cat(adv_list)
    return correct / total, adversarial

def fab_attack(model, test_loader, norm_thread, device):
    x_test, y_test = [], []
    i = 0
    adv_list = []    
    for x, y in test_loader:
        if i == 0:
            x_test = x
            y_test = y
            i += 1
        else:
            x_test = torch.cat((x_test, x), 0)
            y_test = torch.cat((y_test, y), 0)
    #
    x_test, y_test = x_test.to(device), y_test.to(device)
    adversary = AutoAttack(model, norm=norm_thread, eps=0.5, version='custom', attacks_to_run=['fab'])
    x_adv, y_adv = adversary.run_standard_evaluation(x_test, y_test, return_labels=True)
    acc = (y_test == y_adv).sum().item() / len(y_test)
    return acc, x_adv, y_adv

def clever_scores(model, test_loader,  norm_thread, device, minpixel=0., maxpixel=1.0):
    clever_args={'min_pixel_value':  minpixel,
                'max_pixel_value': maxpixel,
                'nb_batches':100,
                'batch_size':100,
                'norm':float(norm_thread[1:]), # eg: 'L2'[1:]=2, 'Linf'[1:]=inf
                'radius':10.,
                'pool_factor':10}
    cl_scores = []
    model.eval()
    for data in tqdm(test_loader):
        clever_dis = clever_score_u(model, data[0][0], **clever_args)
        cl_scores.append(clever_dis)
    return np.array(cl_scores)

def clever_score_u(model, x, **args):
    # set_seeds(1884)
    classifier = PyTorchClassifier(
    model=model,
    clip_values=(args['min_pixel_value'], args['max_pixel_value']),
    loss=None,
    optimizer=None,
    input_shape=(1, 32, 32),
    nb_classes=10,
    )
    res = clever_u(classifier, x.numpy(), 
                    nb_batches=args['nb_batches'], 
                    batch_size=args['batch_size'], 
                    radius=args['radius'], 
                    norm=args['norm'], 
                    pool_factor=args['pool_factor'], verbose=False)
    return res

# def autoattack(model, test_loader, norm_thread, device):
#     x_test, y_test = [], []
#     i = 0
#     for x, y in test_loader:
#         if i == 0:
#             x_test = x
#             y_test = y
#             i += 1
#         else:
#             x_test = torch.cat((x_test, x), 0)
#             y_test = torch.cat((y_test, y), 0)
#     #
#     adversary = AutoAttack(model, norm=norm_thread, eps=8/255, version='standard')
    # x_adv, y_adv = adversary.run_standard_evaluation(x_test.to(device), y_test.to(device), return_labels=True)
    # return clean_accuracy(y_test, y_adv)


def run_experiment(params: dict, args: argparse.Namespace) -> None:
    """Run the experiment using Ray Tune.

    :param params: The hyperparameters.
    :param args: The program arguments.
    """
    config = {}

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpus_per_trial = 1 if use_cuda else 0
    if not params.get('n_batches'):
       run_trial(config=config, params=params, args=args, num_gpus=gpus_per_trial)
    else:
        config = {
            "batch_id": tune.grid_search(list(np.arange(params['n_batches']))),
            "model_name": params['model_name']
        }
        reporter = CLIReporter(
            parameter_columns=["model_name", "batch_id"],
            # metric_columns=["round"],
        )
        tune.run(
            tune.with_parameters(
                run_trial, params=params, args=args, num_gpus=gpus_per_trial
            ),
            resources_per_trial={
                "cpu": args.cpus_per_trial, "gpu": gpus_per_trial},
            config=config,
            progress_reporter=reporter,
            name=args.project_name,
        )

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
