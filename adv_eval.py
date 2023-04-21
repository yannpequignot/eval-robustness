# from dis import dis
# from pprint import pprint
import os
import argparse
import sys
# import time
from ray import tune
from ray.tune import CLIReporter
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
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--no-ray",
        action="store_true",
        default=False,
        help="run without ray",
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


def test(model,test_dataloader, device=None, verbose=True, attackfn=None, **kwargs):
  preds = []
  labels_oneh = []
  correct = 0
  model.eval()
  with torch.no_grad():
      looper = tqdm(test_dataloader) if verbose else test_dataloader
      for data in looper:
          images, labels = data[0].to(device), data[1].to(device)
          if attackfn is None:
            pred = model(images)
          else:
            with torch.enable_grad():
              images = attackfn(model, images, **kwargs)
            pred = model(images)
          
          # Get softmax values for net input and resulting class predictions
          sm = nn.Softmax(dim=1)
          pred = sm(pred)

          _, predicted_cl = torch.max(pred.data, 1)
          pred = pred.cpu().detach().numpy()

          # Convert labels to one hot encoding
          label_oneh = torch.nn.functional.one_hot(labels, num_classes=10)
          label_oneh = label_oneh.cpu().detach().numpy()

          preds.extend(pred)
          labels_oneh.extend(label_oneh)

          # Count correctly classified samples for accuracy
          correct += sum(predicted_cl == labels).item()

  preds = np.array(preds).flatten()
  labels_oneh = np.array(labels_oneh).flatten()

  total = len(test_dataloader.dataset)
  correct_perc = correct / total
  if verbose:
    print('Accuracy of the network on the {total)} test images: %.2f %%' % (100 * correct_perc))
    # print(correct_perc)
  
  return preds, labels_oneh


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
    norm_thread = config['norm_thread']
    model_name = config['model_name']
    root= params['results_root_path']
    resultsDirName = f'{root}/{model_name}_{norm_thread}'
    if not os.path.exists(resultsDirName):
        os.makedirs(resultsDirName)
        print("Results directory ", resultsDirName,  " Created ")
 
    set_seeds(config['seed'])

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

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
    #                                         shuffle=True, num_workers=2)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=128,
                                            shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                            shuffle=False, num_workers=2)

    # x_test, y_test = load_cifar10(n_examples=10000)

    #Load Model
    model = load_model(model_name=config['model_name'], dataset=config['dataset_name'], threat_model=config['norm_thread'])
    model = model.to(device)
    model.eval()

    ## 
    if params['attack'] =='cw':
        adv_acc = cw_attack(model, test_loader, device)
    else:
        adv_acc = autoattack(model, test_loader, config['norm_thread'], device)

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
    config = {
        "dataset_name": params['dataset_name'],
        "model_name": tune.grid_search(params["model_names"]),
        "seed": tune.grid_search(params["seeds"]),
        "norm_thread": tune.grid_search(params["norm_threads"]),
    }
    if args.dry_run:
        config = {
            "strategy_name": args.debug_strategy,
            "seed": 42,
        }

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpus_per_trial = 1 if use_cuda else 0

    if args.no_ray:
        run_trial(config=config, params=params, args=args, num_gpus=gpus_per_trial)
    else:
        reporter = CLIReporter(
            parameter_columns=["seed", "model_name", "norm_thread"],
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