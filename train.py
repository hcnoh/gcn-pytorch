import os
import argparse
import json
import pickle

import torch

from torch.utils.data import DataLoader, random_split

from data_loaders.citation_networks import Citeseer

from models.gcn import GCN


def main(dataset_name, directed):
    ckpt_path = "ckpts"
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    ckpt_path = os.path.join(ckpt_path, dataset_name)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    if directed:
        ckpt_path = os.path.join(ckpt_path, "directed")
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)
    else:
        ckpt_path = os.path.join(ckpt_path, "undirected")
        if not os.path.exists(ckpt_path):
            os.mkdir(ckpt_path)

    with open("config.json") as f:
        config = json.load(f)
        train_config = config["train_config"]
        model_config = config["model_config"]

    num_epochs = train_config["num_epochs"]
    learning_rate = train_config["learning_rate"]

    H = model_config["H"]
    num_layers = model_config["num_layers"]
    dropout = model_config["dropout"]
    regularization = model_config["regularization"]

    if dataset_name == "citeseer":
        dataset = Citeseer()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    with open(os.path.join(ckpt_path, "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=4)
    with open(os.path.join(ckpt_path, "train_config.json"), "w") as f:
        json.dump(train_config, f, indent=4)

    model = GCN(
        dataset.A_hat, dataset.num_feature_maps, H, dataset.num_classes,
        num_layers, dropout, regularization
    )

    model.train_model(
        num_epochs, learning_rate, dataset, dataset.train_indices,
        dataset.test_indices
    )

    # train_size = dataset.train_indices.shape[0]
    # test_size = dataset.test_indices.shape[0]

    # train_dataset, test_dataset = random_split(
    #     dataset, [train_size, test_size]
    # )

    # train_dataset.indices = dataset.train_indices
    # test_dataset.indices = dataset.test_indices

    # train_loader = DataLoader(
    #     train_dataset, batch_size=train_size, shuffle=False
    # )
    # test_loader = DataLoader(
    #     test_dataset, batch_size=test_size, shuffle=False
    # )

    # print(train_dataset.indices)
    # print(train_loader.indices)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="citeseer",
        help="The name of the dataset to train GCN. \
            The possible names are in [citeseer, cora]. \
            The default name is citeseer."
    )
    parser.add_argument(
        "--directed",
        type=bool,
        default=False,
        help="Whether the dataset is to be preprocessed to directed graph case \
            or not."
    )
    args = parser.parse_args()

    main(args.dataset_name, args.directed)
