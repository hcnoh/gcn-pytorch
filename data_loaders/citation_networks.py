import os
import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


DATASET_DIR = "datasets"


class CitationNetworks(Dataset):
    def __init__(self, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        self.dataset_name = None    # will be defined in child classes
        self.dataset_dir = dataset_dir
        self.num_sample_per_class = 20

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.num_nodes

    def preprocess(self):
        cites_path = os.path.join(
            self.dataset_dir, "{}.cites".format(self.dataset_name)
        )

        col_names = ["To", "From"]

        cites_df = pd.read_csv(
            cites_path, sep="\t", names=col_names, header=None
        )

        content_path = os.path.join(
            self.dataset_dir, "{}.content".format(self.dataset_name)
        )

        col_names = ["Node"] + list(range(3703)) + ["Label"]

        content_df = pd.read_csv(
            content_path, sep="\t", names=col_names, header=None
        )
        content_df["Feature"] = content_df[range(3703)].agg(list, axis=1)
        content_df = content_df[["Node", "Feature", "Label"]]

        node_list = np.array([str(node) for node in content_df["Node"].values])
        node2idx = {node: idx for idx, node in enumerate(node_list)}
        num_nodes = node_list.shape[0]

        X = np.array(
            [np.array(feature) for feature in content_df["Feature"].values]
        )
        num_feature_maps = X.shape[-1]

        class_list = np.unique(content_df["Label"].values)
        class2idx = {c: i for i, c in enumerate(class_list)}
        num_classes = class_list.shape[0]
        Y = np.array(
            class2idx[c] for c in content_df["Label"].values
        )

        drop_indices = []

        for i, row in cites_df.iterrows():
            if row["To"] not in node_list or row["From"] not in node_list:
                drop_indices.append(i)

        cites_df = cites_df.drop(drop_indices)

        A = np.zeros([num_nodes, num_nodes])

        for _, row in cites_df.iterrows():
            to_ = row["To"]
            from_ = row["From"]

            A[node2idx[to_], node2idx[from_]] += 1

        # Self Connection
        A_tilde = A
        for i in range(A_tilde.shape[0]):
            A_tilde[i, i] = 1

        # Renormalization Trick
        D_tilde = np.sum(A_tilde, axis=-1)
        A_hat = np.matmul(
            np.diag(D_tilde ** (-0.5)),
            np.matmul(A_tilde, np.diag(D_tilde ** (-0.5)))
        )

        class2indices = {}
        for c in class_list:
            class2indices[c] = []
            for i, row in content_df.iterrows():
                class2indices[c].append(i)

        train_indices = np.hstack(
            [
                np.random.choice(v, self.num_sample_per_class)
                for _, v in class2indices.items()
            ]
        )
        test_indices = np.delete(np.arange(num_nodes), train_indices)

        return A, A_hat, X, Y, node_list, node2idx, num_nodes, \
            num_feature_maps, class_list, class2idx, num_classes, \
            class2indices, train_indices, test_indices


class Citeseer(CitationNetworks):
    def __init__(self) -> None:
        super().__init__()

        self.dataset_name = "citeseer"
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)

        if os.path.exists(os.path.join(self.dataset_dir, "dataset.pkl")):
            with open(
                os.path.join(self.dataset_dir, "dataset.pkl"), "rb"
            ) as f:
                dataset = pickle.load(f)
        else:
            dataset = self.preprocess()

        self.A, self.A_hat, self.X, self.Y, self.node_list, self.node2idx, \
            self.num_nodes, self.num_feature_maps, self.class_list, \
            self.class2idx, self.num_classes, self.class2indices, \
            self.train_indices, self.test_indices = dataset
