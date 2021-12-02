import os
import pickle

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


DATASET_DIR = "datasets"


class CitationNetworks(Dataset):
    def __init__(self, dataset_dir=DATASET_DIR) -> None:
        super().__init__()

        # will be defined in child classes
        self.dataset_name = None
        self.directed = None
        self.num_features = None

        self.dataset_dir = dataset_dir

        self.num_train_samples_per_class = 20
        self.num_test_samples = 1000

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.num_nodes

    def preprocess(self):
        '''
            The preprocess methods are from the following references:
            - http://proceedings.mlr.press/v48/yanga16.pdf
            - https://arxiv.org/pdf/1609.02907.pdf
        '''
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

        col_names = ["Node"] + list(range(self.num_features)) + ["Label"]

        content_df = pd.read_csv(
            content_path, sep="\t", names=col_names, header=None
        )
        content_df["Feature"] = content_df[range(self.num_features)]\
            .agg(list, axis=1)
        content_df = content_df[["Node", "Feature", "Label"]]

        node_list = np.array([str(node) for node in content_df["Node"].values])
        node2idx = {node: idx for idx, node in enumerate(node_list)}
        num_nodes = node_list.shape[0]

        # Row normalization for the feature matrix
        X = np.array(
            [np.array(feature) for feature in content_df["Feature"].values]
        )
        X = X / np.sum(X, axis=-1, keepdims=True)
        num_feature_maps = X.shape[-1]

        class_list = np.unique(content_df["Label"].values)
        class2idx = {c: i for i, c in enumerate(class_list)}
        num_classes = class_list.shape[0]
        Y = np.array(
            [class2idx[c] for c in content_df["Label"].values]
        )

        drop_indices = []

        for i, row in cites_df.iterrows():
            if str(row["To"]) not in node_list or \
                    str(row["From"]) not in node_list:
                drop_indices.append(i)

        cites_df = cites_df.drop(drop_indices)

        A = np.zeros([num_nodes, num_nodes])

        for _, row in cites_df.iterrows():
            to_ = str(row["To"])
            from_ = str(row["From"])

            A[node2idx[to_], node2idx[from_]] = 1
            if not self.directed:
                A[node2idx[from_], node2idx[to_]] = 1

        # Self Connection
        A_tilde = np.copy(A)
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
                np.random.choice(v, self.num_train_samples_per_class)
                for _, v in class2indices.items()
            ]
        )
        test_indices = np.delete(np.arange(num_nodes), train_indices)
        test_indices = np.random.choice(test_indices, self.num_test_samples)

        return A, A_hat, X, Y, node_list, node2idx, num_nodes, \
            num_feature_maps, class_list, class2idx, num_classes, \
            class2indices, train_indices, test_indices


class Citeseer(CitationNetworks):
    def __init__(self, directed) -> None:
        super().__init__()

        self.directed = directed

        self.num_features = 3703

        self.dataset_name = "citeseer"
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)
        if self.directed:
            self.preprocessed_dir = os.path.join(
                self.dataset_dir, "directed"
            )
        else:
            self.preprocessed_dir = os.path.join(
                self.dataset_dir, "undirected"
            )
        print(self.preprocessed_dir)

        if not os.path.exists(self.preprocessed_dir):
            os.mkdir(self.preprocessed_dir)

        if os.path.exists(os.path.join(self.preprocessed_dir, "dataset.pkl")):
            with open(
                os.path.join(self.preprocessed_dir, "dataset.pkl"), "rb"
            ) as f:
                dataset = pickle.load(f)
        else:
            dataset = self.preprocess()
            with open(
                os.path.join(self.preprocessed_dir, "dataset.pkl"), "wb"
            ) as f:
                pickle.dump(dataset, f)

        self.A, self.A_hat, self.X, self.Y, self.node_list, self.node2idx, \
            self.num_nodes, self.num_feature_maps, self.class_list, \
            self.class2idx, self.num_classes, self.class2indices, \
            self.train_indices, self.test_indices = dataset


class Cora(CitationNetworks):
    def __init__(self, directed) -> None:
        super().__init__()

        self.directed = directed

        self.num_features = 1433

        self.dataset_name = "cora"
        self.dataset_dir = os.path.join(self.dataset_dir, self.dataset_name)
        if self.directed:
            self.preprocessed_dir = os.path.join(
                self.dataset_dir, "directed"
            )
        else:
            self.preprocessed_dir = os.path.join(
                self.dataset_dir, "undirected"
            )

        if not os.path.exists(self.preprocessed_dir):
            os.mkdir(self.preprocessed_dir)

        if os.path.exists(os.path.join(self.preprocessed_dir, "dataset.pkl")):
            with open(
                os.path.join(self.preprocessed_dir, "dataset.pkl"), "rb"
            ) as f:
                dataset = pickle.load(f)
        else:
            dataset = self.preprocess()
            with open(
                os.path.join(self.preprocessed_dir, "dataset.pkl"), "wb"
            ) as f:
                pickle.dump(dataset, f)

        self.A, self.A_hat, self.X, self.Y, self.node_list, self.node2idx, \
            self.num_nodes, self.num_feature_maps, self.class_list, \
            self.class2idx, self.num_classes, self.class2indices, \
            self.train_indices, self.test_indices = dataset
