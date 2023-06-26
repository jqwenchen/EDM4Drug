import os
import random
import sys
from pathlib import Path
from time import time
from typing import Tuple

import networkx as nx
import numpy as np
import torch
import pandas as pd
from torch import zeros, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
from tqdm import tqdm

from data.mol import Mol, load_xyz, from_rdkit
from data.ring import RINGS_DICT
from utils.args_edm import Args_EDM
from utils.ring_graph import get_rings, get_rings_adj
from utils.molgraph import get_connectivity_matrix, get_edges

DTYPE = torch.float32
INT_DTYPE = torch.int8
# ATOMS_LIST = __ATOM_LIST__[:8]
ATOMS_LIST = {
    "cata": ["H", "C"],
    "peri": ["H", "C"],
    "hetro": ["H", "C", "B", "N", "O", "S"],
}
RINGS_LIST = {
    "cata": ["Bn"],
    "peri": ["Bn"],
    "hetro": list(RINGS_DICT.keys()) + ["."],
}


class RandomRotation(object):
    def __call__(self, x):
        M = torch.randn(3, 3)
        Q, __ = torch.linalg.qr(M)
        return x @ Q


class AromaticDataset():
    def __init__(self, args, task: str = "train"):
        """
        Args:
            args: All the arguments.
            task: Select the dataset to load from (train/val/test).
        """
        self.csv_file, self.xyz_root = get_paths(args)

        self.task = task
        self.rings_graph = args.rings_graph
        self.normalize = args.normalize
        self.max_nodes = args.max_nodes
        self.return_adj = False
        self.dataset = args.dataset
        self.target_features = getattr(args, "target_features", None)
        self.target_features = (
            self.target_features.split(",") if self.target_features else []
        )
        self.orientation = False if self.dataset == "cata" else True
        self._edge_mask_orientation = None
        self.atoms_list = ATOMS_LIST[self.dataset]
        self.knots_list = RINGS_LIST[self.dataset]

        self.df = getattr(args, f"df_{task}").reset_index()
        if args.normalize:
            train_df = args.df_train
            try:
                target_data = train_df[self.target_features].values
            except:
                self.target_features = [
                    t.replace(" ", "") for t in self.target_features
                ]
                target_data = train_df[self.target_features].values
            self.mean = torch.tensor(target_data.mean(0), dtype=DTYPE)
            self.std = torch.tensor(target_data.std(0), dtype=DTYPE)
        else:
            self.std = torch.ones(1, dtype=DTYPE)
            self.mean = torch.zeros(1, dtype=DTYPE)

        self.examples = np.arange(self.df.shape[0])
        if args.sample_rate < 1:
            random.shuffle(self.examples)
            num_files = round(len(self.examples) * args.sample_rate)
            self.examples = self.examples[:num_files]

    def get_edge_mask_orientation(self):
        if self._edge_mask_orientation is None:
            self._edge_mask_orientation = torch.zeros(
                2 * self.max_nodes, 2 * self.max_nodes, dtype=torch.bool
            )
            for i in range(self.max_nodes):
                self._edge_mask_orientation[i, self.max_nodes + i] = True
                self._edge_mask_orientation[self.max_nodes + i, i] = True
        return self._edge_mask_orientation.clone()

    def rescale_loss(self, x):
        # Convert from normalized to the original representation
        if self.normalize:
            x = x * self.std.to(x.device).mean()
        return x

    def get_mol(self, df_row, skip_hydrogen=False) -> Tuple[Mol, list, Tensor, str]:
        name = df_row["Entry ID"]
        file_path = f"{self.xyz_root}/{name}.xyz"
        if os.path.exists(file_path + ".xyz"):
            mol = load_xyz(file_path + ".xyz")
            atom_connectivity = get_connectivity_matrix(
                mol.atoms, skip_hydrogen=skip_hydrogen
            )  # build connectivity matrix
            # edges = bonds
        elif os.path.exists(file_path + ".pkl"):
            mol, atom_connectivity = from_rdkit(file_path + ".pkl")
        elif os.path.exists(file_path):
            mol = load_xyz(file_path)
            atom_connectivity = get_connectivity_matrix(
                mol.atoms, skip_hydrogen=skip_hydrogen
            )  # build connectivity matrix
            # edges = bonds
        else:
            raise NotImplementedError(file_path)
        edges = get_edges(atom_connectivity)
        return mol, edges, atom_connectivity, name

    def get_rings_num(self, df_row, pbar):
        pbar.update(1)
        mol, edges, atom_connectivity, _ = self.get_mol(df_row, skip_hydrogen=False)
        mol_graph = nx.Graph(edges)
        knots = get_rings(mol.atoms, mol_graph)
        return len(knots)

def get_paths(args):
    if not hasattr(args, "dataset"):
        csv_path = args.csv_file
        xyz_path = args.xyz_root
    elif args.dataset == "cata":
        csv_path = "/home/p1/Desktop/Drug-Dataset/compas/COMPAS-1x.csv"
        xyz_path = "/home/p1/Desktop/Drug-Dataset/compas/pahs-cata-34072-xyz"
    elif args.dataset == "peri":
        csv_path = "/home/tomerweiss/PBHs-design/data/peri-xtb-data-55821.csv"
        xyz_path = "/home/tomerweiss/PBHs-design/data/peri-cata-89893-xyz"
    elif args.dataset == "hetro":
        # csv_path = "/home/tomerweiss/PBHs-design/data/db-474K-OPV-phase-2-filtered.csv"
        # xyz_path = "/home/tomerweiss/PBHs-design/data/db-474K-xyz"
        # csv_path = "/home/p1/Desktop/data/connect.xlsx"
        # xyz_path = "/home/p1/Desktop/data/xyz"
        csv_path = "/home/p1/Desktop/0615_L1700_Compound Library/信息对应表.xlsx"
        xyz_path = "/home/p1/Desktop/0615_L1700_Compound Library/xyz"
    elif args.dataset == "hetro-dft":
        csv_path = "/home/tomerweiss/PBHs-design/data/db-15067-dft.csv"
        xyz_path = ""
    else:
        raise NotImplementedError
    return csv_path, xyz_path


def get_splits(args):
    csv_path, _ = get_paths(args)
    if hasattr(args, "dataset") and args.dataset == "hetro" and not csv_path.endswith("xlsx"):
        targets = (
            args.target_features.split(",")
            if getattr(args, "target_features", None) is not None
            else []
        )
        df = pd.read_csv(csv_path, usecols=["name", "nRings", "inchi"] + targets)
        df.rename(columns={"nRings": "n_rings", "name": "molecule"}, inplace=True)
        args.max_nodes = min(args.max_nodes, 10)
    elif csv_path.endswith("xlsx"):
        df = pd.read_excel(csv_path)
    else:
        # df = pd.read_excel(csv_path)
        df = pd.read_csv(csv_path)

    return df, df, df, df



if __name__ == '__main__':
    args = Args_EDM().parse_args()
    args.df_train, args.df_val, args.df_test, args.df_all = get_splits(args)

    dataset = AromaticDataset(
        args=args,
        task="train",
    )
    pbar =tqdm(total=args.df_train.shape[0])

    args.df_train["n_rings"] = args.df_train.apply(dataset.get_rings_num, axis=1, args=(pbar, ))
    args.df_train.to_excel("/home/p1/Desktop/0615_L1700_Compound Library/connect1.xlsx", index=False)
    from data.ring import RINGS_DICT
    print(RINGS_DICT)


