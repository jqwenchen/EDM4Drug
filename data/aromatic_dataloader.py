import os
import random
import sys
from pathlib import Path
from time import time
from typing import Tuple
from rdkit import Chem

from torch_geometric.data import Data

import networkx as nx
import numpy as np
import torch
import pandas as pd
from torch import zeros, Tensor
from torch.utils.data import Dataset, DataLoader
# from torch_geometric.data import DataLoader
from torch.nn.functional import one_hot
from tqdm import tqdm

from data.mol import Mol, load_xyz, from_rdkit
from data.ring import RINGS_DICT
from utils.args_edm import Args_EDM
from utils.ring_graph import get_rings, get_rings_adj
from utils.molgraph import get_connectivity_matrix, get_edges
from torch_geometric.data import Batch, Data

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





allowable_features = {
    'possible_atomic_num_list':       list(range(1, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple_3D(smile):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    return: graph data object with the attributes: x, edge_index, edge_attr """
    mol = Chem.MolFromSmiles(smile)

    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds, two features: bond type, bond direction
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        num_bond_features = 2
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index,
                edge_attr=edge_attr)
    return data

class RandomRotation(object):
    def __call__(self, x):
        M = torch.randn(3, 3)
        Q, __ = torch.linalg.qr(M)
        return x @ Q


class AromaticDataset(Dataset):
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
        self.df = self.df[(self.df.n_rings <= args.max_nodes) & (self.df.n_rings > 0)].reset_index()
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

        x, node_mask, edge_mask, node_features, y = self.__getitem__(0)[:5]
        self.num_node_features = node_features.shape[1]
        # self.num_targets = y.shape[0]
        if hasattr(args, "num_target"):
            self.num_targets = args.num_target

    def get_edge_mask_orientation(self):
        if self._edge_mask_orientation is None:
            self._edge_mask_orientation = torch.zeros(
                2 * self.max_nodes, 2 * self.max_nodes, dtype=torch.bool
            )
            for i in range(self.max_nodes):
                self._edge_mask_orientation[i, self.max_nodes + i] = True
                self._edge_mask_orientation[self.max_nodes + i, i] = True
        return self._edge_mask_orientation.clone()

    def __len__(self):
        return len(self.examples)

    def rescale_loss(self, x):
        # Convert from normalized to the original representation
        if self.normalize:
            x = x * self.std.to(x.device).mean()
        return x

    def get_mol(self, df_row, skip_hydrogen=False) -> Tuple[Mol, list, Tensor, str]:
        name = df_row["Entry ID"]
        file_path = f"{self.xyz_root}/{name}"#self.xyz_root + "/" + name
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

    def get_rings(self, df_row):
        name = df_row["Entry ID"]
        os.makedirs(self.xyz_root + "_rings_preprocessed", exist_ok=True)
        preprocessed_path = f"{self.xyz_root}_rings_preprocessed/{name}.xyz"#self.xyz_root + "_rings_preprocessed/" + name + ".xyz"
        if Path(preprocessed_path).is_file():
            x, adj, node_features, orientation = torch.load(preprocessed_path)
        else:
            mol, edges, atom_connectivity, _ = self.get_mol(df_row, skip_hydrogen=True)
            # get_figure(mol, edges, showPlot=True, filename='4.png')
            mol_graph = nx.Graph(edges)
            knots = get_rings(mol.atoms, mol_graph)
            adj = get_rings_adj(knots)
            x = torch.tensor([k.get_coord() for k in knots], dtype=DTYPE)
            knot_type = torch.tensor(
                [self.knots_list.index(k.cycle_type) for k in knots]
            ).unsqueeze(1)
            node_features = (
                one_hot(knot_type, num_classes=len(self.knots_list)).squeeze(1).float()
            )
            orientation = [k.orientation for k in knots]
            torch.save([x, adj, node_features, orientation], preprocessed_path)
        return x, adj, node_features, orientation

    def get_atoms(self, df_row):
        name = df_row["molecule"]
        preprocessed_path = self.xyz_root + "_atoms_preprocessed/" + name + ".xyz"
        if Path(preprocessed_path).is_file():
            x, adj, node_features = torch.load(preprocessed_path)
        else:
            mol, edges, atom_connectivity, _ = self.get_mol(df_row)
            # get_figure(mol, edges, showPlot=True)
            x = torch.tensor([a.get_coord() for a in mol.atoms], dtype=DTYPE)
            atom_element = torch.tensor(
                [self.atoms_list.index(atom.element) for atom in mol.atoms]
            ).unsqueeze(1)
            node_features = (
                one_hot(atom_element, num_classes=len(self.atoms_list))
                .squeeze(1)
                .float()
            )
            adj = atom_connectivity
            torch.save([x, adj, node_features], preprocessed_path)
        return x, adj, node_features

    def get_all(self, df_row):
        # extract targets
        y = torch.tensor(
            df_row[self.target_features].values.astype(np.float32), dtype=DTYPE
            # df_row["label"]
        )
        # if self.normalize:
        #     y = (y - self.mean) / self.std

        # creation of nodes, edges and there features
        x, adj, node_features, orientation = self.get_rings(df_row)

        if self.orientation:
            # adjust to max nodes shape
            n_nodes = x.shape[0]
            x_r = torch.tensor([random.sample(o, 1)[0] for o in orientation])
            x_full = zeros(self.max_nodes * 2, 3)
            x_full[:n_nodes] = x
            x_full[self.max_nodes : self.max_nodes + n_nodes] = x_r

            node_mask = zeros(self.max_nodes * 2)
            node_mask[:n_nodes] = 1
            node_mask[self.max_nodes : self.max_nodes + n_nodes] = 1

            node_features_full = zeros(self.max_nodes * 2, node_features.shape[1])
            node_features_full[:n_nodes, :] = node_features
            # mark the orientation nodes as additional ring type
            node_features_full[self.max_nodes : self.max_nodes + n_nodes, -1] = 1

            edge_mask_tmp = node_mask[: self.max_nodes].unsqueeze(0) * node_mask[
                : self.max_nodes
            ].unsqueeze(1)
            # mask diagonal
            diag_mask = ~torch.eye(self.max_nodes, dtype=torch.bool)
            edge_mask_tmp *= diag_mask
            edge_mask = self.get_edge_mask_orientation()
            edge_mask[: self.max_nodes, : self.max_nodes] = edge_mask_tmp

            if self.return_adj:
                adj_full = self.get_edge_mask_orientation()
                adj_full[:n_nodes, :n_nodes] = adj
        else:
            # adjust to max nodes shape
            n_nodes = x.shape[0]
            x_full = zeros(self.max_nodes, 3)

            node_mask = zeros(self.max_nodes)
            x_full[:n_nodes] = x
            node_mask[:n_nodes] = 1

            node_features_full = zeros(self.max_nodes, node_features.shape[1])
            node_features_full[:n_nodes, :] = node_features
            # node_features_full = zeros(self.max_nodes, 0)

            # edge_mask = zeros(self.max_nodes, self.max_nodes)
            # edge_mask[:n_nodes, :n_nodes] = adj
            # edge_mask = edge_mask.view(-1, 1)

            edge_mask = node_mask.unsqueeze(0) * node_mask.unsqueeze(1)
            # mask diagonal
            diag_mask = ~torch.eye(self.max_nodes, dtype=torch.bool)
            edge_mask *= diag_mask
            # edge_mask = edge_mask.view(-1, 1)

            if self.return_adj:
                adj_full = zeros(self.max_nodes, self.max_nodes)
                adj_full[:n_nodes, :n_nodes] = adj

        smile_mol = mol_to_graph_data_obj_simple_3D(df_row["smiles"])
        if self.return_adj:
            return x_full, node_mask, edge_mask, node_features_full, adj_full, y, smile_mol
        else:
            return x_full, node_mask, edge_mask, node_features_full, y, smile_mol

    def __getitem__(self, idx):
        index = self.examples[idx]
        df_row = self.df.loc[index]
        return self.get_all(df_row)

# pahs-cata-34072-xyz
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
        # csv_path = "/home/p1/Desktop/data/connect2.xlsx"
        # xyz_path = "/home/p1/Desktop/data/xyz"
        csv_path = "/home/p1/Desktop/0615_L1700_Compound Library/connect2.xlsx"
        xyz_path = "/home/p1/Desktop/0615_L1700_Compound Library/xyz"
    elif args.dataset == "hetro-dft":
        csv_path = "/home/tomerweiss/PBHs-design/data/db-15067-dft.csv"
        xyz_path = ""
    else:
        raise NotImplementedError
    return csv_path, xyz_path


def get_splits(args, random_seed=42, val_frac=0.1, test_frac=0.1):
    np.random.seed(seed=random_seed)
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
        df = pd.read_csv(csv_path)

    df_all = df.copy()
    df_test = df.sample(frac=test_frac, random_state=random_seed)
    df = df.drop(df_test.index)
    df_val = df.sample(frac=val_frac, random_state=random_seed)
    df_train = df.drop(df_val.index)
    return df_train, df_val, df_test, df_all

def collate_fn(batch):
    x, node_mask, edge_mask, node_features, y, smiles = [], [], [], [], [], []
    for item in batch:
        x.append(item[0].unsqueeze(0))
        node_mask.append(item[1].unsqueeze(0))
        edge_mask.append(item[2].unsqueeze(0))
        node_features.append(item[3].unsqueeze(0))
        y.append(item[4].unsqueeze(0))
        smiles.append(item[5])
    x = torch.cat(x, dim=0)
    node_mask = torch.cat(node_mask, dim=0)
    edge_mask = torch.cat(edge_mask, dim=0)
    node_features = torch.cat(node_features, dim=0)
    y = torch.cat(y, dim=0)
    return x, node_mask, edge_mask, node_features, y, Batch.from_data_list(smiles, [], [])

def create_data_loaders(args):
    args.df_train, args.df_val, args.df_test, args.df_all = get_splits(args)

    train_dataset = AromaticDataset(
        args=args,
        task="train",
    )
    val_dataset = AromaticDataset(
        args=args,
        task="val",
    )
    test_dataset = AromaticDataset(
        args=args,
        task="test",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader
