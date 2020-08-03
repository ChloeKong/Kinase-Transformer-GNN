#!/usr/bin/env python
# coding: utf-8


import os
import os.path as osp


import torch
from torch_geometric.data import (InMemoryDataset, Data)

from molecular_network.mol_feature import (atom_pos, 
                                          bond_feature, 
                                          atom_feature)
import pandas as pd
import rdkit
from rdkit import Chem


class Dataset(InMemoryDataset):
    '''
    需指定raw_file的文件名,默认为全数据集（1亿一千万分子）
    '''
    def __init__(self, root, raw_name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.raw_name = raw_name
        super(Dataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0]) 


    # 这里只要覆写以下四个方法即可，帮助上一句代码实现load（processed文件）
    @property
    def raw_file_names(self):
        return self.raw_name

    @property
    def processed_file_names(self):
        return 'data.pt'


    def process(self):

        df = pd.read_csv(self.raw_paths[0])
        smis = df.iloc[:,0]
        target = df.iloc[:,1]
        
        data_list = []
        i = 0
        for i,smi in enumerate(smis):           

            mol = Chem.MolFromSmiles(smi)

            if mol.GetNumAtoms() == 1:
                continue
                
            mol = Chem.AddHs(mol)

            try: #遇到特殊分子，无法生成特征的就跳过

                x = atom_feature(mol)
                edge_index, edge_attr = bond_feature(mol)  

            except:
                print('第%i个分子被跳过，无法生成节点或者边特征,%s' %(i,smi))
                continue

            y = target[i]

            if i %1000 == 0:
                print('process %i molecules' %(i))

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                                     
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])





