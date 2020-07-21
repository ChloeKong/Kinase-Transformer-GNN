#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import os.path as osp

import torch
from torch_geometric.data import (InMemoryDataset, download_url, extract_zip,
                                  Data)

from molecular_network.mol_feature import (atom_pos, 
                                          bond_feature, 
                                          atom_feature)
import rdkit
from rdkit import Chem


# In[2]:


# 目的：convert target值，对应19个target，后面用到“target * conversion.view(1, -1)”
HAR2EV = 27.2113825435
KCALMOL2EV = 0.04336414

conversion = torch.tensor([
    1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
    1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.
])


# In[3]:


class QM9(InMemoryDataset):

    # raw_url = ('https://s3-us-west-1.amazonaws.com/deepchem.io/datasets/'
    #            'molnet_publish/qm9.zip')
    # raw_url2 = 'https://ndownloader.figshare.com/files/3195404' 
    # processed_url = 'http://www.roemisch-drei.de/qm9.zip' 


    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None, fully_connect=True):
        self.fc = fully_connect
        super(QM9, self).__init__(root, transform, pre_transform, pre_filter)

        # 继承父类，Dataset在初始化时，即对是否需要download，是否需要自己process进行了判断以及运行，最后都是得到了processed文件
        self.data, self.slices = torch.load(self.processed_paths[0]) 

    # 这里只要覆写以下四个方法即可，帮助上一句代码实现load（processed文件）
    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def processed_file_names(self):
        return 'data.pt'  

    # def download(self):
    #     if rdkit is None:
    #         file_path = download_url(self.processed_url, self.raw_dir)
    #         extract_zip(file_path, self.raw_dir)
    #         os.unlink(file_path)
    #     else:
    #         file_path = download_url(self.raw_url, self.raw_dir)
    #         extract_zip(file_path, self.raw_dir)
    #         os.unlink(file_path)

    #         file_path = download_url(self.raw_url2, self.raw_dir)
    #         os.rename(
    #             osp.join(self.raw_dir, '3195404'),
    #             osp.join(self.raw_dir, 'uncharacterized.txt'))

    def process(self):
        # 读取target
        with open(self.raw_paths[1], 'r') as f:
            # 提取target并变形
            target = f.read().split('\n')[1:-1]
            target = [[float(x) for x in line.split(',')[1:20]]
                      for line in target]
            target = torch.tensor(target, dtype=torch.float)
            target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
            target = target * conversion.view(1, -1)

        # 跳过问题分子
        with open(self.raw_paths[2], 'r') as f:
            skip = [int(x.split()[0]) for x in f.read().split('\n')[9:-2]]
        assert len(skip) == 3054

        # 读取sdf文件
        suppl = Chem.SDMolSupplier(self.raw_paths[0],removeHs=False) # 注意此处不除H，需因任务不同，给定不同参数
       
        data_list = []
        for i, mol in enumerate(suppl):
            # 前两步意为跳过分子为空，或者在'uncharacterized.txt'文件中的分子
            if mol is None:
                continue
            if i in skip:
                continue

            x = atom_feature(mol)

            #pos = atom_pos(suppl,i,mol)  

            edge_index, edge_attr = bond_feature(mol,fully_connect=self.fc)  

            y = target[i].unsqueeze(0) 

            data = Data(x=x, edge_index=edge_index,
                        edge_attr=edge_attr, y=y)
            if i%1000 == 0:
                print('process ' + str(i) + ' molecules')
            
            data_list.append(data)

        # collate是InMemoryDataset特有的方法，跟内存有关，另外，保存出这个dataset
        torch.save(self.collate(data_list), self.processed_paths[0])







