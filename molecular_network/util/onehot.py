#!/usr/bin/env python
# coding: utf-8



import numpy as np

import torch
import torch.nn.functional as F
from torch_sparse import coalesce


import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
rdBase.DisableLog('rdApp.error')




def onehot(c, cat, other=False):  # val: 值；cat: 类别，other：True时，不在cat列表中，则归为单独一类
    length = len(cat)
    cat = dict(zip(cat,list(range(length))))
    if other == False:
        onehot = np.zeros(length)
        if c in cat:
            onehot[cat[c]] = 1
        assert onehot.sum() == 1, '该类型不包含在类型列表中，或选用other=True，将此类别归为一类'
    if other == True:
        onehot = np.zeros(length+1)
        if c in cat:
            onehot[cat[c]] = 1
        else:
            onehot[-1] = 1
        assert onehot.sum() == 1 
    return onehot



def ringSize_a(a, rings, max_size = 8, min_size = 3):  # 定义环的长度。在使用时，rings变量输入的是ri_a
    onehot = np.zeros(max_size - min_size + 1)
    aid = a.GetIdx()
    for ring in rings:  # ring是其中一个环
        if aid in ring and len(ring) <= max_size:
            # 如果原子在环上，就会有这项。环长度至少为3，故减去3。用+=，是考虑到原子在多个环上的情况
            onehot[len(ring) - min_size] += 1
    return onehot