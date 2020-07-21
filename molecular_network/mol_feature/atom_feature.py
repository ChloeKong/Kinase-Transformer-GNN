#!/usr/bin/env python
# coding: utf-8

# In[18]:


#!/usr/bin/env python
# coding: utf-8


from ..util import onehot,ringsize
import numpy as np

import torch
import torch.nn.functional as F
from torch_sparse import coalesce

#from rdkit.Chem.Draw import IPythonConsole #Needed to show molecules
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import rdBase
from rdkit.Chem.rdchem import HybridizationType
from rdkit import RDConfig
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem.rdchem import BondType as BT
rdBase.DisableLog('rdApp.error')


# In[79]:


"""
节点特征
假设分子包含 𝑁 个原子，则节点特征可表为 (𝐯1,𝐯2,…,𝐯𝑁) 。其中 𝐯 包含以下分量：

*1.原子类别(14位) atom_types 默认用此列表['H', 'C', 'N', 'O','F', 'S', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Fe', 'Se']
*2.杂化信息(5位) hybrid_spec 默认用此列表['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']，可根据数据集探索结果替换此列表
*3.is_ring(7位)，表示环的长度，默认采用3～8,>8的单独归为一类，可根据数据集探索结果替换此列表
*4.芳香性 aromatic：0或1 (1位)，默认添加，但与键型信息重叠，需要实验是否添加此信息
*5.原子质量mass/80 (1位)
#6.原子序数（1位）atomic_num/35，默认不采用，与原子质量的信息重叠
7.共价键 exp_val/4 (1位)
8.相邻的氢原子数量/4 (1位)
9.Acceptor(1位)：0或1
10.Donor(1位)：0或1
11.Hydrophobe(1位)：0或1
12.NegIonizable(1位)：0或1
13.PosIonizable(1位)：0或1
14.ZnBinder(1位)：0或1

第9-14位是由药效团性质而来：
    根据RDkit写的药效团匹配规则判断，具体规则可通过运行下面代码，查看BaseFeatures.fdef内容
    -fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
    -factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

"""


atom_types = ['H', 'C', 'N', 'O','F', 'S', 'Cl', 'Br', 'I', 'P', 'B', 'Si', 'Fe', 'Se']
hybrid_types = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']


def atom_feature(mol,
                 atom_types=atom_types,
                 hybrid_types=hybrid_types,
                 ring_size=True,
                 aromatic=True,
                 mass=True,
                 atom_num=False,
                 exp_val=True,
                 Hs=True,
                 AccDon=True,
                 Hydro=True,
                 NegPosIon=True,
                 Znb=True
                 ):

    atoms = mol.GetAtoms()
    ri = mol.GetRingInfo()
    ri_a = ri.AtomRings()
    x = []

    for atom in atoms:
        
        feats = []
        
        symbol = onehot(atom.GetSymbol(), atom_types)  # 原子类型
        feats.append(symbol)

        hybrid = onehot(str(atom.GetHybridization()),hybrid_types,other=True)  # 杂化轨道类型
        feats.append(hybrid)
        
        if aromatic == True:
            _ = [atom.GetIsAromatic()+0]  # 是否有芳香性
            feats.append(_)

        if atom_num == True:
            _ = [(atom.GetAtomicNum())/35]  # 原子序号
            feats.append(_) # 按照Br原子序号(35)归一化

        if mass == True:
            _ = [(atom.GetMass())/80]  # 原子质量
            feats.append(_) # 按照Br原子质量(80)归一化
            
        if exp_val == True:
            _ = [(atom.GetExplicitValence())/4]  # 共价键
            feats.append(_) # 按C成键数（4）归一化
        
        if Hs == True:
            _ = [(atom.GetTotalNumHs(includeNeighbors=True))/4]  # 相邻H原子数
            feats.append(_) #按C原子相邻H原子数（4）归一化
        
        if ring_size == True:
            _ = ringsize(atom, ri_a)/8  # 环的长度
            feats.append(_) #归一化
            
        x_ = np.concatenate(feats)

        x.append(x_)

    return torch.Tensor(x)

