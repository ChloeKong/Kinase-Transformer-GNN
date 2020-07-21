#!/usr/bin/env python
# coding: utf-8


import rdkit
from rdkit import Chem
import numpy as np

def ringSize_a(a, rings, max_size = 8, min_size = 3):  # 定义环的长度。在使用时，rings变量输入的是ri_a
    onehot = np.zeros(max_size - min_size + 1)
    aid = a.GetIdx()
    for ring in rings:  # ring是其中一个环
        if aid in ring and len(ring) <= max_size:
            # 如果原子在环上，就会有这项。环长度至少为3，故减去3。用+=，是考虑到原子在多个环上的情况
            onehot[len(ring) - min_size] += 1
    return onehot

def ringsize(atom, rings):  
    bits = np.zeros(7)
    id = atom.GetIdx()
    for ring in rings:  # 大于等于9个原子的环，单独为一类
        a = len(ring)
        if a > 8:
            a = 9
        if id in ring:
            bits[a - 3] += 1
    return bits