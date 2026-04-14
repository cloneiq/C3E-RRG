"""
model list
"""
from .baseline import Baseline
from .dymes import DYMES  # C3E-RRG模型代码

model_fn = {'baseline': Baseline, 'dymes': DYMES}
