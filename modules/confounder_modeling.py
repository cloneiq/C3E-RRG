import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfounderEncoder(nn.Module):
	"""
	混淆因子编码器：
	- Z_lang: PMI语言先验 (从训练语料的词共现矩阵计算)
	- Z_corr: 相关共现编码 (从疾病标签共现统计计算)
	- Z = concat([Z_lang, Z_corr]) -> [B, z_dim]

	输入：
	- reports_ids: [B, T] 文本 token id（或 None）
	- training: bool

	行为：
	- 训练时返回 batch 内打乱的 Z（后门调整）
	"""
	def __init__(self,
	             pmi_matrix_path: str,
	             disease_corr_path: str,
	             z_dim: int = 512):
		super().__init__()
		self.z_dim = int(z_dim)
		self.z_lang_dim = self.z_dim // 2
		self.z_corr_dim = self.z_dim - self.z_lang_dim

		# ---- 加载 PMI 矩阵 [V, V]（稀疏/致密均可，转换为 float32 张量）----
		if not os.path.isfile(pmi_matrix_path):
			raise FileNotFoundError(f"[ConfounderEncoder] PMI matrix not found: {pmi_matrix_path}")
		pmi = torch.load(pmi_matrix_path, map_location="cpu")
		if isinstance(pmi, dict) and "pmi" in pmi:
			pmi = pmi["pmi"]
		if not isinstance(pmi, torch.Tensor):
			pmi = torch.tensor(pmi)
		if pmi.dim() != 2 or pmi.size(0) != pmi.size(1):
			raise RuntimeError(f"[ConfounderEncoder] PMI must be square [V,V], got {tuple(pmi.shape)}")
		self.register_buffer("PMI", pmi.float())  # [V,V]
		self.vocab_size = int(self.PMI.size(0))

		# 将 PMI 向稠密映射：采用可学习的线性投影
		self.lang_proj = nn.Linear(self.vocab_size, self.z_lang_dim, bias=False)

		# ---- 加载疾病相关共现矩阵 [L, L] 并编码为固定上下文
		if not os.path.isfile(disease_corr_path):
			raise FileNotFoundError(f"[ConfounderEncoder] disease corr not found: {disease_corr_path}")
		import numpy as np
		corr = np.load(disease_corr_path)
		corr_t = torch.from_numpy(corr).float()
		if corr_t.dim() != 2 or corr_t.size(0) != corr_t.size(1):
			raise RuntimeError(f"[ConfounderEncoder] disease corr must be square [L,L], got {tuple(corr_t.shape)}")
		self.register_buffer("D_CORR", corr_t)  # [L,L]

		flat_dim = int(corr_t.numel())
		self.corr_mlp = nn.Sequential(
			nn.Linear(flat_dim, max(64, self.z_corr_dim * 2)),
			nn.GELU(),
			nn.Linear(max(64, self.z_corr_dim * 2), self.z_corr_dim)
		)

	def _encode_language_prior(self, reports_ids: torch.Tensor) -> torch.Tensor:
		"""
		基于 PMI 的语言先验编码：
		- 对每条报告构建平均 one-hot 词袋 v ∈ R^V
		- 计算 pmi_vec = v @ PMI ∈ R^V
		- 通过可学习投影到 z_lang_dim
		返回：[B, z_lang_dim]
		"""
		B, T = reports_ids.size()
		device = reports_ids.device
		# one-hot 聚合（不构图地构造词袋，避免稀疏开销）
		bow = reports_ids.new_zeros((B, self.vocab_size)).float()
		# 安全处理：忽略越界 token id
		mask = (reports_ids >= 0) & (reports_ids < self.vocab_size)
		idx = reports_ids[mask]
		b_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, T)[mask]
		bow.index_add_(0, b_ids, F.one_hot(idx, num_classes=self.vocab_size).float())
		# 归一化（每句平均）
		bow = bow / (bow.sum(dim=-1, keepdim=True).clamp(min=1.0))
		# PMI 聚合
		pmi_vec = bow @ self.PMI.to(device)  # [B,V]
		z_lang = self.lang_proj(pmi_vec)     # [B, z_lang_dim]
		return z_lang

	def _encode_disease_corr(self, B: int, device) -> torch.Tensor:
		"""
		将群体疾病共现矩阵编码为固定上下文，并广播到 batch。
		"""
		corr_flat = self.D_CORR.view(1, -1).to(device)  # [1, L*L]
		z_corr_1 = self.corr_mlp(corr_flat)             # [1, z_corr_dim]
		z_corr = z_corr_1.expand(B, -1)                 # [B, z_corr_dim]
		return z_corr

	def forward(self, reports_ids: Optional[torch.Tensor], training: bool = False) -> torch.Tensor:
		if reports_ids is None:
			return torch.zeros(1, self.z_dim, device=self.PMI.device)

		B = reports_ids.size(0)
		device = reports_ids.device

		if not training:
			z_lang = self._encode_language_prior(reports_ids)
			z_corr = self._encode_disease_corr(B, device)
			return torch.cat([z_lang, z_corr], dim=-1)

		# 训练态：正常计算后再打乱（后门干预）
		z_lang = self._encode_language_prior(reports_ids)
		z_corr = self._encode_disease_corr(B, device)
		Z = torch.cat([z_lang, z_corr], dim=-1)
		with torch.no_grad():
			perm = torch.randperm(B, device=device)
		return Z.index_select(0, perm)






