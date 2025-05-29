import torch
import torch.nn.functional as F
# from torch_scatter import scatter_add

class VideoTokenCompressor:
    """
    视频视觉token压缩器：将[32,196,896]视频token压缩为[729,896]
    核心原理：能量-熵联合敏感哈希 + 稀疏能量扩散聚合
    """
    
    def __init__(self, target_tokens=729, hash_dim=64):
        self.target_tokens = target_tokens
        self.hash_dim = hash_dim  # 哈希编码维度
        
    def _compute_energy_entropy(self, x):
        """
        能量-熵联合权重计算（物理意义：量化token的显式能量与隐式信息价值）
        x: [N, D] 输入token
        返回：联合权重 [N]
        """
        # 显式能量计算（L2范数平方）
        energy = torch.norm(x, p=2, dim=1)**2  # [N]
        
        # 近似信息熵计算（使用Nystrom方法降低复杂度）
        with torch.no_grad():
            n = x.size(0)
            m = 128  # 地标点数量
            landmark_idx = torch.randperm(n)[:m]
            landmarks = x[landmark_idx]  # [m, D]
            
            # Nystrom近似
            C = x @ landmarks.T  # [N, m]
            W = landmarks @ landmarks.T  # [m, m]
            S, U = torch.linalg.eigh(W)
            inv_S = torch.diag(1.0 / (S + 1e-6))
            proj = C @ U @ inv_S @ U.T  # [N, m]
            
            # 近似自注意力矩阵
            attn = proj @ proj.T  # [N, N]
            probs = F.softmax(attn / torch.sqrt(torch.tensor(x.size(1))), dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1)  # [N]
        
        # 动态权重融合
        alpha = entropy.std() / (energy.std() + entropy.std() + 1e-6)
        return alpha * energy + (1 - alpha) * entropy
    
    def _adaptive_hash(self, x):
        """
        自适应数据驱动型哈希编码（物理意义：捕捉主脉动模式）
        x: [N, D] 输入token
        返回：哈希编码 [N, H]
        """
        # 主成分分析（数据驱动基向量）
        cov = x.T @ x / x.size(0)  # [D, D]
        S, V = torch.linalg.eigh(cov)
        basis = V[:, -self.hash_dim:]  # 取最大特征值对应的向量 [D, H]
        
        # 分层量化哈希
        proj = x @ basis  # [N, H]
        quant_levels = 4  # 每个维度的量化级数
        scaled = (proj - proj.min(dim=0)[0]) / (proj.ptp(dim=0) + 1e-6)
        hash_codes = (scaled * quant_levels).long() % quant_levels  # [N, H]
        
        # 生成紧凑哈希签名
        hash_sign = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for i in range(self.hash_dim):
            hash_sign += hash_codes[:, i] * (quant_levels ** i)
        return hash_sign
    
    def _sparse_aggregation(self, x, weights, hash_sign):
        """
        稀疏能量扩散聚合（物理意义：热传导式能量再平衡）
        x: [N, D] 输入token
        weights: [N] 联合权重
        hash_sign: [N] 哈希编码
        返回：压缩后token [K, D]
        """
        # 选择高能量核心节点
        core_idx = torch.topk(weights, self.target_tokens).indices  # [K]
        core_hash = hash_sign[core_idx]  # [K]
        
        # 构建哈希邻接图（避免全矩阵存储）
        unique_hash, inverse = torch.unique(hash_sign, return_inverse=True)
        hash_table = torch.full((unique_hash.max()+1,), -1, device=x.device)
        hash_table[core_hash] = torch.arange(core_idx.size(0), device=x.device)
        
        # 映射到核心节点
        mask = (hash_table[hash_sign] != -1)
        mapped_idx = hash_table[hash_sign[mask]]  # [M]
        
        # 能量加权聚合
        weighted_x = x[mask] * weights[mask].unsqueeze(1)  # [M, D]
        sum_weights = torch.scatter_add_(weights[mask], mapped_idx, dim_size=self.target_tokens)  # [K]
        sum_x = torch.scatter_add_(weighted_x, mapped_idx, dim_size=self.target_tokens)  # [K, D]
        
        # 归一化处理（能量守恒）
        compressed = sum_x / (sum_weights.unsqueeze(1) + 1e-6)
        return compressed
    
    def forward(self, video_tokens):
        """
        输入：video_tokens [B, F, T, D] = [batch, 32帧, 196token, 896dim]
        输出：压缩后token [B, K, D]
        """
        B, F, T, D = video_tokens.shape
        x = video_tokens.view(B, -1, D)  # [B, N=F*T, D]
        
        compressed_list = []
        for b in range(B):
            # 能量-熵联合权重
            weights = self._compute_energy_entropy(x[b])
            
            # 自适应哈希编码
            hash_sign = self._adaptive_hash(x[b])
            
            # 稀疏能量聚合
            compressed = self._sparse_aggregation(x[b], weights, hash_sign)
            compressed_list.append(compressed)
        
        return torch.stack(compressed_list, dim=0)  # [B, K, D]