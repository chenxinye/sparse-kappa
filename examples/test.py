import numpy as np
from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as csp
from sparse_kappa.backend.sparse import linalg as cspla
import torch
import scipy.sparse as sp
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ================== 你之前要的最直接 Power 方法 ==================
def cond_1norm_power(A, maxiter=60, tol=1e-8, verbose=False):
    """最朴素的 Power / fixed-point iteration（完全不使用 Hager）"""
    if not csp.isspmatrix(A):
        raise ValueError("本函数专为 PyTorch backend sparse 矩阵设计")
    
    norm_A1 = cspla.norm(A, ord=1)
    if norm_A1 == 0:
        return 0.0
    
    n = A.shape[0]
    x = cp.ones(n, dtype=A.dtype)
    x = x / cp.sum(cp.abs(x))
    
    est_prev = 0.0
    for k in range(maxiter):
        y = cspla.spsolve(A, x)
        est = float(cp.sum(cp.abs(y)))
        
        if verbose:
            print(f"Iter {k+1:2d}: ||A⁻¹||₁ ≈ {est:.6e}")
        
        if k > 3 and abs(est - est_prev) < tol * est:
            break
            
        x = y / (est + 1e-12)
        est_prev = est
    
    kappa_est = float(norm_A1 * est)
    return kappa_est


# ====================== 对比测试 ======================
np.random.seed(42)
print("正在生成 50 个随机稀疏矩阵并进行对比...\n")

results = []

for i in tqdm(range(50)):
    n = np.random.randint(200, 501)
    density = np.random.uniform(0.03, 0.08)
    
    # 生成稀疏矩阵（加一点对角占优保证可逆）
    A_scipy = sp.random(n, n, density=density, format='csr', dtype=np.float64, random_state=i)
    A_scipy = A_scipy + sp.eye(n, format='csr') * 0.05 * n
    
    # PyTorch 估计
    A_torch = csp.csr_matrix(A_scipy)
    kappa_est = cond_1norm_power(A_torch)
    
    # Torch exact
    A_dense = torch.tensor(A_scipy.toarray(), dtype=torch.float64)
    kappa_exact = torch.linalg.cond(A_dense, p=1).item()
    
    ratio = kappa_est / kappa_exact if kappa_exact > 0 else 0
    rel_error = abs(kappa_est - kappa_exact) / kappa_exact if kappa_exact > 0 else 0
    
    results.append([n, kappa_exact, kappa_est, ratio, rel_error])

results = np.array(results)

# ====================== 结果统计 ======================
print("\n" + "="*70)
print("=== 50个随机稀疏矩阵对比结果（Power 方法 vs Torch exact） ===")
print("="*70)
print(f"矩阵尺寸范围       : {int(results[:,0].min())} ~ {int(results[:,0].max())}")
print(f"Exact κ₁ 平均值    : {results[:,1].mean():.2e}")
print(f"Est κ₁ 平均值      : {results[:,2].mean():.2e}")
print(f"平均 ratio (est/exact) : {results[:,3].mean():.4f}  ← 通常远小于 1")
print(f"最大 ratio         : {results[:,3].max():.4f}")
print(f"平均相对误差       : {results[:,4].mean()*100:.2f}%")
print(f"最大相对误差       : {results[:,4].max()*100:.2f}%")
print(f"中位数相对误差     : {np.median(results[:,4])*100:.2f}%")
print("="*70)
