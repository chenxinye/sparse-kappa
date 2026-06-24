from sparse_kappa.backend import torch_api as cp
from sparse_kappa.backend import sparse as sp
from sparse_kappa import cond_estimate
import time
import pandas as pd

def benchmark_all_methods():
    """
    全面测试sparse-kappa的所有1-norm和2-norm方法
    测量条件数估计值和计算时间
    """
    
    # 创建测试稀疏矩阵 (5000×5000, 0.5%密度)
    print("=" * 80)
    print("创建测试稀疏矩阵...")
    print("=" * 80)
    
    n = 5000
    density = 0.005
    A = sp.random(n, n, density=density, format='csr', dtype=cp.float64)
    
    # 添加对角主导性以确保可逆
    A = A + sp.eye(n, format='csr') * 10
    
    print(f"矩阵大小: {n}×{n}")
    print(f"密度: {density*100:.2f}%")
    print(f"非零元素数: {A.nnz}")
    print(f"数据类型: {A.dtype}\n")
    
    results = []
    
    # =====================================================================
    # 测试 1-NORM 方法
    # =====================================================================
    print("=" * 80)
    print("测试 1-NORM 方法")
    print("=" * 80)
    
    norm1_methods = [
        ('hager', 'lu'),
        ('hager-higham', 'lu'),
        ('power', 'lu'),
        ('oettli-prager', 'lu'),
        ('block-hager', 'lu'),
        # 使用迭代求解器
        ('hager', 'lsmr'),
        ('hager', 'bicgstab'),
        ('hager', 'cg'),
    ]
    
    for method, solver in norm1_methods:
        print(f"\n方法: {method:20s} | 求解器: {solver:10s}")
        print("-" * 60)
        
        try:
            # 预热GPU
            _ = cond_estimate(A, norm=1, method=method, solver=solver, max_iter=5)
            cp.cuda.Stream.null.synchronize()
            
            # 正式测试
            start_time = time.time()
            result = cond_estimate(
                A, 
                norm=1, 
                method=method, 
                solver=solver,
                max_iter=100,
                return_dict=True
            )
            cp.cuda.Stream.null.synchronize()
            elapsed_time = time.time() - start_time
            
            cond_num = result['condition_number']
            iterations = result.get('iterations', 'N/A')
            
            print(f"  条件数:    κ₁(A) = {cond_num:.6e}")
            print(f"  迭代次数:  {iterations}")
            print(f"  计算时间:  {elapsed_time:.4f} 秒")
            print(f"  ||A||₁:    {result.get('norm_A', 'N/A'):.6e}")
            print(f"  ||A⁻¹||₁:  {result.get('norm_Ainv', 'N/A'):.6e}")
            
            results.append({
                'Norm': '1-norm',
                'Method': method,
                'Solver': solver,
                'Condition Number': f"{cond_num:.6e}",
                'Iterations': iterations,
                'Time (s)': f"{elapsed_time:.4f}",
                'Status': '✓'
            })
            
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            results.append({
                'Norm': '1-norm',
                'Method': method,
                'Solver': solver,
                'Condition Number': 'N/A',
                'Iterations': 'N/A',
                'Time (s)': 'N/A',
                'Status': f'✗ ({str(e)[:30]})'
            })
    
    # =====================================================================
    # 测试 2-NORM 方法
    # =====================================================================
    print("\n" + "=" * 80)
    print("测试 2-NORM 方法")
    print("=" * 80)
    
    norm2_methods = [
        'svds',
        'eigsh',
        'lobpcg',
        'power',
        'lanczos',
        'lanczos_unsym',
        'golub-kahan',
        'auto',
    ]
    
    for method in norm2_methods:
        print(f"\n方法: {method:20s}")
        print("-" * 60)
        
        try:
            # 预热GPU
            _ = cond_estimate(A, norm=2, method=method, max_iter=5)
            cp.cuda.Stream.null.synchronize()
            
            # 正式测试
            start_time = time.time()
            result = cond_estimate(
                A, 
                norm=2, 
                method=method,
                max_iter=100,
                return_dict=True
            )
            cp.cuda.Stream.null.synchronize()
            elapsed_time = time.time() - start_time
            
            cond_num = result['condition_number']
            iterations = result.get('iterations', 'N/A')
            
            print(f"  条件数:    κ₂(A) = {cond_num:.6e}")
            print(f"  迭代次数:  {iterations}")
            print(f"  计算时间:  {elapsed_time:.4f} 秒")
            print(f"  σ_max:     {result.get('sigma_max', 'N/A'):.6e}")
            print(f"  σ_min:     {result.get('sigma_min', 'N/A'):.6e}")
            
            results.append({
                'Norm': '2-norm',
                'Method': method,
                'Solver': 'N/A',
                'Condition Number': f"{cond_num:.6e}",
                'Iterations': iterations,
                'Time (s)': f"{elapsed_time:.4f}",
                'Status': '✓'
            })
            
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            results.append({
                'Norm': '2-norm',
                'Method': method,
                'Solver': 'N/A',
                'Condition Number': 'N/A',
                'Iterations': 'N/A',
                'Time (s)': 'N/A',
                'Status': f'✗ ({str(e)[:30]})'
            })
    
    # =====================================================================
    # 测试对称矩阵 (2-NORM eigsh优化)
    # =====================================================================
    print("\n" + "=" * 80)
    print("测试对称矩阵 (eigsh方法优化)")
    print("=" * 80)
    
    # 创建对称矩阵
    A_sym = sp.random(n, n, density=density, format='csr', dtype=cp.float64)
    A_sym = (A_sym + A_sym.T) / 2 + sp.eye(n, format='csr') * 10
    
    print(f"\n对称矩阵大小: {n}×{n}")
    print(f"密度: {density*100:.2f}%\n")
    
    try:
        start_time = time.time()
        result = cond_estimate(A_sym, norm=2, method='eigsh', return_dict=True)
        cp.cuda.Stream.null.synchronize()
        elapsed_time = time.time() - start_time
        
        print(f"  条件数:    κ₂(A_sym) = {result['condition_number']:.6e}")
        print(f"  计算时间:  {elapsed_time:.4f} 秒")
        
        results.append({
            'Norm': '2-norm (sym)',
            'Method': 'eigsh',
            'Solver': 'N/A',
            'Condition Number': f"{result['condition_number']:.6e}",
            'Iterations': result.get('iterations', 'N/A'),
            'Time (s)': f"{elapsed_time:.4f}",
            'Status': '✓'
        })
    except Exception as e:
        print(f"  ❌ 错误: {str(e)}")
    
    # =====================================================================
    # 汇总结果
    # =====================================================================
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # 统计成功率
    success_count = df[df['Status'] == '✓'].shape[0]
    total_count = df.shape[0]
    print(f"\n成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    # 找出最快的方法
    print("\n" + "=" * 80)
    print("性能排名 (Top 5 最快方法)")
    print("=" * 80)
    
    df_success = df[df['Status'] == '✓'].copy()
    df_success['Time_Float'] = df_success['Time (s)'].astype(float)
    df_top5 = df_success.nsmallest(5, 'Time_Float')[['Norm', 'Method', 'Solver', 'Condition Number', 'Time (s)']]
    print("\n" + df_top5.to_string(index=False))
    
    return df


if __name__ == '__main__':
    # 确保GPU可用
    print("=" * 80)
    print("GPU信息")
    print("=" * 80)
    
    # 获取GPU信息的正确方式
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    print(f"GPU设备ID: {device.id}")
    print(f"GPU名称: {props['name'].decode()}")
    print(f"计算能力: {props['major']}.{props['minor']}")
    print(f"总显存: {props['totalGlobalMem'] / 1024**3:.2f} GB")
    print(f"PyTorch版本: {cp.__version__}")
    print()
    
    # 运行基准测试
    results_df = benchmark_all_methods()
    
    # 可选: 保存结果到CSV
    results_df.to_csv('sparse_kappa_benchmark_results.csv', index=False)
    print("\n✅ 结果已保存到 sparse_kappa_benchmark_results.csv")