"""
quicktest.py — 投稿用快速运行示例
===================================
基于合成钻井数据，快速演示五个方法的核心功能：
  1. CDMsDDIM   — 条件扩散模型生成差异性钻井方案
  2. HNSW++-CHA — 多分支 HNSW 模糊匹配相似方案
  3. PGM-Index  — PGM 可学习索引精确匹配
  4. Transformer — Transformer 回归预测钻时
  5. TOPSIS     — AHP-熵权-TOPSIS 多指标综合评价

运行方式：
    python quicktest.py

无需外部数据文件，所有数据均为程序内合成。
每个方法独立运行，单个依赖缺失不影响其他方法演示。
"""

import numpy as np
import time
import sys
import os

# 切换到脚本所在目录，确保能正确导入同目录下的模块
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
def import_module_from_file(module_name, filepath):
    """从文件路径导入 Python 模块（支持含特殊字符的文件名）"""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================
# 生成合成钻井数据（无需外部文件）
# ============================================================
np.random.seed(42)

def generate_synthetic_data(n_samples=100):
    """生成 n_samples 条合成钻井数据，共 11 个参数"""
    data = np.zeros((n_samples, 11), dtype=np.float32)

    data[:, 0]  = np.random.choice([0, 1, 2], size=n_samples)          # 地层类型
    data[:, 1]  = np.random.normal(39, 1.5, n_samples).clip(35, 43)    # 钻压 WOB (kN)
    data[:, 2]  = np.random.normal(60, 1.0, n_samples).clip(57, 63)    # 转速 RPM
    data[:, 3]  = np.random.normal(50, 1.0, n_samples).clip(47, 53)    # 排量
    data[:, 4]  = np.random.normal(21, 2.0, n_samples).clip(16, 28)    # 泵压 (MPa)
    data[:, 5]  = np.random.normal(1.3, 0.1, n_samples).clip(1.2, 1.9) # 密度 (g/cm³)
    data[:, 6]  = np.random.normal(50, 1.0, n_samples).clip(47, 53)    # 漏斗粘度 (s)
    data[:, 7]  = np.random.normal(2, 0.5, n_samples).clip(0.5, 4)     # θ3
    data[:, 8]  = np.random.normal(33, 2.0, n_samples).clip(26, 40)    # θ300
    data[:, 9]  = np.random.normal(65, 15, n_samples).clip(50, 200)    # θ600

    # 钻时 = 基础值 + 参数影响 + 随机噪声
    base = (10.0
            + (data[:, 1] - 39) * 2.0
            + (data[:, 5] - 1.3) * 30.0
            + (data[:, 4] - 21) * 0.5
            + np.random.normal(0, 2, n_samples))
    data[:, 10] = base.clip(5, 50)

    print(f"合成数据: {n_samples} 条方案 x 11 个参数")
    print(f"钻时范围: [{data[:, 10].min():.1f}, {data[:, 10].max():.1f}] min")
    return data


schemes_array = generate_synthetic_data(100)
schemes_list = schemes_array.tolist()


# ============================================================
print("\n" + "=" * 60)
print("  1. CDMsDDIM — 条件扩散模型（差异性方案生成）")
print("=" * 60)

try:
    import torch
    cdms = import_module_from_file("CDMsDDIM", "CDMsDDIM.py")
    DiffusionDrillingGenerator = cdms.DiffusionDrillingGenerator

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"运行设备: {device}")

    t0 = time.time()

    # 精简参数以加速演示
    generator = DiffusionDrillingGenerator(
        schemes_list, missing_rate=5/11,
        T=50, num_steps=10,
        hidden_dim=64, num_blocks=2,
        dropout=0.1, device=device
    )

    train_time = generator.train(epochs=30, batch_size=16, lr=5e-4)
    print(f"训练用时: {train_time:.1f}s (快速模式, 30 epochs)")

    known_positions = [0, 1, 2, 3, 6, 8]
    my_query = {0: 0, 1: 39, 2: 60, 3: 50, 6: 50, 8: 34}
    gen_ids, gen_data = generator.generate(my_query, known_positions, num_samples=5)

    print(f"已知条件: {my_query}")
    print(f"生成 {len(gen_data)} 条差异性方案 (补全未知参数 4,5,7,9,10):")
    for sid, data in zip(gen_ids, gen_data):
        print(f"  方案{sid}: 泵压={data[4]:.1f}, 密度={data[5]:.2f}, "
              f"θ3={data[7]:.1f}, θ600={data[9]:.0f}, 钻时={data[10]:.1f}")

    print(f"总用时: {time.time() - t0:.1f}s")

except ImportError as e:
    print(f"[跳过] 缺少依赖: {e}")
except Exception as e:
    print(f"[错误] {type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
print("  2. HNSW++-CHA — 多分支 HNSW 模糊匹配")
print("=" * 60)

try:
    hnsw = import_module_from_file("hnsw_module", "HNSW++-CHA.py")
    FuzzyMatcherHNSWPP = hnsw.FuzzyMatcherHNSWPP

    t0 = time.time()

    kept_positions = [0, 1, 2, 3, 6, 8]
    fuzzy_matcher = FuzzyMatcherHNSWPP(
        schemes_list,
        kept_positions=kept_positions,
        num_branches=3, sample_ratio=0.8,
        M=8, ef_construction=100, ef_search=50
    )

    print(f"索引构建用时: {time.time() - t0:.2f}s")

    my_query = {0: 0, 1: 39, 2: 60, 3: 50, 6: 50, 8: 34}
    topk = 10
    result_ids, result_data = fuzzy_matcher.fuzzy_match(my_query, k=topk)

    print(f"查询条件: {my_query}")
    print(f"匹配 top-{topk} 方案 ID: {result_ids}")
    if result_data:
        print(f"最近似方案: ID={result_ids[0]}, 钻时={result_data[0][10]:.1f} min")

    print(f"总用时: {time.time() - t0:.2f}s")

except ImportError as e:
    print(f"[跳过] 缺少依赖: {e}")
except Exception as e:
    print(f"[错误] {type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
print("  3. PGM-Index — PGM 可学习索引精确匹配")
print("=" * 60)

try:
    pgm = import_module_from_file("pgm_module", "PGM-Index.py")
    PGMDrillingMatcher = pgm.PGMDrillingMatcher

    t0 = time.time()

    matcher = PGMDrillingMatcher(schemes_list, epsilon=32)
    print(f"索引构建用时: {time.time() - t0:.4f}s (对 {len(schemes_list[0])} 个参数分别建 PG 索引)")

    # 取数据中某条方案的值作为精确匹配查询
    target = schemes_list[50]
    my_query = {0: target[0], 1: target[1], 2: target[2],
                3: target[3], 6: target[6], 8: target[8]}

    result_ids, result_data = matcher.exact_match(my_query)
    print(f"查询条件: {my_query}")
    print(f"匹配方案 ID: {result_ids}")
    if result_data:
        print(f"匹配方案钻时: {[f'{d[10]:.1f}' for d in result_data]}")

    print(f"总用时: {time.time() - t0:.4f}s")

except ImportError as e:
    print(f"[跳过] 缺少依赖: {e}")
except Exception as e:
    print(f"[错误] {type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
print("  4. Transformer — Transformer 回归预测钻时")
print("=" * 60)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from sklearn.preprocessing import RobustScaler

    tf_mod = import_module_from_file("transformer_module", "Transformer.py")
    TransformerRegressor = tf_mod.TransformerRegressor
    DrillingDataset = tf_mod.DrillingDataset
    train_model = tf_mod.train_model
    evaluate_model = tf_mod.evaluate_model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"运行设备: {device}")

    t0 = time.time()

    # 准备数据: 前10列作为特征，最后一列(钻时)作为目标
    X_raw = schemes_array[:, :-1].astype(np.float64)
    y_raw = schemes_array[:, -1].astype(np.float64)
    y_log = np.log1p(y_raw)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw)

    seq_length = 10
    X_seq, y_seq = [], []
    for i in range(seq_length, len(X_scaled)):
        X_seq.append(X_scaled[i - seq_length:i])
        y_seq.append(y_log[i])

    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32)

    split_idx = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

    print(f"训练样本: {len(X_train)}, 测试样本: {len(X_test)}")

    # 精简模型以加速演示
    input_dim = X_train.shape[2]
    model = TransformerRegressor(
        input_dim=input_dim, d_model=32, nhead=4, num_layers=1, dropout=0.1
    ).to(device)

    train_dataset = DrillingDataset(X_train, y_train, augment=False)
    test_dataset = DrillingDataset(X_test, y_test, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    train_losses, train_maes = train_model(
        model, train_loader, criterion, optimizer, device, epochs=30
    )

    actuals, predictions, r2, mae, rmse = evaluate_model(model, test_loader, device)

    print(f"R²: {r2:.4f}  |  MAE: {mae:.2f} min  |  RMSE: {rmse:.2f} min")
    print(f"前5个预测 vs 实际:")
    for i in range(min(5, len(predictions))):
        print(f"  预测={predictions[i]:.1f} min, 实际={actuals[i]:.1f} min")

    print(f"总用时: {time.time() - t0:.1f}s")

except ImportError as e:
    print(f"[跳过] 缺少依赖: {e}")
except Exception as e:
    print(f"[错误] {type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
print("  5. TOPSIS — AHP-熵权-TOPSIS 综合评价排序")
print("=" * 60)

try:
    topsis_mod = import_module_from_file("topsis_module", "topsis.py")
    forward_transform = topsis_mod.forward_transform
    ahp_weight = topsis_mod.ahp_weight
    entropy_weight = topsis_mod.entropy_weight
    fuse_weights_multiplicative = topsis_mod.fuse_weights_multiplicative
    topsis = topsis_mod.topsis

    t0 = time.time()

    # 取前20条方案的后10列作为评价矩阵
    eval_matrix = schemes_array[:20, 1:].copy()

    indicator_names = ['WOB(kN)', 'RPM', 'Flow rate', 'Pump pressure',
                       'Density', 'Funnel visc', 'θ3', 'θ300', 'θ600', 'Drill time']

    # max=越大越好, min=越小越好
    indicator_types = ['max', 'max', 'max', 'max',
                       'min', 'min', 'min', 'min', 'min', 'min']
    bounds = [None] * len(indicator_types)

    # Step 1: 正向化
    forward_matrix = forward_transform(eval_matrix, indicator_types, bounds)

    # Step 2: AHP 主观权重
    ahp_matrix = np.array([
        [1,   1,   2,   4,   5,   6,   7, 7, 7, 1  ],
        [1,   1,   2,   4,   5,   6,   7, 7, 7, 1  ],
        [1/2, 1/2, 1,   3,   4,   5,   6, 6, 6, 1/2],
        [1/4, 1/4, 1/3, 1,   2,   3,   4, 4, 4, 1/3],
        [1/5, 1/5, 1/4, 1/2, 1,   2,   3, 3, 3, 1/4],
        [1/6, 1/6, 1/5, 1/3, 1/2, 1,   2, 2, 2, 1/5],
        [1/7, 1/7, 1/6, 1/4, 1/3, 1/2, 1, 1, 1, 1/6],
        [1/7, 1/7, 1/6, 1/4, 1/3, 1/2, 1, 1, 1, 1/6],
        [1/7, 1/7, 1/6, 1/4, 1/3, 1/2, 1, 1, 1, 1/6],
        [1,   1,   2,   3,   4,   5,   6, 6, 6, 1  ]
    ])
    subj_weights, cr = ahp_weight(ahp_matrix)
    print(f"AHP 主观权重: {np.round(subj_weights, 3)}")
    print(f"一致性比率 CR: {cr:.4f}")

    # Step 3: 熵权法客观权重
    obj_weights = entropy_weight(forward_matrix)
    print(f"熵权客观权重: {np.round(obj_weights, 3)}")

    # Step 4: 乘法融合
    fused_weights = fuse_weights_multiplicative(subj_weights, obj_weights)
    print(f"融合权重:      {np.round(fused_weights, 3)}")

    # Step 5: TOPSIS
    impacts = ['+'] * len(indicator_types)
    scores, rank = topsis(forward_matrix, fused_weights, impacts)

    print(f"\nTOPSIS 排序结果 (Top 5):")
    sorted_idx = np.argsort(-scores)
    for i in range(min(5, len(scores))):
        idx = sorted_idx[i]
        print(f"  第{i+1}名: 方案{idx+1}, 贴近度={scores[idx]:.4f}, "
              f"钻时={eval_matrix[idx, -1]:.1f} min")

    print(f"总用时: {time.time() - t0:.4f}s")

except ImportError as e:
    print(f"[跳过] 缺少依赖: {e}")
except Exception as e:
    print(f"[错误] {type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
print("  全部 5 个方法演示完成")
print("=" * 60)
