"""
quicktest.py — Quick-run demonstration for submission
======================================================
Rapid demonstration of five core methods based on synthetic drilling data:
  1. CDMsDDIM   — Conditional diffusion model for diverse plan generation
  2. HNSW++-CHA — Multi-branch HNSW fuzzy matching of similar plans
  3. PGM-Index  — PGM learned index for exact matching
  4. Transformer — Transformer regression for drilling time prediction
  5. TOPSIS     — AHP-entropy-TOPSIS multi-criteria evaluation

Usage:
    python quicktest.py

No external data files required; all data is synthesized in-program.
Each method runs independently; missing a single dependency does not block others.
"""

import numpy as np
import time
import sys
import os

# Change to the script's directory to ensure correct module imports
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
def import_module_from_file(module_name, filepath):
    """Import a Python module from a file path (supports special characters in filenames)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ============================================================
# Generate synthetic drilling data (no external files required)
# ============================================================
np.random.seed(42)

def generate_synthetic_data(n_samples=100):
    """Generate n_samples synthetic drilling plans with 11 parameters."""
    data = np.zeros((n_samples, 11), dtype=np.float32)

    data[:, 0]  = np.random.choice([0, 1, 2], size=n_samples)          # Formation type
    data[:, 1]  = np.random.normal(39, 1.5, n_samples).clip(35, 43)    # WOB (kN)
    data[:, 2]  = np.random.normal(60, 1.0, n_samples).clip(57, 63)    # RPM
    data[:, 3]  = np.random.normal(50, 1.0, n_samples).clip(47, 53)    # Flow rate
    data[:, 4]  = np.random.normal(21, 2.0, n_samples).clip(16, 28)    # Pump pressure (MPa)
    data[:, 5]  = np.random.normal(1.3, 0.1, n_samples).clip(1.2, 1.9) # Density (g/cm^3)
    data[:, 6]  = np.random.normal(50, 1.0, n_samples).clip(47, 53)    # Funnel viscosity (s)
    data[:, 7]  = np.random.normal(2, 0.5, n_samples).clip(0.5, 4)     # theta_3
    data[:, 8]  = np.random.normal(33, 2.0, n_samples).clip(26, 40)    # theta_300
    data[:, 9]  = np.random.normal(65, 15, n_samples).clip(50, 200)    # theta_600

    # Drilling time = base + parameter effects + random noise
    base = (10.0
            + (data[:, 1] - 39) * 2.0
            + (data[:, 5] - 1.3) * 30.0
            + (data[:, 4] - 21) * 0.5
            + np.random.normal(0, 2, n_samples))
    data[:, 10] = base.clip(5, 50)

    print(f"Synthetic data: {n_samples} plans x 11 parameters")
    print(f"Drilling time range: [{data[:, 10].min():.1f}, {data[:, 10].max():.1f}] min")
    return data


schemes_array = generate_synthetic_data(100)
schemes_list = schemes_array.tolist()


# ============================================================
print("\n" + "=" * 60)
print("  1. CDMsDDIM — Conditional Diffusion Model (diverse plan generation)")
print("=" * 60)

try:
    import torch
    cdms = import_module_from_file("CDMsDDIM", "CDMsDDIM.py")
    DiffusionDrillingGenerator = cdms.DiffusionDrillingGenerator

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    t0 = time.time()

    # Reduced parameters for faster demonstration
    generator = DiffusionDrillingGenerator(
        schemes_list, missing_rate=5/11,
        T=50, num_steps=10,
        hidden_dim=64, num_blocks=2,
        dropout=0.1, device=device
    )

    train_time = generator.train(epochs=30, batch_size=16, lr=5e-4)
    print(f"Training time: {train_time:.1f}s (fast mode, 30 epochs)")

    known_positions = [0, 1, 2, 3, 6, 8]
    my_query = {0: 0, 1: 39, 2: 60, 3: 50, 6: 50, 8: 34}
    gen_ids, gen_data = generator.generate(my_query, known_positions, num_samples=5)

    print(f"Known conditions: {my_query}")
    print(f"Generated {len(gen_data)} diverse plans (completing unknown params 4,5,7,9,10):")
    for sid, data in zip(gen_ids, gen_data):
        print(f"  Plan {sid}: PumpPress={data[4]:.1f}, Density={data[5]:.2f}, "
              f"theta3={data[7]:.1f}, theta600={data[9]:.0f}, DrillTime={data[10]:.1f}")

    print(f"Total time: {time.time() - t0:.1f}s")

except ImportError as e:
    print(f"[Skip] Missing dependency: {e}")
except Exception as e:
    print(f"[Error] {type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
print("  2. HNSW++-CHA — Multi-branch HNSW Fuzzy Matching")
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

    print(f"Index build time: {time.time() - t0:.2f}s")

    my_query = {0: 0, 1: 39, 2: 60, 3: 50, 6: 50, 8: 34}
    topk = 10
    result_ids, result_data = fuzzy_matcher.fuzzy_match(my_query, k=topk)

    print(f"Query conditions: {my_query}")
    print(f"Top-{topk} matched plan IDs: {result_ids}")
    if result_data:
        print(f"Nearest plan: ID={result_ids[0]}, DrillTime={result_data[0][10]:.1f} min")

    print(f"Total time: {time.time() - t0:.2f}s")

except ImportError as e:
    print(f"[Skip] Missing dependency: {e}")
except Exception as e:
    print(f"[Error] {type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
print("  3. PGM-Index — PGM Learned Index Exact Matching")
print("=" * 60)

try:
    pgm = import_module_from_file("pgm_module", "PGM-Index.py")
    PGMDrillingMatcher = pgm.PGMDrillingMatcher

    t0 = time.time()

    matcher = PGMDrillingMatcher(schemes_list, epsilon=32)
    print(f"Index build time: {time.time() - t0:.4f}s "
          f"(built PG index for each of {len(schemes_list[0])} parameters)")

    # Use a plan from the dataset as the exact-match query
    target = schemes_list[50]
    my_query = {0: target[0], 1: target[1], 2: target[2],
                3: target[3], 6: target[6], 8: target[8]}

    result_ids, result_data = matcher.exact_match(my_query)
    print(f"Query conditions: {my_query}")
    print(f"Matched plan IDs: {result_ids}")
    if result_data:
        print(f"Matched plan drilling times: {[f'{d[10]:.1f}' for d in result_data]}")

    print(f"Total time: {time.time() - t0:.4f}s")

except ImportError as e:
    print(f"[Skip] Missing dependency: {e}")
except Exception as e:
    print(f"[Error] {type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
print("  4. Transformer — Transformer Regression for Drilling Time Prediction")
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
    print(f"Device: {device}")

    t0 = time.time()

    # Prepare data: first 10 columns as features, last column (drilling time) as target
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

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # Reduced model for faster demonstration
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

    print(f"R^2: {r2:.4f}  |  MAE: {mae:.2f} min  |  RMSE: {rmse:.2f} min")
    print(f"First 5 predictions vs actuals:")
    for i in range(min(5, len(predictions))):
        print(f"  Pred={predictions[i]:.1f} min, Actual={actuals[i]:.1f} min")

    print(f"Total time: {time.time() - t0:.1f}s")

except ImportError as e:
    print(f"[Skip] Missing dependency: {e}")
except Exception as e:
    print(f"[Error] {type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
print("  5. TOPSIS — AHP-Entropy-TOPSIS Comprehensive Ranking")
print("=" * 60)

try:
    topsis_mod = import_module_from_file("topsis_module", "topsis.py")
    forward_transform = topsis_mod.forward_transform
    ahp_weight = topsis_mod.ahp_weight
    entropy_weight = topsis_mod.entropy_weight
    fuse_weights_multiplicative = topsis_mod.fuse_weights_multiplicative
    topsis = topsis_mod.topsis

    t0 = time.time()

    # Use last 10 columns of first 20 plans as evaluation matrix
    eval_matrix = schemes_array[:20, 1:].copy()

    indicator_names = ['WOB(kN)', 'RPM', 'Flow rate', 'Pump pressure',
                       'Density', 'Funnel visc', 'theta3', 'theta300', 'theta600', 'Drill time']

    # max=larger is better, min=smaller is better
    indicator_types = ['max', 'max', 'max', 'max',
                       'min', 'min', 'min', 'min', 'min', 'min']
    bounds = [None] * len(indicator_types)

    # Step 1: Forward transformation (positive normalization)
    forward_matrix = forward_transform(eval_matrix, indicator_types, bounds)

    # Step 2: AHP subjective weights
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
    print(f"AHP subjective weights: {np.round(subj_weights, 3)}")
    print(f"Consistency ratio CR: {cr:.4f}")

    # Step 3: Entropy weight method objective weights
    obj_weights = entropy_weight(forward_matrix)
    print(f"Entropy objective weights: {np.round(obj_weights, 3)}")

    # Step 4: Multiplicative fusion
    fused_weights = fuse_weights_multiplicative(subj_weights, obj_weights)
    print(f"Fused weights:            {np.round(fused_weights, 3)}")

    # Step 5: TOPSIS
    impacts = ['+'] * len(indicator_types)
    scores, rank = topsis(forward_matrix, fused_weights, impacts)

    print(f"\nTOPSIS ranking results (Top 5):")
    sorted_idx = np.argsort(-scores)
    for i in range(min(5, len(scores))):
        idx = sorted_idx[i]
        print(f"  Rank {i+1}: Plan {idx+1}, closeness={scores[idx]:.4f}, "
              f"DrillTime={eval_matrix[idx, -1]:.1f} min")

    print(f"Total time: {time.time() - t0:.4f}s")

except ImportError as e:
    print(f"[Skip] Missing dependency: {e}")
except Exception as e:
    print(f"[Error] {type(e).__name__}: {e}")


# ============================================================
print("\n" + "=" * 60)
print("  All 5 method demonstrations completed")
print("=" * 60)
