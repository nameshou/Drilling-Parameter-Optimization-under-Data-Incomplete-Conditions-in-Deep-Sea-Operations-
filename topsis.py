import numpy as np
import matplotlib.pyplot as plt
import math

plt.rcParams['axes.unicode_minus'] = False  

def forward_transform(matrix, types, bounds=None):

    matrix = np.array(matrix, dtype=float)
    m, n = matrix.shape
    result = matrix.copy()
    for j in range(n):
        t = types[j]
        if t == 'max':
            continue
        elif t == 'min':
            max_val = np.max(matrix[:, j])
            result[:, j] = max_val - matrix[:, j]
        elif t == 'mid':
            x_best = bounds[j]
            M = np.max(np.abs(matrix[:, j] - x_best))
            if M == 0:
                result[:, j] = 1
            else:
                result[:, j] = 1 - np.abs(matrix[:, j] - x_best) / M
        elif t == 'range':
            a, b = bounds[j]
            col = matrix[:, j]
            d = np.zeros(m)
            d[col < a] = a - col[col < a]
            d[col > b] = col[col > b] - b
            M = np.max(d)
            if M == 0:
                result[:, j] = 1
            else:
                result[:, j] = 1 - d / M
        else:
            raise ValueError("指标类型必须为 'max','min','mid','range' 之一")
    return result
def ahp_weight(comparison_matrix):
    n = comparison_matrix.shape[0]
    product = np.power(np.prod(comparison_matrix, axis=1), 1/n)
    weights = product / np.sum(product)
    Aw = np.dot(comparison_matrix, weights)
    lambda_max = np.mean(Aw / weights)
    CI = (lambda_max - n) / (n - 1)
    RI_dict = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    RI = RI_dict.get(n, 1.49)
    CR = CI / RI if RI != 0 else 0.0
    return weights, CR

def entropy_weight(matrix):
    min_vals = matrix.min(axis=0)
    max_vals = matrix.max(axis=0)
    norm_matrix = (matrix - min_vals) / (max_vals - min_vals + 1e-10)
    p = norm_matrix / (norm_matrix.sum(axis=0, keepdims=True) + 1e-10)
    m, n = matrix.shape
    e = - (1 / np.log(m)) * np.sum(p * np.log(p + 1e-10), axis=0)
    g = 1 - e
    weights = g / g.sum()
    return weights

def fuse_weights_multiplicative(ahp_w, entropy_w):
    combined = np.sqrt(ahp_w * entropy_w)
    fused = combined / np.sum(combined)
    return fused

def topsis(matrix, weights, impacts=None):
    if impacts is None:
        impacts = ['+'] * matrix.shape[1]
    norm_matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0, keepdims=True))
    weighted_matrix = norm_matrix * weights
    ideal_best = []
    ideal_worst = []
    for j, impact in enumerate(impacts):
        if impact == '+':
            ideal_best.append(weighted_matrix[:, j].max())
            ideal_worst.append(weighted_matrix[:, j].min())
        else:
            ideal_best.append(weighted_matrix[:, j].min())
            ideal_worst.append(weighted_matrix[:, j].max())
    ideal_best = np.array(ideal_best)
    ideal_worst = np.array(ideal_worst)
    dist_best = np.sqrt(np.sum((weighted_matrix - ideal_best)**2, axis=1))
    dist_worst = np.sqrt(np.sum((weighted_matrix - ideal_worst)**2, axis=1))
    closeness = dist_worst / (dist_best + dist_worst)
    return closeness, np.argsort(-closeness)




def plot_single_radar(data, labels, best=None, worst=None, weights=None,
                      rank_indices=None, title="Radar Chart", save_path=None,
                      figsize=(8, 8), colors_map=plt.cm.tab10):
    n_indicators = data.shape[1]
    if rank_indices is None:
        rank_indices = np.arange(len(data))
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, n_indicators, endpoint=False).tolist()
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    all_vals = data[rank_indices].flatten()
    if best is not None:
        all_vals = np.concatenate([all_vals, best])
    if worst is not None:
        all_vals = np.concatenate([all_vals, worst])
    vmin, vmax = all_vals.min(), all_vals.max()
    margin = 0.1 * (vmax - vmin + 1e-10)
    vmin = max(0, vmin - margin)
    vmax = vmax + margin
    ax.set_ylim(vmin, vmax)
    ax.grid(True)
    
    if best is not None:
        best_vals = best.tolist() + [best[0]]
        ax.plot(angles, best_vals, 'r--', linewidth=2.5, label='Positive Ideal (Best)')

    if worst is not None:
        worst_vals = worst.tolist() + [worst[0]]
        ax.plot(angles, worst_vals, 'b--', linewidth=2.5, label='Negative Ideal (Worst)')

    group_data = data[rank_indices]
    colors = colors_map(np.linspace(0, 1, len(rank_indices)))
    for i, idx in enumerate(rank_indices):
        vals = group_data[i].tolist() + [group_data[i, 0]]
        if i == 0:
            ax.plot(angles, vals, 'o-', linewidth=2.5, color=colors[i],
                    label=f'Plan {idx+1} (Best)')
        else:
            ax.plot(angles, vals, 'o-', linewidth=1.5, color=colors[i],
                    label=f'Plan {idx+1}')
        ax.fill(angles, vals, alpha=0.05, color=colors[i])
    
    label_offset = vmax * 1.15
    for i, (angle, label) in enumerate(zip(angles[:-1], labels)):
        ha = 'left' if 0 <= angle <= np.pi else 'right'
        ax.text(angle, label_offset, label, ha=ha, va='center',
                rotation=0, fontsize=10, rotation_mode='anchor')
        if weights is not None:
            ax.text(angle, label_offset * 0.95, f"({weights[i]:.2f})",
                    ha=ha, va='center', rotation=0, fontsize=8, color='gray')
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":

    indicator_names = ['WOB (kN)', 'RPM', 'Flow rate (r/min)', 
                   'Pump pressure (MPa)', 'Density (g/cm³)', 
                   'Funnel viscosity (s)', 'θ3', 'θ300', 'θ600', 
                   'Drilling time']
    original_matrix = np.array([
        [50, 60, 50, 21, 1.27, 50, 1, 32, 58, 13],
        [50, 60, 50, 21, 1.27, 50, 1, 32, 58, 16.9],
        [50, 60, 50, 21, 1.26, 50, 1, 32, 58, 18.9],
        [51, 60, 50, 21, 1.27, 50, 1, 32, 58, 15.2],
        [49, 60, 50, 21, 1.27, 50, 1, 32, 58, 19],
        [50, 60, 50, 21, 1.27, 49, 1, 32, 58, 10.3],
        [49, 60, 50, 21, 1.27, 50, 1, 32, 58, 18.1],
        [51, 60, 50, 21, 1.26, 50, 1, 32, 58, 20.3],
        [51, 60, 50, 21, 1.26, 50, 1, 32, 58, 34.4],
        [50, 60, 50, 20, 1.26, 50, 1, 32, 58, 31],
        [50.0, 60.0, 13.54, 21.0, 1.27, 50.0, 2.06, 110.82, 131.97, 11.46],
        [50.0, 60.0, 34.79, 21.0, 1.27, 50.0, 3.26, 95.7, 99.8, 11.01],
        [50.0, 60.0, 16.14, 21.0, 1.27, 50.0, 3.17, 40.13, 65.2, 10.31],
        [50.0, 60.0, 13.69, 21.0, 1.27, 50.0, 2.87, 57.59, 76.68, 13.27],
        [50.0, 60.0, 14.19, 21.0, 1.27, 50.0, 3.01, 36.95, 91.89,14.69],
        [50.0, 60.0, 12.18, 21.0, 1.27, 50.0, 2.66, 126.26, 55.03, 16.20],
        [50.0, 60.0, 37.17, 21.0, 1.27, 50.0, 2.32, 26.85, 104.18, 15.47],
        [50.0, 60.0, 47.61, 21.0, 1.27, 50.0, 2.27, 51.68, 100.97, 16.23],
        [50.0, 60.0, 41.13, 21.0, 1.27, 50.0, 2.25, 54.67, 113.89, 15.95],
        [50.0, 60.0, 15.47, 21.0, 1.27, 50.0, 3.43, 39.46, 60.74, 15.59]
    ])
    indicator_types = [ 'max', 'max', 'max', 'max','min','min','min','min','min','min'] 
    bounds = [None] * len(indicator_types)

    forward_matrix = forward_transform(original_matrix, indicator_types, bounds)
    print("正向化后的决策矩阵：")
    print(forward_matrix)

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
    print("\nAHP 主观权重:", np.round(subj_weights, 4))
    print("一致性比率 CR =", round(cr, 4))
    print("矩阵形状:", original_matrix.shape)
    print("types长度:", len(indicator_types))
    print("bounds长度:", len(bounds))

    obj_weights = entropy_weight(forward_matrix)
    print("熵权法客观权重:", np.round(obj_weights, 4))

    fused_weights = fuse_weights_multiplicative(subj_weights, obj_weights)
    print("乘法合成融合权重:", np.round(fused_weights, 4))

    impacts = ['+'] * len(indicator_types) 
    scores, rank = topsis(forward_matrix, fused_weights, impacts)
    print("\nTOPSIS 评价结果：")
    for i in range(len(scores)):
        print(f"方案 {i+1}: 贴近度 = {scores[i]:.4f}, 排名 = {np.where(rank == i)[0][0]+1}")

    max_vals = forward_matrix.max(axis=0)
    max_vals[max_vals == 0] = 1
    radar_data = forward_matrix / max_vals

    best_ideal = np.ones(len(indicator_names))

    min_vals = forward_matrix.min(axis=0)
    worst_ideal = min_vals / max_vals

    ranked_indices = rank.tolist() 

    group_size = 5
    for g in range(4):
        start = g * group_size
        end = min((g + 1) * group_size, len(ranked_indices))
        group_indices = ranked_indices[start:end]
        
        plot_single_radar(
    data=radar_data,
    labels=indicator_names,
    best=best_ideal,
    worst=worst_ideal,
    weights=fused_weights,
    rank_indices=group_indices,
    title=f"Radar Chart - Group {g+1} (Rank {start+1}–{end})",
    save_path=f"radar_group{g+1}.png",
    figsize=(8, 8)
)