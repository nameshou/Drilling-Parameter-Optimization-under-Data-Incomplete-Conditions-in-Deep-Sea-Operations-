import numpy as np
import pandas as pd
import random
import time
import hnswlib

class HNSWPlusPlus:
    def __init__(self, dim, max_elements, num_branches=3, sample_ratio=0.8,
                 M=16, ef_construction=200, ef_search=100):
        self.num_branches = num_branches
        self.sample_ratio = sample_ratio
        self.ef_search = ef_search
        self.branches = []
        self.id_maps = []
        self.internal_counters = []

        for _ in range(num_branches):
            idx = hnswlib.Index(space='l2', dim=dim)
            idx.init_index(max_elements=max_elements, ef_construction=ef_construction, M=M)
            idx.set_ef(ef_search)
            self.branches.append(idx)
            self.id_maps.append({})
            self.internal_counters.append(0)

    def build_index(self, vectors, external_ids):

        n = len(vectors)
        for b in range(self.num_branches):
            sample_size = int(n * self.sample_ratio)
            sampled_indices = np.random.choice(n, size=sample_size, replace=False)
            sampled_vectors = vectors[sampled_indices]
            sampled_ext_ids = [external_ids[i] for i in sampled_indices]

            internal_ids = list(range(self.internal_counters[b],
                                      self.internal_counters[b] + sample_size))
            self.branches[b].add_items(sampled_vectors, internal_ids)
            for iid, eid in zip(internal_ids, sampled_ext_ids):
                self.id_maps[b][iid] = eid
            self.internal_counters[b] += sample_size

    def query(self, query_vec, k=10):

        all_candidates = []
        seen = set()
        for b in range(self.num_branches):
            labels, dists = self.branches[b].knn_query(query_vec.reshape(1, -1), k=k)
            for label, dist in zip(labels[0], dists[0]):
                ext_id = self.id_maps[b].get(label)
                if ext_id is not None and ext_id not in seen:
                    all_candidates.append((ext_id, dist))
                    seen.add(ext_id)
        all_candidates.sort(key=lambda x: x[1])
        return [cand[0] for cand in all_candidates[:k]]

def load_drilling_data_from_excel(file_path):

    df = pd.read_excel(file_path, header=0)
    schemes = []
    for idx, row in df.iterrows():
        scheme = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                val = 0.0
            try:
                scheme.append(float(val))
            except:
                scheme.append(0.0)
        schemes.append(scheme)

    return schemes

class FuzzyMatcherHNSWPP:

    def __init__(self, schemes, kept_positions,
                 num_branches=3, sample_ratio=0.8,
                 M=16, ef_construction=200, ef_search=100):
        self.schemes = schemes
        self.kept_positions = kept_positions
        self.dim = len(kept_positions)

        vectors = []
        ext_ids = []
        for sid, scheme in enumerate(schemes):
            vec = [scheme[pos] for pos in kept_positions]
            vectors.append(vec)
            ext_ids.append(sid)
        self.vectors = np.array(vectors, dtype=np.float32)
        self.ext_ids = ext_ids

        self.index = HNSWPlusPlus(
            dim=self.dim,
            max_elements=len(schemes),
            num_branches=num_branches,
            sample_ratio=sample_ratio,
            M=M,
            ef_construction=ef_construction,
            ef_search=ef_search
        )
        self.index.build_index(self.vectors, self.ext_ids)

    def fuzzy_match(self, query, k=10):

        q_vec = np.array([query[pos] for pos in self.kept_positions], dtype=np.float32)
        topk_ids = self.index.query(q_vec, k=k)
        topk_data = [self.schemes[sid] for sid in topk_ids]
        return topk_ids, topk_data

if __name__ == "__main__":

    excel_path = r"E:\tool\python project\test\钻速机器学习.xlsm"
    schemes = load_drilling_data_from_excel(excel_path)

    kept_positions = [0, 1, 2, 4, 5, 6]  

    fuzzy_matcher = FuzzyMatcherHNSWPP(
        schemes,
        kept_positions=kept_positions,
        num_branches=3,
        sample_ratio=0.8,
        M=16,
        ef_construction=200,
        ef_search=100
    )

    my_query = {0: 0, 1: 50, 2: 60, 4: 21, 5: 1.27, 6: 50}
    topk = 20
    result_ids, result_data = fuzzy_matcher.fuzzy_match(my_query, k=topk)

    print(f"\n模糊匹配 top‑{topk} 方案 ID: {result_ids}")
    if result_ids:
        print("对应方案数据:")
        for sid, data in zip(result_ids, result_data):
            print(f"  方案{sid}: {data}")
    else:
        print("未找到相似方案")