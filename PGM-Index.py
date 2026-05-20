import numpy as np
import pandas as pd
import random
import time
from collections import defaultdict
from pympler import asizeof

class PurePGMIndex:
    
    def __init__(self, epsilon=64):
        self.epsilon = epsilon
        self.keys = []          
        self.row_ids = []      
        self.segments = []      

    def build(self, values, ids):

        val_to_ids = defaultdict(list)
        for v, i in zip(values, ids):
            val_to_ids[v].append(i)
        sorted_vals = sorted(val_to_ids.items())
        self.keys = [v for v, _ in sorted_vals]
        self.row_ids = [ids for _, ids in sorted_vals]
        if not self.keys:
            return
        self._build_segments()

    def _build_segments(self):
        self.segments = []
        n = len(self.keys)
        i = 0
        while i < n:
            start_idx = i
            start_key = self.keys[i]
            j = i + 1

            while j < n:
                end_key = self.keys[j]
                if end_key == start_key:
                    slope, intercept = 0.0, start_idx
                else:
                    slope = (j - start_idx) / (end_key - start_key)
                    intercept = start_idx - slope * start_key

                max_err = 0
                for k in range(start_idx, j + 1):
                    pred = slope * self.keys[k] + intercept
                    err = abs(pred - k)
                    if err > self.epsilon:
                        max_err = err
                        break
                if max_err > self.epsilon:
                    break
                else:
                    j += 1
            end_idx = j - 1
            end_key = self.keys[end_idx]
            if end_key == start_key:
                slope, intercept = 0.0, start_idx
            else:
                slope = (end_idx - start_idx) / (end_key - start_key)
                intercept = start_idx - slope * start_key
            self.segments.append({
                'start_key': start_key,
                'end_key': end_key,
                'slope': slope,
                'intercept': intercept
            })
            i = j

    def query(self, value):

        if not self.keys:
            return []
        lo, hi = 0, len(self.segments) - 1
        seg = None
        while lo <= hi:
            mid = (lo + hi) // 2
            s = self.segments[mid]
            if value < s['start_key']:
                hi = mid - 1
            elif value > s['end_key']:
                lo = mid + 1
            else:
                seg = s
                break
        if seg is None:
            return []

        pred = int(seg['slope'] * value + seg['intercept'])
        left = max(0, pred - self.epsilon)
        right = min(len(self.keys) - 1, pred + self.epsilon)
        while left <= right:
            m = (left + right) // 2
            if self.keys[m] == value:
                return self.row_ids[m]
            elif self.keys[m] < value:
                left = m + 1
            else:
                right = m - 1
        return []

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
    print(f"成功加载 {len(schemes)} 个钻井方案")
    return schemes

class PGMDrillingMatcher:

    def __init__(self, schemes, primary_params=None, epsilon=64):
        self.schemes = schemes
        self.num_params = len(schemes[0]) if schemes else 0

        if primary_params is None:
            self.primary_params = list(range(self.num_params))
        else:
            self.primary_params = primary_params
        self.epsilon = epsilon
        self.pgm_indices = {}
        self._build_indices()

    def _build_indices(self):
        for param_idx in self.primary_params:
            values = []
            ids = []
            for sid, scheme in enumerate(self.schemes):
                values.append(scheme[param_idx])
                ids.append(sid)
            pgm = PurePGMIndex(epsilon=self.epsilon)
            pgm.build(values, ids)
            self.pgm_indices[param_idx] = pgm

    def exact_match(self, query, max_results=20):

        result = None
        for param_idx, value in query.items():
            if param_idx not in self.pgm_indices:
 
                return [], []
            ids = set(self.pgm_indices[param_idx].query(value))
            if result is None:
                result = ids
            else:
                result &= ids
            if not result:      
                return [], []
        scheme_ids = sorted(list(result))[:max_results]
        scheme_data = [self.schemes[sid] for sid in scheme_ids]
        return scheme_ids, scheme_data

def test_pgm_matcher():
    excel_path = r"E:\tool\python project\test\钻速机器学习.xlsm"
    schemes = load_drilling_data_from_excel(excel_path)
    matcher = PGMDrillingMatcher(schemes, epsilon=64)
    return matcher

if __name__ == "__main__":
    matcher = test_pgm_matcher()

    print("\n" + "="*50)
    my_query = {0: 0, 1: 50, 2: 60, 4: 21, 5: 1.27, 6: 50}
    my_scheme_ids, my_scheme_data = matcher.exact_match(my_query)
    print(f"匹配到的方案编号: {my_scheme_ids}")

    if my_scheme_ids:
        print("匹配到的方案数据:")
        for i, (sid, data) in enumerate(zip(my_scheme_ids, my_scheme_data)):
            print(f"  方案{sid}: {data}")
    else:
        print("未匹配到任何方案")