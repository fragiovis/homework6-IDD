import sys, subprocess
from pathlib import Path

def ensure_deps():
    try:
        import pandas
        import numpy
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'numpy'])

class HLL:
    def __init__(self, p=12):
        self.p = p
        self.m = 1 << p
        self.reg = bytearray(self.m)

    def _rho(self, x, bits):
        r = 1
        while r <= bits and (x >> (bits - r)) & 1 == 0:
            r += 1
        return r

    def add(self, v):
        import hashlib
        if v is None:
            return
        h = int.from_bytes(hashlib.sha1(str(v).encode('utf-8')).digest()[:8], 'big')
        idx = h >> (64 - self.p)
        w = h & ((1 << (64 - self.p)) - 1)
        r = self._rho(w, 64 - self.p)
        if r > self.reg[idx]:
            self.reg[idx] = r

    def estimate(self):
        m = self.m
        import math
        if m == 16:
            alpha = 0.673
        elif m == 32:
            alpha = 0.697
        elif m == 64:
            alpha = 0.709
        else:
            alpha = 0.7213 / (1 + 1.079 / m)
        Z = sum((1.0 / (1 << r)) for r in self.reg)
        E = alpha * (m * m) / Z
        V = self.reg.count(0)
        if E <= 2.5 * m and V > 0:
            E = m * math.log(m / V)
        return E

def eda_file(path):
    import pandas as pd
    chunksize = 100000
    first = True
    cols = []
    missing = {}
    hll = {}
    for chunk in pd.read_csv(path, dtype=str, na_values=['', 'na', 'n/a', 'unknown', 'None'], keep_default_na=True, low_memory=False, chunksize=chunksize):
        if first:
            cols = chunk.columns.tolist()
            for c in cols:
                missing[c] = 0
                hll[c] = HLL()
            first = False
        missing_chunk = chunk.isna().sum()
        for c in cols:
            missing[c] += int(missing_chunk[c])
            vals = chunk[c].dropna().unique()
            for v in vals:
                hll[c].add(v)
    import pandas as pd
    summary = pd.DataFrame({
        'attribute': cols,
        'missing_count': [int(missing[c]) for c in cols],
        'unique_count': [int(round(hll[c].estimate())) for c in cols]
    })
    return summary

def main():
    ensure_deps()
    base = Path(__file__).resolve().parent.parent / 'Dataset' / 'processed'
    used_path = base / 'used_cars_data_mediato.csv'
    veh_path = base / 'vehicles_mediato.csv'
    s_used = eda_file(used_path)
    s_used.to_csv(base / 'eda_used_cars_data_mediato_summary.csv', index=False)
    print(s_used.sort_values('missing_count', ascending=False).head(50).to_string(index=False))
    s_veh = eda_file(veh_path)
    s_veh.to_csv(base / 'eda_vehicles_mediato_summary.csv', index=False)
    print(s_veh.sort_values('missing_count', ascending=False).head(50).to_string(index=False))

if __name__ == '__main__':
    main()

