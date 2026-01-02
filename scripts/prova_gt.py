from pathlib import Path
import pandas as pd

def sample_view(csv_path: Path, n: int = 10):
    pos, neg = [], []
    for chunk in pd.read_csv(csv_path, dtype=str, low_memory=False, chunksize=100000):
        cpos = chunk[chunk['label'] == '1']
        cneg = chunk[chunk['label'] == '0']
        if len(pos) < n:
            pos.append(cpos.head(n - len(pos)))
        if len(neg) < n:
            neg.append(cneg.head(n - len(neg)))
        if len(pos) >= n and len(neg) >= n:
            break
    dpos = pd.concat(pos, ignore_index=True) if pos else pd.DataFrame()
    dneg = pd.concat(neg, ignore_index=True) if neg else pd.DataFrame()
    return dpos, dneg

def main():
    base = Path(__file__).resolve().parent.parent / 'Dataset' / 'processed'
    for name in ['gt_b1.csv', 'gt_b2.csv']:
        path = base / name
        dpos, dneg = sample_view(path, 10)
        stem = Path(name).stem
        out_pos = base / f"{stem}_label1_10.csv"
        out_neg = base / f"{stem}_label0_10.csv"
        if not dpos.empty:
            dpos.to_csv(out_pos, index=False)
            print(f"Salvato {out_pos}")
        else:
            print(f"Nessuna riga label=1 in {name}")
        if not dneg.empty:
            dneg.to_csv(out_neg, index=False)
            print(f"Salvato {out_neg}")
        else:
            print(f"Nessuna riga label=0 in {name}")

if __name__ == '__main__':
    main()
