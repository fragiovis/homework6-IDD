import sys, subprocess
from pathlib import Path

def ensure_deps():
    try:
        import pandas  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'numpy'])

def find_csv():
    candidates = [
        Path(__file__).resolve().parent.parent / 'Dataset' / 'vehicles.csv',
        Path.cwd() / 'Dataset' / 'vehicles.csv',
        Path('Dataset/vehicles.csv'),
        Path('../Dataset/vehicles.csv'),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("File 'vehicles' non trovato")

def main():
    ensure_deps()
    import pandas as pd
    import numpy as np
    path = find_csv()
    df = pd.read_csv(path)
    print(f"Percorso: {path}")
    print("Shape:", df.shape)
    print("Tipi:")
    print(df.dtypes)
    print("Prime 10 righe:")
    print(df.head(10))
    cols = df.columns.tolist()
    missing = df.isna().sum().astype(int)
    unique = df.nunique(dropna=True).astype(int)
    summary = pd.DataFrame({
        'attribute': cols,
        'missing_count': [int(missing[c]) for c in cols],
        'unique_count': [int(unique[c]) for c in cols]
    })
    print("Missing e valori unici per colonna:")
    print(summary.sort_values('missing_count', ascending=False))
    desc = df.select_dtypes(include=[np.number]).describe().T
    print("Statistiche numeriche:")
    print(desc)

if __name__ == '__main__':
    main()
