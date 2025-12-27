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
        Path(__file__).resolve().parent.parent / 'Dataset' / 'used_cars_data.csv',
        Path.cwd() / 'Dataset' / 'used_cars_data.csv',
        Path('Dataset/used_cars_data.csv'),
        Path('../Dataset/used_cars_data.csv'),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("File 'used_cars_data.csv' non trovato")

def main():
    ensure_deps()
    import pandas as pd
    import numpy as np
    import math

    path = find_csv()
    print(f"Percorso: {path}")

    # colonne numeriche
    numeric_cols = [
        'city_fuel_economy','combine_fuel_economy','daysonmarket','dealer_zip',
        'engine_displacement','highway_fuel_economy','horsepower','latitude','longitude'
    ]
    # colonne categoriche
    cat_cols = [
        'vin','back_legroom','bed','bed_height','bed_length','body_type','cabin','city',
        'description','engine_cylinders','engine_type','exterior_color','fleet','frame_damaged',
        'franchise_dealer','franchise_make','front_legroom','fuel_tank_volume','fuel_type',
        'has_accidents','height','interior_color','isCab','is_certified','is_cpo','is_new',
        'is_oemcpo','length','listed_date','listing_color','listing_id','main_picture_url'
    ]

    cols = None
    missing = None
    first_chunk = True
    stats = {c: {'count': 0, 'mean': 0.0, 'M2': 0.0, 'min': None, 'max': None} for c in numeric_cols}
    uniq = {c: set() for c in cat_cols}

    for chunk in pd.read_csv(
        path,
        dtype=str,
        chunksize=100000,
        na_values=['', 'na', 'n/a', 'unknown', 'None'],
        keep_default_na=True,
    ):
        if first_chunk:
            print("Prime 10 righe:")
            print(chunk.head(10))
            cols = chunk.columns.tolist()
            missing = chunk.isna().sum()
            first_chunk = False
        else:
            missing += chunk.isna().sum()

        for c in numeric_cols:
            s = pd.Series(pd.to_numeric(chunk.get(c), errors='coerce'))
            vals = s.dropna().values
            for x in vals:
                st = stats[c]
                st['count'] += 1
                d = x - st['mean']
                st['mean'] += d / st['count']
                st['M2'] += d * (x - st['mean'])
                st['min'] = x if st['min'] is None else min(st['min'], x)
                st['max'] = x if st['max'] is None else max(st['max'], x)

        for c in cat_cols:
            if c in chunk.columns:
                uniq[c].update(chunk[c].dropna().unique().tolist())

    summary = pd.DataFrame({
        'attribute': cols,
        'missing_count': [int(missing[c]) for c in cols],
        'unique_count': [len(uniq[c]) if c in uniq else None for c in cols]
    })
    print("Missing e valori unici per colonna:")
    print(summary.sort_values('missing_count', ascending=False).head(50))

    rows = []
    for c, st in stats.items():
        n = st['count']
        std = math.sqrt(st['M2'] / (n - 1)) if n > 1 else 0.0
        rows.append({'attribute': c, 'count': n, 'mean': st['mean'],
                     'std': std, 'min': st['min'], 'max': st['max']})
    desc = pd.DataFrame(rows).set_index('attribute')
    print("Statistiche numeriche:")
    print(desc)

if __name__ == '__main__':
    main()