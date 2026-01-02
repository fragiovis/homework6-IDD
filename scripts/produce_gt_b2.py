import sys, subprocess
from pathlib import Path

def ensure_deps():
    # Garantisce dipendenze minime per ambienti puliti
    try:
        import pandas
        import numpy
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'numpy'])

def load_csv(path):
    # Lettura come stringhe per controllo totale sui tipi e NA
    import pandas as pd
    return pd.read_csv(path, dtype=str, na_values=['', 'na', 'n/a', 'unknown', 'None'], keep_default_na=True, low_memory=False)

def norm_str(s):
    # Normalizzazione testuale per chiavi di blocking
    import re
    if s is None:
        return None
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s != '' else None

def clean_vin(v):
    # Pulizia VIN: mantieni 17 caratteri alfanumerici uppercase
    import re
    if v is None:
        return None
    v = re.sub(r"[^A-Za-z0-9]", "", str(v)).upper()
    return v if len(v) == 17 else None

def mileage_bin(x):
    import math
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return int(v // 10000)

def key_b2(make, model, year, mile_bin):
    # Blocking B2: make+model+year+mileage_bin
    return f"{make}|{model}|{year}|{mile_bin}"

def build_b_index(out_dir, b_path):
    # Crea un indice SQLite per B con chiave di blocco B2
    import sqlite3
    import pandas as pd
    db_path = out_dir / 'b2_index.sqlite'
    if db_path.exists():
        db_path.unlink()
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE b_index (
            vin_clean TEXT PRIMARY KEY,
            id TEXT,
            make TEXT, model TEXT, year TEXT, price TEXT, mileage TEXT,
            fuel TEXT, transmission TEXT, drive TEXT, body_type TEXT,
            engine_cylinders TEXT, engine_displacement TEXT,
            state TEXT, description TEXT,
            make_norm TEXT, model_norm TEXT,
            year_int INTEGER, mile_bin INTEGER,
            block TEXT
        )
        """
    )
    chunksize = 100_000
    for chunk in pd.read_csv(b_path, dtype=str, na_values=['', 'na', 'n/a', 'unknown', 'None'], keep_default_na=True, low_memory=False, chunksize=chunksize):
        chunk['vin_clean'] = chunk['vin'].map(clean_vin)
        chunk = chunk.dropna(subset=['vin_clean']).drop_duplicates(subset=['vin_clean'])
        chunk['make_norm'] = chunk['make'].map(norm_str)
        chunk['model_norm'] = chunk['model'].map(norm_str)
        chunk['year_int'] = pd.to_numeric(chunk['year'], errors='coerce')
        chunk['mile_bin'] = chunk['mileage'].map(mileage_bin)
        chunk['block'] = chunk.apply(lambda r: key_b2(r['make_norm'], r['model_norm'], r['year_int'], r['mile_bin']), axis=1)
        rows = [
            (
                r['vin_clean'], r['id'], r['make'], r['model'], r['year'], r['price'], r['mileage'],
                r['fuel'], r['transmission'], r['drive'], r['body_type'],
                r['engine_cylinders'], r.get('engine_displacement'),
                r['state'], r['description'], r['make_norm'], r['model_norm'],
                r['year_int'], r['mile_bin'], r['block']
            )
            for _, r in chunk.iterrows()
        ]
        cur.executemany(
            """
            INSERT OR REPLACE INTO b_index (
                vin_clean, id, make, model, year, price, mileage,
                fuel, transmission, drive, body_type,
                engine_cylinders, engine_displacement,
                state, description,
                make_norm, model_norm,
                year_int, mile_bin, block
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            rows
        )
        conn.commit()
    cur.execute("CREATE INDEX idx_block ON b_index(block)")
    conn.commit()
    return conn

def main():
    ensure_deps()
    import pandas as pd
    import sqlite3, csv
    base = Path(__file__).resolve().parent.parent
    out_dir = base / 'Dataset' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    a_path = out_dir / 'used_cars_data_mediato.csv'
    b_path = out_dir / 'vehicles_mediato.csv'
    # Costruisci indice SQLite per B (streaming)
    conn = build_b_index(out_dir, b_path)
    cur = conn.cursor()
    MAX_POS = 200_000
    MAX_NEG = 200_000
    pos_count = 0
    neg_count = 0
    out_path = out_dir / 'gt_b2.csv'
    header = [
        'id_A','id_B','make_A','make_B','model_A','model_B','year_A','year_B',
        'price_A','price_B','mileage_A','mileage_B','fuel_A','fuel_B',
        'transmission_A','transmission_B','drive_A','drive_B','body_type_A','body_type_B',
        'engine_cylinders_A','engine_cylinders_B','engine_displacement_A','engine_displacement_B',
        'state_A','state_B','description_A','description_B','label'
    ]
    f = open(out_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(header)
    chunksize = 100_000
    for chunk in pd.read_csv(a_path, dtype=str, na_values=['', 'na', 'n/a', 'unknown', 'None'], keep_default_na=True, low_memory=False, chunksize=chunksize):
        chunk['vin_clean'] = chunk['vin'].map(clean_vin)
        chunk = chunk.dropna(subset=['vin_clean']).drop_duplicates(subset=['vin_clean'])
        chunk['make_norm'] = chunk['make'].map(norm_str)
        chunk['model_norm'] = chunk['model'].map(norm_str)
        chunk['year_int'] = pd.to_numeric(chunk['year'], errors='coerce')
        chunk['mile_bin'] = chunk['mileage'].map(mileage_bin)
        chunk['block'] = chunk.apply(lambda r: key_b2(r['make_norm'], r['model_norm'], r['year_int'], r['mile_bin']), axis=1)
        for _, ra in chunk.iterrows():
            if pos_count < MAX_POS:
                cur.execute("SELECT id, make, model, year, price, mileage, fuel, transmission, drive, body_type, engine_cylinders, engine_displacement, state, description FROM b_index WHERE vin_clean=?", (ra['vin_clean'],))
                rb = cur.fetchone()
                if rb:
                    row = [
                        ra['id'], rb[0], ra['make'], rb[1], ra['model'], rb[2], ra['year'], rb[3],
                        ra['price'], rb[4], ra['mileage'], rb[5], ra['fuel'], rb[6],
                        ra['transmission'], rb[7], ra['drive'], rb[8], ra['body_type'], rb[9],
                        ra['engine_cylinders'], rb[10], ra.get('engine_displacement'), rb[11],
                        ra['state'], rb[12], ra['description'], rb[13], 1
                    ]
                    writer.writerow(row)
                    pos_count += 1
            if neg_count < MAX_NEG:
                cur.execute("SELECT id, vin_clean, make, model, year, price, mileage, fuel, transmission, drive, body_type, engine_cylinders, engine_displacement, state, description FROM b_index WHERE block=? LIMIT 5", (ra['block'],))
                for rb in cur.fetchall():
                    if rb[1] == ra['vin_clean']:
                        continue
                    row = [
                        ra['id'], rb[0], ra['make'], rb[2], ra['model'], rb[3], ra['year'], rb[4],
                        ra['price'], rb[5], ra['mileage'], rb[6], ra['fuel'], rb[7],
                        ra['transmission'], rb[8], ra['drive'], rb[9], ra['body_type'], rb[10],
                        ra['engine_cylinders'], rb[11], ra.get('engine_displacement'), rb[12],
                        ra['state'], rb[13], ra['description'], rb[14], 0
                    ]
                    writer.writerow(row)
                    neg_count += 1
                    if neg_count >= MAX_NEG:
                        break
        if pos_count >= MAX_POS and neg_count >= MAX_NEG:
            break
    f.close()
    conn.close()
    print(f"Salvato {out_path} pos={pos_count} neg={neg_count}")
    import pandas as pd
    total_rows = 0
    cols_count = None
    for chunk in pd.read_csv(out_path, dtype=str, low_memory=False, chunksize=100_000):
        if cols_count is None:
            cols_count = len(chunk.columns)
        total_rows += len(chunk)
    print(f"Shape gt_b2: ({total_rows}, {cols_count})")

if __name__ == '__main__':
    main()
