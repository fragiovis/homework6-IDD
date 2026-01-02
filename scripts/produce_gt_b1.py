import sys, subprocess
from pathlib import Path

def ensure_deps():
    # Verifica dipendenze minime per eseguire lo script in ambienti puliti
    try:
        import pandas
        import numpy
    except Exception:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pandas', 'numpy'])

def load_csv(path):
    # Lettura robusta: tutto come stringa per evitare inferenze di tipo costose
    # e gestire valori mancanti/rumorosi senza errori
    import pandas as pd
    return pd.read_csv(path, dtype=str, na_values=['', 'na', 'n/a', 'unknown', 'None'], keep_default_na=True, low_memory=False)

def norm_str(s):
    # Normalizzazione testuale aggressiva per chiavi di blocking
    # - lowercase, trimming, rimozione non alfanumerici
    # - compattazione spazi
    import re
    if s is None:
        return None
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s != '' else None

def clean_vin(v):
    # Pulizia VIN: tieni solo A-Z/0-9, uppercase, accetta solo lunghezza 17
    # Serve per unire positivi affidabili e deduplicare
    import re
    if v is None:
        return None
    v = re.sub(r"[^A-Za-z0-9]", "", str(v)).upper()
    return v if len(v) == 17 else None

def key_b1(make, model):
    # Blocking B1: chiave basata su make e model normalizzati
    return f"{make}|{model}"

def build_b_index(out_dir, b_path):
    # Crea un indice SQLite per il dataset B con chiave di blocco e VIN
    import sqlite3
    import pandas as pd
    db_path = out_dir / 'b1_index.sqlite'
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
        chunk['block'] = chunk.apply(lambda r: key_b1(r['make_norm'], r['model_norm']), axis=1)
        rows = [
            (
                r['vin_clean'], r['id'], r['make'], r['model'], r['year'], r['price'], r['mileage'],
                r['fuel'], r['transmission'], r['drive'], r['body_type'],
                r['engine_cylinders'], r.get('engine_displacement'),
                r['state'], r['description'], r['make_norm'], r['model_norm'], r['block']
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
                make_norm, model_norm, block
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
    out_path = out_dir / 'gt_b1.csv'
    header = [
        'id_A','id_B','make_A','make_B','model_A','model_B','year_A','year_B',
        'price_A','price_B','mileage_A','mileage_B','fuel_A','fuel_B',
        'transmission_A','transmission_B','drive_A','drive_B','body_type_A','body_type_B',
        'engine_cylinders_A','engine_cylinders_B','engine_displacement_A','engine_displacement_B',
        'state_A','state_B','description_A','description_B','vin_A','vin_B','label'
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
        chunk['block'] = chunk.apply(lambda r: key_b1(r['make_norm'], r['model_norm']), axis=1)
        for _, ra in chunk.iterrows():
            if pos_count < MAX_POS:
                cur.execute("SELECT id, vin_clean, make, model, year, price, mileage, fuel, transmission, drive, body_type, engine_cylinders, engine_displacement, state, description FROM b_index WHERE vin_clean=?", (ra['vin_clean'],))
                rb = cur.fetchone()
                if rb:
                    row = [
                        ra['id'], rb[0], ra['make'], rb[2], ra['model'], rb[3], ra['year'], rb[4],
                        ra['price'], rb[5], ra['mileage'], rb[6], ra['fuel'], rb[7],
                        ra['transmission'], rb[8], ra['drive'], rb[9], ra['body_type'], rb[10],
                        ra['engine_cylinders'], rb[11], ra.get('engine_displacement'), rb[12],
                        ra['state'], rb[13], ra['description'], rb[14], ra['vin_clean'], rb[1], 1
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
                        ra['state'], rb[13], ra['description'], rb[14], ra['vin_clean'], rb[1], 0
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

if __name__ == '__main__':
    main()
