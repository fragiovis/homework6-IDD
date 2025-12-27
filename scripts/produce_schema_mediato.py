import pandas as pd
from pathlib import Path

# crea cartella output se non esiste
out_dir = Path(__file__).resolve().parent.parent / 'Dataset' / 'processed'
out_dir.mkdir(parents=True, exist_ok=True)

# schema mediato finale
SCHEMA_MEDIATO = [
    'id','make','model','year','price','mileage','fuel','transmission',
    'drive','body_type','engine_cylinders','engine_displacement','state','description','vin'
]

def load_csv(path):
    """Carica CSV in chunk per risparmiare memoria"""
    return pd.read_csv(path, dtype=str, na_values=['', 'na', 'n/a', 'unknown', 'None'])

def save_dataset(df, name):
    """Salva dataset processato"""
    file_path = out_dir / f"{name}.csv"
    df.to_csv(file_path, index=False)
    print(f"Salvato {file_path}  shape={df.shape}")

def build_used_cars():
    """Mappa used_cars_data.csv -> schema mediato"""
    df = load_csv(Path(__file__).resolve().parent.parent / 'Dataset' / 'used_cars_data.csv')
    out = pd.DataFrame()
    out['id'] = df.get('listing_id')          # id tecnico
    out['make'] = df.get('franchise_make')
    out['model'] = df.get('model')
    out['year'] = pd.to_numeric(df.get('year'), errors='coerce')
    out['price'] = pd.to_numeric(df.get('price'), errors='coerce')
    out['mileage'] = pd.to_numeric(df.get('mileage'), errors='coerce')
    out['fuel'] = df.get('fuel_type')
    out['transmission'] = df.get('transmission')
    out['drive'] = df.get('wheel_system')
    out['body_type'] = df.get('body_type')
    out['engine_cylinders'] = df.get('engine_cylinders')
    out['engine_displacement'] = pd.to_numeric(df.get('engine_displacement'), errors='coerce')
    out['state'] = None                       # non presente -> NaN
    out['description'] = df.get('description')
    out['vin'] = df.get('vin')
    return out[SCHEMA_MEDIATO]

def build_vehicles():
    """Mappa vehicles.csv -> schema mediato"""
    df = load_csv(Path(__file__).resolve().parent.parent / 'Dataset' / 'vehicles.csv')
    out = pd.DataFrame()
    out['id'] = df.get('id')
    out['make'] = df.get('manufacturer')
    out['model'] = df.get('model')
    out['year'] = pd.to_numeric(df.get('year'), errors='coerce')
    out['price'] = pd.to_numeric(df.get('price'), errors='coerce')
    out['mileage'] = pd.to_numeric(df.get('odometer'), errors='coerce')
    out['fuel'] = df.get('fuel')
    out['transmission'] = df.get('transmission')
    out['drive'] = df.get('drive')
    out['body_type'] = df.get('type')
    out['engine_cylinders'] = df.get('cylinders')
    out['engine_displacement'] = None         # non presente -> NaN
    out['state'] = df.get('state')
    out['description'] = df.get('description')
    out['vin'] = df.get('VIN')
    return out[SCHEMA_MEDIATO]

if __name__ == '__main__':
    print("Inizio trasformazione...")
    df_a = build_used_cars()
    save_dataset(df_a, 'used_cars_data_mediato')
    df_b = build_vehicles()
    save_dataset(df_b, 'vehicles_mediato')
    print("Fatto.")