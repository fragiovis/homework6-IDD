import pandas as pd
import recordlinkage as rl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from pathlib import Path
import sys
import re
import math

def get_records_from_pairs(df):
    """Extracts unique records from a dataframe of pairs."""
    
    cols_a = {col: col.replace('_A', '') for col in df.columns if col.endswith('_A')}
    cols_b = {col: col.replace('_B', '') for col in df.columns if col.endswith('_B')}

    df_a = df[list(cols_a.keys())].rename(columns=cols_a)
    df_b = df[list(cols_b.keys())].rename(columns=cols_b)

    if 'id_A' not in df.columns or 'id_B' not in df.columns:
        raise ValueError("Columns 'id_A' or 'id_B' are missing.")
    
    df_a['id'] = df['id_A']
    df_b['id'] = df['id_B']

    # Combine and get unique records
    all_records = pd.concat([df_a, df_b]).drop_duplicates(subset='id').set_index('id')
    return all_records

def engineer_features(df):
    """Creates normalized and binned columns for comparison."""

    def normalize_string(s):
        if s is None: return ''
        s = str(s).lower().strip()
        s = re.sub(r'[^a-z0-9\s]', '', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def mileage_bin(x):
        try: v = float(x)
        except (ValueError, TypeError): return None
        if not math.isfinite(v): return None
        return int(v // 10000)

    # Ensure original columns are string type for normalization
    df['make'] = df['make'].astype(str)
    df['model'] = df['model'].astype(str)

    df['make_norm'] = df['make'].apply(normalize_string)
    df['model_norm'] = df['model'].apply(normalize_string)
    df['year_int'] = pd.to_numeric(df['year'], errors='coerce')
    df['mile_bin'] = df['mileage'].apply(mileage_bin)
    
    return df

def main(strategy='b1'):
    """
    Main function to run the Record Linkage pipeline for a given strategy.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'Dataset' / 'models' / strategy

    # Check if data exists
    if not data_path.exists():
        print(f"Error: Data directory not found for strategy '{strategy}' at {data_path}", file=sys.stderr)
        print("Please run 'scripts/prepare_datasets.py' first.", file=sys.stderr)
        return

    print(f"--- Running Record Linkage for Strategy: {strategy.upper()} ---")

    # 1. Load Data
    print("\n1. Loading data...")
    train_pairs_df = pd.read_csv(data_path / 'training' / f'{strategy}_train.csv')
    test_pairs_df = pd.read_csv(data_path / 'test' / f'{strategy}_test.csv')

    # Extract unique records for the library to use
    train_records = get_records_from_pairs(train_pairs_df)
    test_records = get_records_from_pairs(test_pairs_df)

    # Add feature engineering step
    print("  - Engineering features (normalization, binning)...")
    train_records = engineer_features(train_records)
    test_records = engineer_features(test_records)
    
    # Get the ground truth links and candidate pairs
    train_true_links = train_pairs_df.set_index(['id_A', 'id_B'])['label']
    train_candidate_links = train_pairs_df.set_index(['id_A', 'id_B']).index
    
    test_true_links = test_pairs_df.set_index(['id_A', 'id_B'])['label']
    test_candidate_links = test_pairs_df.set_index(['id_A', 'id_B']).index
    
    print(f"  - Loaded {len(train_records)} unique training records.")
    print(f"  - Loaded {len(test_records)} unique test records.")
    print(f"  - {len(train_candidate_links)} training pairs and {len(test_candidate_links)} test pairs.")

    # 2. Define Comparison Rules
    print("\n2. Defining comparison rules...")
    comparer = rl.Compare()
    comparer.string('model_norm', 'model_norm', method='jarowinkler', threshold=0.90, label='model')
    comparer.numeric('year_int', 'year_int', method='linear', scale=5, label='year')
    comparer.numeric('mile_bin', 'mile_bin', method='linear', scale=2, offset=0, label='mileage')
    comparer.exact('state', 'state', label='state')
    comparer.exact('make_norm', 'make_norm', label='make')
    print("  - Rules defined for: model, year, mileage, state, make.")

    # 3. Compute Feature Vectors
    print("\n3. Computing feature vectors...")
    train_features = comparer.compute(train_candidate_links, train_records)
    test_features = comparer.compute(test_candidate_links, test_records)
    print("  - Feature vectors computed for training and test sets.")
    print("\nSample of training features:")
    print(train_features.head())

    # 4. Train the Classifier
    print("\n4. Training the classifier...")
    classifier = LogisticRegression()
    classifier.fit(train_features, train_true_links)
    print("  - Logistic Regression classifier trained.")

    # 5. Evaluate on Test Set
    print("\n5. Evaluating on the test set...")
    test_predictions = classifier.predict(test_features)

    # Calculate and print metrics using sklearn
    conf_matrix = confusion_matrix(test_true_links, test_predictions)
    precision = precision_score(test_true_links, test_predictions)
    recall = recall_score(test_true_links, test_predictions)
    f1 = f1_score(test_true_links, test_predictions)

    print("\n--- Evaluation Results ---")
    print(f"Confusion Matrix:\n{conf_matrix}\n")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("--------------------------")

if __name__ == '__main__':
    # You can switch between 'b1' and 'b2' here
    main(strategy='b1')