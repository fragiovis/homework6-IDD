import os
import re
import logging
import optparse
import json
import pandas as pd
from pathlib import Path
import dedupe

def preProcess(column):
    """
    Clean and standardize a data column.
    """
    if pd.isna(column):
        return None
    column = str(column)
    column = re.sub('  +', ' ', column)
    column = re.sub('\n', ' ', column)
    column = column.strip().strip('\"').strip("'").lower().strip()
    return column if column else None

def prepare_data_for_dedupe(pairs_filepath):
    """
    Reads a CSV of paired records and prepares it for Dedupe.
    It creates two dictionaries of records (data_1, data_2) and a set of true links.
    """
    data_1 = {}
    data_2 = {}
    true_links = set()

    df_pairs = pd.read_csv(pairs_filepath, dtype=str)

    cols_a = {col: col.replace('_A', '') for col in df_pairs.columns if col.endswith('_A')}
    cols_b = {col: col.replace('_B', '') for col in df_pairs.columns if col.endswith('_B')}

    numeric_fields = {'price', 'mileage'}

    for _, row in df_pairs.iterrows():
        id_a, id_b = row['id_A'], row['id_B']

        record_a = {}
        for old_col, new_col in cols_a.items():
            value = row[old_col]
            if new_col in numeric_fields:
                try:
                    record_a[new_col] = float(value) if not pd.isna(value) else None
                except (ValueError, TypeError):
                    record_a[new_col] = None
            else:
                record_a[new_col] = preProcess(value)

        record_b = {}
        for old_col, new_col in cols_b.items():
            value = row[old_col]
            if new_col in numeric_fields:
                try:
                    record_b[new_col] = float(value) if not pd.isna(value) else None
                except (ValueError, TypeError):
                    record_b[new_col] = None
            else:
                record_b[new_col] = preProcess(value)
        
        data_1[id_a] = record_a
        data_2[id_b] = record_b

        if row['label'] == '1':
            true_links.add((id_a, id_b))

    return data_1, data_2, true_links

def main(strategy='b1'):
    """
    Main function to run the Dedupe Record Linkage pipeline.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / 'Dataset' / 'models' / strategy
    settings_file = project_root / 'scripts' / f'dedupe_{strategy}_learned_settings'
    training_file = project_root / 'scripts' / f'dedupe_{strategy}_training.json'

    print(f"--- Running Dedupe Record Linkage for Strategy: {strategy.upper()} ---")

    # 1. Load and Prepare Training Data
    print("\n1. Loading and preparing training data...")
    train_path = data_path / 'training' / f'{strategy}_train.csv'
    data_1, data_2, train_true_links = prepare_data_for_dedupe(train_path)

    # Format the ground truth for Dedupe's training
    training_pairs = {'match': list(train_true_links), 'distinct': []}
    
    # We need to add some distinct pairs for training. Let's sample a few.
    all_ids_1 = list(data_1.keys())
    all_ids_2 = list(data_2.keys())
    num_distinct_to_sample = min(len(train_true_links) * 2, 10000)
    
    i = 0
    while i < num_distinct_to_sample:
        id1 = all_ids_1[i % len(all_ids_1)]
        id2 = all_ids_2[i % len(all_ids_2)]
        if (id1, id2) not in train_true_links:
            training_pairs['distinct'].append((id1, id2))
            i += 1

    print(f"  - Loaded {len(data_1)} records for dataset 1.")
    print(f"  - Loaded {len(data_2)} records for dataset 2.")
    print(f"  - Using {len(training_pairs['match'])} matching pairs and {len(training_pairs['distinct'])} distinct pairs for training.")

    # 2. Define Dedupe Model Fields
    fields = [
        dedupe.variables.String('model'),
        dedupe.variables.String('make'),
        dedupe.variables.ShortString('year', has_missing=True),
        dedupe.variables.Price('price', has_missing=True),
        dedupe.variables.Price('mileage', has_missing=True),
        dedupe.variables.ShortString('state', has_missing=True),
    ]

    linker = dedupe.RecordLink(fields, num_cores=0)

    # 3. Train the Model
    loaded = False
    if os.path.exists(settings_file) and os.path.getsize(settings_file) > 0:
        print('\n3. Loading trained model from', settings_file)
        try:
            with open(settings_file, 'rb') as f:
                linker = dedupe.StaticRecordLink(f)
            loaded = True
        except Exception:
            print('  - Settings file invalid, deleting and retraining...')
            try:
                os.remove(settings_file)
            except Exception:
                pass
            loaded = False
    if not loaded:
        print('\n3. Training new model...')

        # Build training samples that include all labeled matches to avoid losing them
        SAMPLE_SIZE = 3000
        import random

        ids1_match = {a for (a, b) in train_true_links}
        ids2_match = {b for (a, b) in train_true_links}

        # Start with all match ids, then add random extras up to SAMPLE_SIZE
        data_1_sample = {k: data_1[k] for k in ids1_match if k in data_1}
        if len(data_1_sample) < SAMPLE_SIZE and len(data_1) > len(data_1_sample):
            remaining = [k for k in data_1.keys() if k not in data_1_sample]
            add_n = min(SAMPLE_SIZE - len(data_1_sample), len(remaining))
            if add_n > 0:
                extras = random.sample(remaining, add_n)
                for k in extras:
                    data_1_sample[k] = data_1[k]

        data_2_sample = {k: data_2[k] for k in ids2_match if k in data_2}
        if len(data_2_sample) < SAMPLE_SIZE and len(data_2) > len(data_2_sample):
            remaining = [k for k in data_2.keys() if k not in data_2_sample]
            add_n = min(SAMPLE_SIZE - len(data_2_sample), len(remaining))
            if add_n > 0:
                extras = random.sample(remaining, add_n)
                for k in extras:
                    data_2_sample[k] = data_2[k]

        print(f"  - Sampling dataset 1 to {len(data_1_sample)} records for training.")
        print(f"  - Sampling dataset 2 to {len(data_2_sample)} records for training.")

        # Build labeled examples as record dict pairs, filtered to sampled ids
        match_pairs = [
            (data_1_sample[a], data_2_sample[b])
            for (a, b) in train_true_links
            if a in data_1_sample and b in data_2_sample
        ]

        # Create distinct pairs from sampled ids that are not known matches
        distinct_pairs = []
        want_distinct = min(len(match_pairs) * 2, 5000)
        ids1_list = list(data_1_sample.keys())
        ids2_list = list(data_2_sample.keys())
        known = train_true_links
        tries = 0
        while len(distinct_pairs) < want_distinct and tries < want_distinct * 10:
            a = random.choice(ids1_list)
            b = random.choice(ids2_list)
            if (a, b) not in known:
                distinct_pairs.append((data_1_sample[a], data_2_sample[b]))
            tries += 1

        labeled_examples = {"match": match_pairs, "distinct": distinct_pairs}
        print(f"  - Preparing training with {len(match_pairs)} match and {len(distinct_pairs)} distinct pairs...")

        # Write labeled examples to JSON and load via prepare_training
        with open(training_file, 'w') as tf:
            json.dump(labeled_examples, tf)
        with open(training_file, 'r') as tf:
            linker.prepare_training(data_1_sample, data_2_sample, training_file=tf, sample_size=5000)

        linker.train(index_predicates=False)

        with open(settings_file, 'wb') as sf:
            linker.write_settings(sf)
        with open(training_file, 'w') as tf:
            linker.write_training(tf)

    # 4. Evaluate on Test Set
    print("\n4. Evaluating on the test set...")
    test_path = data_path / 'test' / f'{strategy}_test.csv'
    test_data_1, test_data_2, test_true_links = prepare_data_for_dedupe(test_path)

    print("  - Finding linked pairs in test set...")
    linked_records = linker.join(test_data_1, test_data_2, 0.5)

    # 5. Calculate Metrics
    print("\n5. Calculating evaluation metrics...")
    
    predicted_links = set()
    for (id_1, id_2), score in linked_records:
        predicted_links.add((id_1, id_2))

    true_positives = len(test_true_links.intersection(predicted_links))
    false_positives = len(predicted_links - test_true_links)
    false_negatives = len(test_true_links - predicted_links)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\n--- Evaluation Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("--------------------------")

if __name__ == '__main__':
    import sys

    # Setup logging
    optp = optparse.OptionParser()
    optp.add_option('-v', '--verbose', dest='verbose', action='count',
                    help='Increase verbosity (specify multiple times for more)')
    (opts, args) = optp.parse_args()
    log_level = logging.WARNING
    if opts.verbose:
        if opts.verbose == 1:
            log_level = logging.INFO
        elif opts.verbose >= 2:
            log_level = logging.DEBUG
    logging.getLogger().setLevel(log_level)

    strategy = 'b1'
    # The first argument after the script name is the strategy
    if len(args) > 0 and args[0] in ['b1', 'b2']:
        strategy = args[0]

    main(strategy=strategy)
