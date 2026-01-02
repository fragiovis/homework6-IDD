import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

def main():
    """
    This script prepares the ground truth datasets for the model training phase.
    It performs the following steps for each blocking strategy (b1, b2):
    1. Reads the ground truth file (e.g., gt_b1.csv).
    2. Removes the VIN columns to prevent data leakage.
    3. Splits the data into stratified training (70%), validation (15%), and test (15%) sets.
    4. Saves the resulting sets into structured directories under Dataset/models/.
       The filenames will include the blocking strategy (e.g., b1_train.csv).
    """
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / 'Dataset' / 'processed'
    models_dir = project_root / 'Dataset' / 'models'

    ground_truths = {
        'b1': processed_dir / 'gt_b1.csv',
        'b2': processed_dir / 'gt_b2.csv'
    }

    print("Starting dataset preparation...")

    for strategy, gt_path in ground_truths.items():
        if not gt_path.exists():
            print(f"Error: Ground truth file not found at {gt_path}", file=sys.stderr)
            continue

        print(f"\nProcessing strategy '{strategy}' from {gt_path.name}...")

        # 1. Read and clean the dataset
        df = pd.read_csv(gt_path, dtype=str)
        
        vin_cols = [col for col in df.columns if 'vin' in col.lower()]
        if vin_cols:
            df = df.drop(columns=vin_cols)
            print(f"  - Removed columns: {', '.join(vin_cols)}")
        else:
            print("  - No VIN columns to remove.")

        if 'label' not in df.columns:
            print(f"Error: 'label' column not found in {gt_path.name}", file=sys.stderr)
            continue
        df['label'] = df['label'].astype(int)

        # 2. Split data (70% train, 15% validation, 15% test)
        train_df, temp_df = train_test_split(
            df,
            test_size=0.30,
            random_state=42,
            stratify=df['label']
        )

        valid_df, test_df = train_test_split(
            temp_df,
            test_size=0.50,
            random_state=42,
            stratify=temp_df['label']
        )
        
        print(f"  - Data split completed:")
        print(f"    - Training set size:   {len(train_df)}")
        print(f"    - Validation set size: {len(valid_df)}")
        print(f"    - Test set size:       {len(test_df)}")

        # 3. Create directories and save files with strategy in the name
        output_base_path = models_dir / strategy
        
        paths = {
            'train': output_base_path / 'training',
            'valid': output_base_path / 'validation',
            'test': output_base_path / 'test'
        }
        
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
            
        train_df.to_csv(paths['train'] / f'{strategy}_train.csv', index=False)
        valid_df.to_csv(paths['valid'] / f'{strategy}_valid.csv', index=False)
        test_df.to_csv(paths['test'] / f'{strategy}_test.csv', index=False)
        
        print(f"  - Files saved successfully in {output_base_path}/ with names like '{strategy}_train.csv'")

    print("\nDataset preparation complete.")

if __name__ == '__main__':
    main()