# 🚗 Used-Car Listings — Data Integration & Record Linkage

*Data Engineering course (Ingegneria dei Dati), homework 6 · Università degli Studi Roma Tre (2025/26)*
*Team: Alessandro Schmitt, Alessandro Cavina, Francesco Giovanardi, Michele Guida*

End-to-end **entity resolution** between two heterogeneous sources of US used-car listings: the US Used Cars dataset (`used_cars_data.csv`, source A) and the Craigslist vehicles dataset (`vehicles.csv`, source B). The goal is to recognise when two listings describe the same vehicle.

## Pipeline
1. **Exploratory data analysis** (`eda_used_cars.py`, `eda_vehicles.py`, `eda_mediato.py`): missing values and cardinality of every attribute, used to decide which features are useful for blocking and matching. Results: `formattato_dataset_largo.txt`, `formattato_dataset_piccolo.txt`.
2. **Mediated schema** (`produce_schema_mediato.py`): both sources are mapped onto a shared schema (make, model, year, price, mileage, fuel, transmission, drive, body type, engine, state, description). The design rationale is in [`latex/schema_mediato.tex`](latex/schema_mediato.tex) (Italian).
3. **Ground truth from VINs** (`produce_gt_b1.py`, `produce_gt_b2.py`): listings sharing a cleaned 17-character VIN are labelled as matches, and non-matching pairs are sampled from the same block. The VIN is then dropped from the features to avoid leakage. Records are streamed in chunks into an SQLite index, so the large CSVs never have to fit in memory.
4. **Two blocking strategies**
   - **B1**: normalised make + model;
   - **B2**: make + model + year + mileage bin (tighter blocks, fewer candidate pairs).
5. **Train/test split** (`prepare_datasets.py`).
6. **Matching**, with two approaches:
   - **`recordlinkage` + logistic regression** (`run_recordlinkage.py`): Jaro-Winkler similarity on the model name, linear similarity on year and mileage, exact match on make and state; evaluated with precision, recall and F1;
   - **`dedupe`** (`run_dedupe_linkage.py`): an active-learning record linker trained on the labelled pairs (`String`, `ShortString` and `Price` variables); trained settings are cached in `scripts/`.

## Running it
The raw datasets are not versioned. Download them from Kaggle and place them in `Dataset/` as `used_cars_data.csv` and `vehicles.csv`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/produce_schema_mediato.py
python scripts/produce_gt_b1.py
python scripts/produce_gt_b2.py
python scripts/prepare_datasets.py       # train/test split for both strategies
python scripts/run_recordlinkage.py
python scripts/run_dedupe_linkage.py b1  # or b2
```
`run_recordlinkage.py` uses blocking strategy `b1` by default; switch to `b2` in its `main(strategy=...)` call.

## Tech stack
Python · pandas · NumPy · SQLite · recordlinkage · dedupe · scikit-learn · LaTeX
