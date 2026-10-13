# Predicting Daily Stress Level from Wearable Data (CT4101 Project)

Multi-class classification (Low / Medium / High stress) on the Kaggle "Wearable Tech: Sleep Quality & Stress" dataset (see `data/README.md`).

## Setup (Python 3.10+)
```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Run
```
jupyter notebook notebooks/01_data_audit_and_baseline.ipynb
```
Then Kernel > Restart & Run All. Or headless:
```
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_audit_and_baseline.ipynb
```

## Structure
- `notebooks/01_data_audit_and_baseline.ipynb` - audit, pipeline, split, baseline
- `data/` - dataset + source/licence
- `requirements.txt`

## Repository
https://github.com/Mykkey/Wearable-Tech-Stress-Project