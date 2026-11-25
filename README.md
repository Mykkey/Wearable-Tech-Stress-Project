# Predicting Daily Stress Level from Wearable Data (CT4101 Project)

## Project
This project tries to predict a person's daily stress level (Low, Medium or High) from lifestyle habits and smartwatch-style measurements. It is a multi-class classification problem. A Dummy baseline is compared against Logistic Regression, a Decision Tree, a Random Forest and Gradient Boosting.

## Dataset
Wearable Tech: Sleep Quality & Stress, from Kaggle (CC0 licence). 10,000 rows. The data appears to be synthetic, so the results do not describe real people. The CSV is included in `data/`, and `data/README.md` has the source, licence and citation.

## Structure
- `notebooks/01_stress_prediction.ipynb` - the full workflow: data audit, preprocessing pipeline, train/test split, baseline, model comparison, tuning, final test evaluation, sensitivity and fairness checks, and a summary of results
- `data/` - the dataset and its source/licence
- `requirements.txt` - the packages needed

## Setup
Tested on Python 3.14.7 (Windows 11). Run these from the project folder.

Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
macOS / Linux:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Versions used when testing: pandas 3.0.6, numpy 2.5.4, scikit-learn 1.9.1, matplotlib 3.11.2.

## Run
Open the notebook and choose Restart and Run All:
```
jupyter notebook notebooks/01_stress_prediction.ipynb
```
Or run it without opening it:
```
jupyter nbconvert --to notebook --execute --inplace notebooks/01_stress_prediction.ipynb
```
It takes a few minutes to run, mostly because of the tuning section. In VS Code, open the notebook and select the `.venv` Python as the kernel.

## Main results
Macro-F1 is the main metric, because High stress is only about 4% of the data.

| Model | Cross-validated Macro-F1 (training set) |
|---|---|
| Dummy baseline | 0.244 |
| Decision Tree (hand-picked settings) | 0.561 |
| Logistic Regression (tuned) | 0.620 |
| Gradient Boosting (tuned) | 0.621 |
| Random Forest (tuned) | 0.634 |

The tuned Random Forest was chosen as the final model. On the held-out test set it scores a Macro-F1 of 0.62 (baseline 0.244), with a High stress recall of 0.40.

## Main limitation
High stress is only found about 40% of the time, and the dataset appears synthetic, so this model should not be used to make decisions about real people.

## Repository
https://github.com/Mykkey/Wearable-Tech-Stress-Project

## AI Assistance Declaration
Claude Code was used throughout this project. It was used for suggestions, planning, and wording. Where Claude Code's suggestions were used, I rewrote and ran the code, and edited explanations. Claude Code was also used to help with setup files and No other external code or collaboration was used.