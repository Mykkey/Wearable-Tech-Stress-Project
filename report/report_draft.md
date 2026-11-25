# Predicting Stress Through Wearable Tech Data
## CT4101 Machine Learning Portfolio Project: Final Report

Repository: https://github.com/Mykkey/Wearable-Tech-Stress-Project

---

## 1. Introduction

This project predicts a person's daily stress level (Low, Medium or High) from lifestyle habits and smartwatch-style measurements such as sleep, screen time and caffeine intake. It is a multi-class classification problem.

The aim was not to get the highest score at any cost. I wanted every choice in the workflow to have a reason behind it: a fair baseline, models picked for stated reasons, settings chosen from evidence instead of assumed, one clean test evaluation, and an honest look at where the model fails. The main question is whether lifestyle data can tell stress levels apart at all, and in particular whether it can find the High stress cases, because those are the ones where a prediction would matter most.

## 2. Data

I used the Kaggle dataset "Wearable Tech: Sleep Quality & Stress" (CC0 licence), which has 10,000 rows and 16 columns. The CSV is included in the repository, and ```data/README.md``` has the source, licence and citation.

The target is ```Daily_Stress_Level```. It is heavily imbalanced, with roughly 58% Low, 38% Medium and only 4% High. This affects almost every later decision. A model that always says Low is right 58% of the time, so accuracy alone would make a useless model look decent. Because of this, Macro-F1 is the main metric. It averages the F1 score of the three classes equally, so a model cannot score well by ignoring High stress.

The audit showed that ```BMI```, ```Caffeine_Intake_mg```, ```Resting_Heart_Rate``` and ```Daily_Sleep_Hours``` each have around 200 missing values. That affects about 8% of rows and 2% of all values. There are no duplicated rows. The histograms show a spike at the lowest allowed value in ```BMI```, ```Caffeine_Intake_mg``` and ```Resting_Heart_Rate```, and ```Age``` is spread almost uniformly with evenly spaced drops. This points to synthetic data, where a number is generated randomly and clipped to a minimum. This is the biggest limitation of the project and I come back to it in Section 7.

Two columns were dropped before modelling. ```User_ID``` has no predictive meaning. ```Sleep_Quality_Score``` is recorded on the same day as stress, so it risks leakage. I did not just assume this was the right call, and I tested it later as a sensitivity check (Section 6.4).

## 3. Methodology

### 3.1 Split and leakage

The data was split once into 80% training (8,000 rows) and 20% test (2,000 rows). The split is stratified on the target so both parts keep the same class proportions, which matters because High stress is so rare. A fixed seed (136) is used everywhere so the results can be reproduced. The test set was used once, at the end, and no decisions were made from it. All model comparison and tuning used stratified 5-fold cross-validation on the training set only.

All preprocessing is inside a scikit-learn ```Pipeline```. Numeric columns (10 of them) get median imputation then standardisation, and categorical columns (```Gender```, ```Physical_Activity_Level``` and ```Diet_Type```) get most-frequent imputation then one-hot encoding. Putting this in a pipeline means the medians, means and categories are learned from the training part of each fold only. Preprocessing the whole dataset first would let information from the held-out data leak into training and make the scores look better than they are.

### 3.2 Metrics

Macro-F1 is the main metric. I also report balanced accuracy, accuracy, ROC-AUC (one-vs-rest), per-class precision and recall, and the confusion matrix. I pay the most attention to High stress recall, because missing a High stress case is the costly mistake.

### 3.3 Models

Four learned models were compared against a dummy baseline that always predicts Low. I chose them for different reasons:

- ```Logistic Regression``` is a simple straight-line model that is fast and easy to explain. It shows how much can be learned without anything complicated.
- ```Decision Tree``` can find non-straight-line patterns and is easy to explain, but it overfits easily, so it shows what happens when complexity is not controlled.
- ```Random Forest``` averages many trees trained on random parts of the data, which stops it overfitting as badly as one tree.
- ```Gradient Boosting``` builds trees one after another, each correcting the last, and is often strong on tabular data. It was not in my proposal. I added it to check that I was not missing an easy improvement.

## 4. Baseline

The baseline predicts the most common class (Low) every time. It is correct about 58% of the time, which looks decent but is misleading, because it never detects Medium or High (recall 0 for both). Its Macro-F1 is about 0.244, its balanced accuracy is 0.333 and its ROC-AUC is about 0.5, which is chance level. These are the scores the learned models have to beat.

## 5. Model Comparison and Tuning

### 5.1 First comparison

First I ran the three models from my proposal with hand-picked settings (```class_weight = "balanced"```, a ```max_depth``` of 5 for the tree, and 200 trees with a ```max_depth``` of 10 for the forest), using the same pipeline and the same cross-validation. All three clearly beat the baseline. Random Forest had the best Macro-F1 (0.624) and accuracy (0.724). Logistic Regression was close on Macro-F1 (0.599) and had the best balanced accuracy (0.705) and ROC-AUC (0.857). The Decision Tree was weakest (0.561).

### 5.2 Error analysis on the training set

Using cross-validated predictions from the Random Forest, High stress is the weakest class, with a precision of only 0.37 and a recall of 0.49. Only 340 training rows are High, which makes it hard to learn. Most missed High cases are predicted as Medium (169 of 340) and almost none as Low (6). Low and Medium are confused with each other (1,069 Low predicted as Medium and 676 Medium predicted as Low), which suggests the classes overlap in the features. Low (F1 0.80) and Medium (F1 0.66) perform reasonably.

### 5.3 Does class weighting help?

I scored each model with and without ```class_weight```, because I did not want to assume ```balanced``` was best.

| Model | class_weight | Macro-F1 | High recall |
|---|---|---|---|
| Logistic Regression | None | 0.615 | 0.241 |
| Logistic Regression | balanced | 0.599 | 0.776 |
| Decision Tree | None | 0.533 | 0.085 |
| Decision Tree | balanced | 0.561 | 0.674 |
| Random Forest | None | 0.542 | 0.079 |
| Random Forest | balanced | 0.624 | 0.485 |

Without weights the models mostly ignore High stress (recall 0.08 to 0.24). Weights helped the Decision Tree and Random Forest on both Macro-F1 and High recall. For Logistic Regression it is a trade-off, because Macro-F1 is a bit lower with weights (0.599 compared to 0.615) but High recall goes from 0.24 to 0.78. Missing a High stress case is the costly mistake, so ```class_weight``` was included as a setting in the tuning search instead of being fixed.

### 5.4 Validation and learning curves

Each validation curve changes one setting and shows the training score and the cross-validated score. If both are low, the model is too simple (underfitting). If training is high but cross-validated is much lower, the model has memorised the training data (overfitting). The best setting is where the cross-validated line peaks. Differences smaller than the shaded spread between folds do not mean much.

- ```Logistic Regression``` ```C```: scores rise from about 0.48 at 0.001 to about 0.62 at 1 and then stay flat. Training and cross-validated lines are almost the same, so it is not overfitting. It is limited because it is a straight-line model.
- ```Decision Tree``` ```max_depth```: the cross-validated score peaks around depth 5 (about 0.56) and then gets worse while the training score climbs to 1.0. That is overfitting, so a depth of 5 is used.
- ```Random Forest``` ```max_depth```: peaks around depth 8 (about 0.63) and falls to about 0.55 at depth 20, while the training score reaches 1.0.
- ```Random Forest``` ```min_samples_leaf```: the best value is 5 (about 0.635). At 1 the forest overfits, and at 40 both lines are low, which is underfitting.
- ```Random Forest``` ```n_estimators```: the score stops improving after about 100 trees, so more trees just cost more time.

```max_depth``` and ```min_samples_leaf``` both limit how big the trees get, so they affect each other. That is why they were searched together instead of one at a time.

The learning curve goes from about 0.59 with 1,280 rows to about 0.63 with 5,120 rows and then flattens. The training score stays around 0.9, so the model still overfits a bit. More data like this probably would not help much.

### 5.5 Tuning

Each model was tuned with cross-validation on the training set only, scored by Macro-F1. ```Logistic Regression``` and ```Gradient Boosting``` used a grid search, which tries every combination. ```Random Forest``` used a random search of 20 combinations, because there are too many combinations to try them all. ```class_weight``` was one of the searched settings.

| Model | Best CV Macro-F1 | Notes |
|---|---|---|
| Dummy baseline | 0.244 | always predicts Low |
| Decision Tree (hand-picked) | 0.561 | max_depth = 5 |
| Logistic Regression (tuned) | 0.620 | C = 100, no class weights |
| Gradient Boosting (tuned) | 0.621 | grid search |
| Random Forest (tuned) | 0.634 | min_samples_leaf = 5, max_features = 0.5, max_depth = 14, balanced_subsample |

The top Random Forest settings are within 0.004 of each other, which is smaller than the spread between folds (about 0.02), so they are basically the same and picking the exact winner is partly luck. The settings that did well had medium-sized trees, a ```min_samples_leaf``` of at least 2, and balanced class weights. Tuning only improved the Random Forest a little (0.624 before and 0.634 after).

### 5.6 Is the best model really better?

The three tuned models were compared fold by fold, because they are all scored on the same 5 folds. ```Random Forest``` beats ```Logistic Regression``` by 0.014 Macro-F1 on average and wins 4 of the 5 folds. This means Random Forest is probably slightly better, but the evidence is not strong. ```Gradient Boosting``` is not better than Random Forest. I went with Random Forest because it has the highest score, but Logistic Regression is close and easier to explain, so if being easy to explain mattered more, it would be a fair choice too.

## 6. Results

### 6.1 Final test evaluation

The final ```Random Forest``` was evaluated on the test set once. It scores a Macro-F1 of 0.62, balanced accuracy of 0.615, accuracy of 0.72 and ROC-AUC of 0.852, compared to a baseline Macro-F1 of 0.244. The test Macro-F1 is a little below the cross-validated 0.634, which is expected because the cross-validated score was used to pick the model and the test set is small (only 85 High cases).

| Metric | Dummy baseline | Final Random Forest |
|---|---|---|
| Macro-F1 | 0.244 | 0.620 |
| Balanced accuracy | 0.333 | 0.615 |
| Accuracy | 0.579 | 0.720 |
| ROC-AUC | 0.5 | 0.852 |

Confusion matrix (rows are the true class, columns are the predicted class):

| True \ Predicted | Low | Medium | High |
|---|---|---|---|
| Low | 901 | 255 | 2 |
| Medium | 213 | 504 | 40 |
| High | 3 | 48 | 34 |

High stress is still the weakest class, with a precision of 0.45 and recall of 0.40. The model finds 34 of the 85 High cases, 48 are predicted as Medium, and only 3 are predicted as Low. Low and Medium are mixed up with each other (255 Low predicted as Medium and 213 Medium predicted as Low), which suggests those classes overlap in the features. Nearly all errors are between neighbouring levels, so the model does follow the order of stress levels, and it mostly struggles at the boundaries. A recall of 0.40 for High stress is too low for any real use.

### 6.2 What the model relies on

The final ```Random Forest``` relies most on ```Screen_Time_Hours``` (about 0.29), followed by ```Caffeine_Intake_mg``` (about 0.16) and ```Daily_Sleep_Hours``` (about 0.08). These importances show what the model used, not what causes stress, and they might reflect how this synthetic dataset was made rather than real physiology.

### 6.3 Summary of the model comparison

```Random Forest``` has the best Macro-F1 in cross-validation (0.634), but ```Logistic Regression``` has the higher accuracy and ROC-AUC, because it has no class weights and so does better on the common classes and worse on High stress. That is why Macro-F1 was used as the main metric.

### 6.4 Sensitivity check: Sleep Quality Score

As planned in the proposal, ```Sleep_Quality_Score``` was added as an input and the selected model was cross-validated again. The Macro-F1 barely changes (0.638 with it and 0.634 without), so it adds almost nothing here. It stays excluded, because it is recorded on the same day as stress and could cause leakage in a real use.

### 6.5 Fairness check by gender

The final model's test results were split by gender.

| Group | Macro-F1 | High recall | High cases |
|---|---|---|---|
| Female | 0.621 | 0.364 | 44 |
| Male | 0.617 | 0.439 | 41 |
| Other | not comparable | 0.00 | 0 (55 rows) |

```Female``` and ```Male``` have almost the same Macro-F1, but the High stress recall is different (0.36 compared to 0.44). That is based on only 44 and 41 High cases, so the gap could easily be down to chance. The ```Other``` group only has 55 rows in the test set and no High stress cases, so its High recall of 0.00 means nothing and its Macro-F1 cannot be compared with the others. Error rates for this group cannot be measured properly, which is the problem I mentioned in the proposal. This check only covers gender, not age or anything else.

## 7. Limitations and Responsible Use

1. **High stress is found only about 40% of the time.** This is the main limit of the model. A High stress prediction should never be treated as a diagnosis, and a missing one should not be read as "fine".
2. **The data looks synthetic.** The clipped minimum values and the evenly spread ```Age``` suggest the numbers were generated by rules. Patterns found in synthetic data may not exist in real people, so these results do not describe real people. The flat learning curve, and the fact that very different models all land close together, also fit with noise built into the data.
3. **The labels cannot be checked.** I do not know how the stress labels were produced, so the model may only be learning how the data was generated.
4. **The top models are very close.** Random Forest, Gradient Boosting and Logistic Regression are within about 0.014 of each other, so the choice of best model is not strongly supported.
5. **The fairness check is limited.** The ```Other``` group is too small and has no High cases, and the per-group High counts are low, so differences cannot be trusted.
6. **Misuse is a real risk.** Stress is sensitive health information. A model like this could be misused by employers, insurers or schools through wrong labels or surveillance. Predictions from it should only be used by a person for self-reflection, with their consent, and not for decisions about people.

## 8. Conclusion

The final model is the tuned ```Random Forest```. It scored a Macro-F1 of 0.634 in cross-validation and 0.62 on the test set, compared to 0.244 for the baseline. It is only slightly better than the tuned ```Logistic Regression``` (0.620 in cross-validation), and further tuning did not improve it by much, so the models seem close to the limit of what these features can show.

The main limit is High stress, which is only found about 40% of the time on the test set. The dataset also looks synthetic, so these results do not describe real people. The ```Sleep_Quality_Score``` check made almost no difference, and the gender check could not be done properly for the ```Other``` group because it had no High stress cases.

The part of the work I would stress most is the process. Preprocessing is inside the pipeline to avoid leakage, Macro-F1 is used because of the imbalance, settings are chosen from curves and searches, the test set was used once, and I checked my own assumptions. Some of those checks changed my mind. For example, I expected balanced class weights to be best for every model, but for Logistic Regression it was a trade-off.

With more time, the most useful next step would be real, non-synthetic data with a documented way of measuring stress, and enough people in the small groups to check fairness properly.

## 9. AI Assistance Declaration

Claude Code was used throughout this project for suggestions, planning and wording. Where its code suggestions were used, I rewrote and ran the code and edited the explanations. It was also used to help draft this report and to generate an illustrative video of the repository, which is labelled as AI-generated. No other external code or collaboration was used.
