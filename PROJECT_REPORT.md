# Hotel Reservation Cancellation Prediction Using KNN Classifier
**Project Report**

Submitted by: Nikita Pancheshwar (B.Tech AIML)
Organisation: Naviotech Solution Pvt. Limited, Indore
Date: November 2026

---
## Abstract
Hotel booking cancellations reduce revenue and make planning difficult. This project uses the Hotel Reservations dataset (about 36,000 bookings) to predict whether a booking will be cancelled, using the k-Nearest Neighbors algorithm. After cleaning, feature engineering, scaling and tuning k by 5-fold cross-validation, the final model reached an accuracy of 81.63% and an F1-score of 0.657 on unseen data (ROC-AUC 0.851), and was compared with Logistic Regression (80.79% accuracy).

## 1. Introduction
### 1.1 Background
Online booking platforms make it easy to cancel, so cancellations are common in the hospitality industry.
### 1.2 Problem statement
Given the details of a reservation, predict `booking_status` (Canceled / Not_Canceled).
### 1.3 Objectives
1. Explore the data and find the factors linked to cancellation.
2. Build and tune a KNN classifier.
3. Evaluate with accuracy, precision, recall, F1 and a confusion matrix.
4. Compare with Logistic Regression.
### 1.4 Scope
Binary classification on a single historical dataset.

## 2. Literature / Reference
Kaggle notebook "Hotel Reservations Dataset – Best Machine Learning" (anurag629) was used as a structural reference. KNN, train/test split, cross-validation, metrics, logistic regression and EDA follow the concepts taught in the Naviotech ML with Python course.

## 3. Dataset description
| Item | Value |
|---|---|
| Source | Kaggle – Hotel Reservations Classification Dataset |
| Rows x columns | 36,275 x 19 |
| Target | booking_status |
| Missing values | 0 |
| Duplicate rows | 10,275 (removed during cleaning, leaving 26,000 rows) |
| Class balance | Not_Canceled 67.24% / Canceled 32.76% |

Main columns: number of adults/children, weekend/week nights, meal plan, car parking, room type, lead time, arrival year/month/date, market segment, repeated guest, previous cancellations, previous bookings not cancelled, average room price, number of special requests.

## 4. Exploratory Data Analysis
### 4.1 Target distribution
![](figures/01_target_distribution.png)
Observation: about one in three bookings (32.76%) is cancelled, so the classes are moderately imbalanced.

### 4.2 Lead time
![](figures/02_lead_time.png)
Observation: cancelled bookings have a much longer median lead time (122 days) than kept bookings (39 days). Cancellation rate climbs from 8.9% for bookings made 0-7 days ahead to 73.9% for bookings made more than 180 days ahead.

### 4.3 Room price
![](figures/03_price.png)
Observation: the average room price is about 103 (median 108 for cancelled vs 95 for kept bookings). The distribution is right-skewed with a few very expensive outliers (maximum 540).

### 4.4 Market segment
![](figures/04_cancel_by_segment.png)
Observation: Online bookings have the highest cancellation rate (36.5%), followed by Offline (30.0%) and Aviation (29.6%). Corporate bookings cancel far less (10.9%) and Complementary bookings never cancelled in this data.

### 4.5 Special requests
![](figures/05_cancel_by_requests.png)
Observation: the more special requests a guest makes, the less likely the booking is cancelled: 43.2% with no requests, 23.8% with one, 14.6% with two, and 0% with three or more.

### 4.6 Repeated guests
![](figures/06_cancel_by_repeated_guest.png)
Observation: repeated guests almost never cancel (1.7%) compared with new guests (33.6%), although repeated guests are only about 2.6% of all bookings.

### 4.7 Monthly trend
![](figures/07_monthly_trend.png)
Observation: cancellations peak in July (45.0%) and June (40.3%) and are lowest in January (2.4%) and December (13.3%).

### 4.8 Correlation heatmap
![](figures/08_correlation_heatmap.png)
Observation: lead_time has the strongest positive correlation with cancellation (+0.44), followed by arrival_year (+0.18) and avg_price_per_room (+0.14). no_of_special_requests has the strongest negative correlation (-0.25).

## 5. Methodology
### 5.1 Data preprocessing
- Dropped `Booking_ID` (no predictive value) and duplicate rows.
- Encoded the target: Not_Canceled = 0, Canceled = 1.
- One-hot encoded `type_of_meal_plan`, `room_type_reserved`, `market_segment_type`.
- Created `total_nights` and `total_guests`.
### 5.2 Train/test split
80% training, 20% testing, stratified so both sets keep the same class ratio, `random_state = 42`.
### 5.3 Feature scaling
StandardScaler fitted on the training set only. KNN is based on Euclidean distance, so a large-range feature such as `lead_time` would otherwise dominate small-range features.
### 5.4 The KNN algorithm
A new booking is assigned the majority class among its k nearest training bookings, where distance is d(p,q) = sqrt(sum (p_i - q_i)^2).
### 5.5 Choosing k
Mean accuracy over 5-fold stratified cross-validation for odd k from 1 to 25.
![](figures/09_k_vs_accuracy.png)
Observation: the best mean CV accuracy among odd k from 1 to 25 was at k = 25, the edge of the range, so accuracy was still rising slowly. Very small k overfits (noisy); very large k underfits. The grid search below (scored on F1) selected k = 15.
### 5.6 Hyper-parameter tuning
GridSearchCV over `n_neighbors` {3,5,7,9,11,15}, `weights` {uniform, distance}, `metric` {euclidean, manhattan}, scored on F1. Best parameters: `n_neighbors = 15`, `weights = 'distance'`, `metric = 'manhattan'`.

## 6. Results
### 6.1 Metrics
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN (k=5) | 0.8094 | 0.6867 | 0.6142 | 0.6485 | 0.8358 |
| KNN (tuned) | 0.8163 | 0.7058 | 0.6142 | 0.6568 | 0.8512 |
| Logistic Regression | 0.8079 | 0.7009 | 0.5733 | 0.6307 | 0.8543 |

![](figures/12_model_comparison.png)

### 6.2 Confusion matrix
![](figures/10_confusion_matrix.png)
Of 5,200 test bookings (1,488 real cancellations), the tuned KNN caught 914 cancellations (true positives) and missed 574 (false negatives). It also raised 381 false alarms and correctly identified 3,331 kept bookings.
Precision = TP/(TP+FP): of bookings predicted as cancelled, how many truly were.
Recall = TP/(TP+FN): of real cancellations, how many were found.

### 6.3 ROC curve
![](figures/11_roc_curve.png)
AUC = 0.851.

## 7. Discussion
- Strongest drivers of cancellation: long lead time, Online booking channel, new (not repeated) guests, no special requests, higher room price and summer months.
- Business use: flag high-risk bookings for reminders or deposit policies.
- Cost of errors: a missed cancellation (false negative) means an empty room; a false alarm (false positive) means a needless reminder. Recall is therefore important.
- KNN vs Logistic Regression: tuned KNN is slightly better on accuracy (81.6% vs 80.8%), recall (0.614 vs 0.573) and F1 (0.657 vs 0.631), while Logistic Regression has a marginally higher ROC-AUC (0.854 vs 0.851). The differences are small. Note that the tuned KNN scores 99.6% on its own training data (because of distance weighting) versus 81.6% on test data, so it memorises the training set more than Logistic Regression, whose train and test accuracy are almost equal.

## 8. Limitations
1. KNN is slow at prediction time on large data.
2. Sensitive to scaling and irrelevant features.
3. Single dataset from one hotel group; may not generalise.
4. Class imbalance not specially treated, and 10,275 duplicate rows were removed, which reduces the amount of data (some of these may be genuinely separate but similar bookings).

## 9. Future work
Random Forest/XGBoost, SMOTE, feature selection, SHAP explanations, a Streamlit web app.

## 10. Conclusion
The KNN classifier, after scaling and tuning, predicts booking cancellation with 81.63% accuracy (precision 0.71, recall 0.61 on the Canceled class). The project demonstrates the full ML workflow from EDA to evaluation.

## 11. References
1. Kaggle: Hotel Reservations Classification Dataset.
2. Kaggle notebook by anurag629 (link in README).
3. scikit-learn documentation: https://scikit-learn.org
4. Naviotech Solution Pvt. Limited, Machine Learning using Python course material.

## Appendix: How to convert to Word/PDF
Open this file in VS Code (Markdown preview) or paste it into Word, then insert the figures from `reports/figures/`.
