# Hotel Reservation Cancellation Prediction using KNN

Predict whether a hotel booking will be **cancelled** or **kept** using a k-Nearest Neighbors (KNN) classifier.

> Major project – Machine Learning using Python, Naviotech Solution Pvt. Limited, Indore
> Author: [Nikita Pancheshwar](https://github.com/Nikita160621) (B.Tech, Artificial Intelligence & Machine Learning)

![Confusion matrix](reports/figures/10_confusion_matrix.png)

## 1. Problem statement
Cancellations cost hotels revenue because rooms stay empty. If a hotel can predict which bookings are likely to be cancelled, it can plan overbooking, send targeted reminders and apply flexible pricing. This project builds a binary classifier for `booking_status` = **Canceled** / **Not_Canceled**.

## 2. Dataset
- **Hotel Reservations Dataset** (Kaggle), file `Hotel Reservations.csv`: 36,275 bookings, 19 columns, no missing values.
- 10,275 rows are duplicates (ignoring `Booking_ID`) and are removed, leaving 26,000 rows.
- Class balance: 67.24% Not_Canceled, 32.76% Canceled.
- Key columns: `lead_time`, `avg_price_per_room`, `market_segment_type`, `no_of_special_requests`, `repeated_guest`, `arrival_month`, `no_of_week_nights`.

## 3. Approach
1. Data quality checks (nulls, duplicates, class balance)
2. Exploratory data analysis (8 figures)
3. Preprocessing: drop ID and duplicates, one-hot encoding, feature engineering (`total_nights`, `total_guests`), stratified 80/20 split (20,800 train / 5,200 test, 29 features), **StandardScaler** fitted on train only (KNN is distance based)
4. Baseline KNN (k = 5)
5. Choosing k with 5-fold stratified cross-validation
6. Hyper-parameter tuning with GridSearchCV (k, weights, distance metric), scored on F1
7. Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
8. Comparison with Logistic Regression

## 4. Results (test set, 5,200 bookings)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN (k=5) | 0.8094 | 0.6867 | 0.6142 | 0.6485 | 0.8358 |
| **KNN (tuned)** | **0.8163** | 0.7058 | **0.6142** | **0.6568** | 0.8512 |
| Logistic Regression | 0.8079 | 0.7009 | 0.5733 | 0.6307 | 0.8543 |

Best parameters: `n_neighbors=15`, `weights='distance'`, `metric='manhattan'`.

Tuned KNN, per class:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Not_Canceled | 0.85 | 0.90 | 0.87 | 3712 |
| Canceled | 0.71 | 0.61 | 0.66 | 1488 |

**Key findings**
- Long lead time is the strongest signal: 73.9% of bookings made more than 180 days ahead are cancelled vs 8.9% of those made within a week.
- Online bookings cancel most (36.5%); repeated guests almost never cancel (1.7%); more special requests means fewer cancellations.
- The tuned KNN beats Logistic Regression slightly on accuracy, recall and F1, but the gap is small. It also fits its training data almost perfectly (distance weighting), so some overfitting remains.

| | |
|---|---|
| ![ROC](reports/figures/11_roc_curve.png) | ![Comparison](reports/figures/12_model_comparison.png) |

Full numbers are in `reports/results.txt`; the written analysis is in `reports/PROJECT_REPORT.md`.

## 5. Repository structure
```
Hotel-Reservation-Cancellation/
├── data/                  # "Hotel Reservations.csv"
├── notebooks/
│   └── hotel_cancellation_knn.ipynb   # step-by-step notebook
├── src/                   # config, data_loader, eda, preprocess, train, evaluate
├── reports/
│   ├── figures/           # generated graphs
│   ├── results.txt        # generated metrics
│ 
├── Project PPT.pptx
├── models/                # saved model (generated, git-ignored)
├── main.py                # runs the full ML pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

## 6. How to run
```bash
git clone https://github.com/Nikita160621/Hotel-Reservation-Cancellation.git
cd Hotel-Reservation-Cancellation
python -m venv venv
venv\Scripts\activate        # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python main.py               # figures -> reports/figures, metrics -> reports/results.txt
```
Or open `notebooks/hotel_cancellation_knn.ipynb` in Jupyter / Google Colab.

## 7. Tech stack
Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, Jupyter.

## 8. Limitations and future work
- KNN is slow at prediction time on large data and sensitive to scaling and irrelevant features.
- Duplicate removal drops 28% of rows; some may be genuinely separate bookings.
- Class imbalance is not specially treated; recall on the Canceled class is 0.61.
- Next steps: Random Forest / Gradient Boosting, SMOTE, feature selection, a Streamlit app.

## 9. Acknowledgements
Naviotech Solution Pvt. Limited, Indore (training and internship); Kaggle Hotel Reservations dataset; reference notebook by anurag629.

## 10. License
MIT
