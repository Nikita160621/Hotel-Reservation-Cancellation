# 🏨 HotelGuard AI – Hotel Reservation Cancellation Predictor

An interactive, browser-based machine learning dashboard that predicts the likelihood of a hotel reservation being cancelled, estimates the financial loss, and groups guests into behavioural personas.

**🔗 Live Demo:** [https://nikita160621.github.io/Hotel-Reservation-Cancellation/](https://nikita160621.github.io/Hotel-Reservation-Cancellation/)

---

## 📌 Overview

Last-minute cancellations and no-shows cost hotels significant revenue. HotelGuard AI helps revenue and front-desk teams identify high-risk bookings early so they can take action, such as requiring deposits, overbooking strategically, or sending reminders.

The project combines **classification**, **regression**, **clustering** and **NLP** on a dataset of **4,500 historical reservations** (overall cancellation rate: **23.02%**).

---

## ✨ Features

- **Live Cancellation Prediction** – risk probability updates as you change any booking detail.
- **Expected Financial Loss** – estimates the revenue at risk for each booking.
- **Notice Timing Estimate** – predicts how many days before check-in a cancellation is likely.
- **NLP Guest Note Analysis** – scores special requests and notes (e.g. *"flight confirmed"* vs. *"may have to cancel if visa is delayed"*) for commitment level.
- **KNN Nearest Bookings** – shows the most similar historical reservations and their outcomes.
- **Guest Persona Clustering** – assigns each booking to one of 4 K-Means clusters.
- **Model Benchmark Dashboard** – compare models via metrics, ROC curves and confusion matrices.
- **Hotel Insights** – cancellation rates by room type and deposit policy.
- **Quick Demo Presets** – one-click scenarios: *Safe Solo Traveler*, *High-Risk Group Booking*, *Reliable Digital Nomad*, *Visa Delayed International*.
- **Multi-Device Access** – works on desktop and mobile, with a QR code for quick phone access.

---

## 🧠 Machine Learning Approach

### Models Evaluated

| Model | 5-Fold CV Acc | Test Acc | Precision | Recall | F1-Score | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| K-Nearest Neighbours (k=11) | 82.86% (±1.1%) | 82.89% | 0.648 | 0.560 | 0.601 | 0.841 |
| Logistic Regression | 85.39% (±1.3%) | 83.78% | 0.678 | 0.560 | 0.614 | 0.891 |
| Gradient Boosting | 84.81% (±1.12%) | 84.67% | 0.695 | 0.594 | 0.641 | 0.886 |

### Additional Models

| Task | Model | Result |
| --- | --- | --- |
| Financial loss prediction | Ridge Regression | RMSE ≈ $338.56 |
| Cancellation notice timing | Regression | MAE ≈ 5.44 days |
| Guest segmentation | K-Means (k=4) + PCA | Silhouette ≈ 0.186 |
| Text analysis | TF-IDF NLP features | Included in 94-dimensional feature space |

### Validation Strategy

- 80/20 train–test split
- 5-fold stratified cross-validation
- No data leakage
- Generalization gap reported for each model

---

## 🔍 Key Insights

**Factors that increase cancellation risk**

| Factor | Effect on Odds |
| --- | --- |
| No deposit policy | +4.21x |
| NLP uncertainty score | +3.11x |
| Booking.com (OTA) channel | +2.28x |
| Long lead time (>90 days) | +1.91x |

**Protective factors**

| Factor | Effect on Odds |
| --- | --- |
| Walk-in / same-day arrival | 0.12x |
| Non-refundable 100% deposit | 0.13x |
| Note keyword: "flight confirmed" | 0.23x |
| Repeat guest / previous stays | 0.47x |

---

## 👥 Guest Personas (K-Means Clusters)

| Cluster | Persona | Risk | Cancel Rate | Avg Lead Time | Avg Spend |
| --- | --- | --- | --- | --- | --- |
| 0 | Spontaneous Solo Travelers | Low | 14.2% | 11.5d | $210.00 |
| 1 | High-Risk Speculative Groups | High | 42.8% | 98.4d | $985.55 |
| 2 | Reliable Digital Nomads & Long Stayers | Very Low | 9.8% | 22.7d | $769.73 |
| 3 | Weekend Tourists & City Explorers | Moderate | 24.5% | 31.4d | $354.60 |

---

## 🧾 Input Fields

| Field | Description |
| --- | --- |
| Booking Lead Time | Days between booking and arrival |
| Room Type | Standard Single, Deluxe Double, Executive Suite, Family Suite, 4-Bed Shared Dorm |
| Booking Channel | Direct Website, Booking.com, Hotelworld, Airbnb, Walk-in, Expedia |
| Customer Persona | Solo Traveler, Digital Nomad, Group Booking, Weekend Tourist, Business Traveler |
| Deposit Policy | No Deposit, Partial Deposit (10%), Non-Refundable 100% |
| Price / Nights / Guests | Booking value and size |
| Past Cancels / Past Stays / Changes | Guest history and booking modifications |
| Repeat Guest | Yes / No |
| Guest Note | Free text analysed with NLP |

---

## 🚀 Getting Started

### Use Online

Open the live demo in any modern browser (Chrome, Safari, Edge, Firefox) on desktop, iOS or Android:

👉 **https://nikita160621.github.io/Hotel-Reservation-Cancellation/**

### Run Locally

```bash
# Clone the repository
git clone https://github.com/nikita160621/Hotel-Reservation-Cancellation.git

# Move into the project folder
cd Hotel-Reservation-Cancellation

# Open index.html in your browser, or serve it locally:
python -m http.server 8000
# then visit http://localhost:8000
```

---

## 🛠️ Tech Stack

- **Machine Learning:** Scikit-Learn (KNN, Logistic Regression, Gradient Boosting, Ridge Regression, K-Means, PCA)
- **NLP:** TF-IDF text features
- **Frontend:** HTML, CSS, JavaScript
- **Deployment:** GitHub Pages (HTTPS)

---

## 📁 Project Structure

```
Hotel-Reservation-Cancellation/
├── index.html      # Main dashboard / web app
├── README.md       # Project documentation
└── ...             # Additional assets, notebooks or data files
```

> Update this section to match your actual repository layout.

---

## 🔮 Future Improvements

- Connect to a live reservation system or API
- Add model explainability (e.g. SHAP values)
- Improve recall to reduce missed cancellations (current miss rate ≈ 44%)
- Support multilingual guest notes
- Add automated retraining on new booking data

---

## 👩‍💻 Author

**Nikita Pancheshwar**

Built with Scikit-Learn, KNN, Regression, K-Means Clustering & NLP.

---

## 📄 License

This project is open source. Add your preferred license (e.g. MIT) here.
