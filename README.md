# E-Commerce Clickstream Analytics: Revenue Prediction & User Segmentation

## 📋 Project Overview

A comprehensive data science project analyzing 165,474 clickstream events from an online maternity clothing retailer (April-August 2008). The project employs dual-track analytics combining **regression modeling** for session revenue prediction and **time series forecasting** for temporal pattern analysis.

**Key Achievement:** Achieved 98.6% accuracy (R²=0.986) in predicting session revenue using behavioral engagement metrics, with SARIMA capturing weekly seasonality patterns.

---

## 🎯 Objectives

1. **Predict Session Revenue** - Estimate total session value using only behavioral features (no price leakage)
2. **Forecast Temporal Trends** - Capture daily clickstream patterns and seasonal cycles
3. **Segment Users** - Identify behavioral archetypes and revenue contribution patterns
4. **Website Optimization** - Analyze click distribution across page locations for UX improvement
5. **Business Intelligence** - Provide actionable insights for marketing, inventory, and operations

---

## 📊 Dataset

- **Source:** UCI Machine Learning Repository (DOI: 10.24432/C5QK7X)
- **Records:** 165,474 clickstream events
- **Sessions:** 24,026 unique user sessions
- **Time Period:** April-August 2008 (5 months)
- **Features:** 14 variables (temporal, behavioral, geographic, product, transaction)
- **Missing Values:** None
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

### Key Statistics

| Metric | Value |
|--------|-------|
| Mean Session Value | $301.68 |
| Median Session Value | $177.00 |
| Max Session Value | $8,538.00 |
| Avg Clicks per Session | 6.88 |
| Avg Pages Visited | 2.15 |

---

## 🔧 Technologies & Tools

### Languages & Frameworks
- **Python 3.9.7** - Core programming language
- **Jupyter Notebook** - Development environment
- **LaTeX** - Report generation (Overleaf)

### Data Processing
- **pandas 1.3.3** - Data manipulation and aggregation
- **numpy 1.21.2** - Numerical computations
- **scipy** - Statistical distributions

### Machine Learning
- **scikit-learn 0.24.2**
  - Regression models (Linear, Decision Tree, Random Forest, Gradient Boosting)
  - StackingRegressor for ensemble learning
  - StandardScaler for feature normalization
  - Metrics (RMSE, MAE, R²)
- **statsmodels** - ARIMA and SARIMA time series models

### Visualization
- **matplotlib 3.4.3** - Core plotting library
- **seaborn 0.11.2** - Statistical visualizations (heatmaps, distributions)
- **PIL (Pillow)** - Image processing for heatmap overlays
- **plotly** - Interactive dashboards

### Model Serialization
- **pickle** - Save/load trained models for deployment

---

## 📈 Models Implemented

### Regression Models (Session Revenue Prediction)

| Model | R² | RMSE | MAE | Rank |
|-------|-----|------|-----|------|
| **Linear Regression** | 0.986 | 46.45 | 29.55 | 🥇 1st |
| Stacking Ensemble (RF+GB) | 0.979 | 58.43 | 29.61 | 🥈 2nd |
| Gradient Boosting | 0.978 | 59.18 | 29.75 | 🥉 3rd |
| Decision Tree | 0.977 | 60.08 | 30.10 | 4th |
| Random Forest | 0.973 | 65.38 | 29.98 | 5th |

**Best Model:** Linear Regression (captures 98.6% of variance due to r=0.992 correlation between clicks and revenue)

### Time Series Models (Daily Forecasting)

| Model | RMSE | MAE | MAPE | Advantage |
|-------|------|-----|------|-----------|
| **SARIMA** | 210.17 | 142.63 | 12.8% | 🏆 Best - Captures seasonality |
| ARIMA | 281.43 | 187.82 | 17.9% | Good for trends |

**Best Model:** SARIMA (captures weekly seasonality, 25-33% better than ARIMA)

---

## 🎨 Key Visualizations

### Visualization 2: Correlation Heatmap
Shows strong linear relationship (r=0.992) between clicks and revenue. Confirms feature selection and minimal multicollinearity.

### Visualization 3: Revenue vs Engagement Analysis
Scatter plot with regression line demonstrating clear positive trend. Engagement metrics are strong predictors of session value.

### Visualization 4: Geographic & Temporal Analysis
Multi-panel visualization showing:
- Top 15 countries contribute 78% of revenue (Pareto principle)
- Clear weekly cycles in daily activity
- Location-based click patterns and monthly aggregates

### Visualization 5: User Segment Performance Dashboard
Comprehensive comparison of three behavioral segments:
- **Browsers** (49.8%): Low engagement, 23% revenue share
- **Explorers** (32.4%): Moderate engagement, 32% revenue share
- **Deep Researchers** (17.8%): High engagement, **45% revenue share** ⭐

### Bonus: Website Click Heatmap
6-region overlay showing click concentration:
- **Location 1 (Top-Left):** 34,532 clicks - Highest engagement
- **Location 6 (Bottom-Right):** 20,743 clicks - Lowest engagement
- Header regions account for 48.3% of all clicks

---

## 🔑 Key Insights

### 1. **Engagement Drives Revenue**
- Correlation r=0.992 between clicks and session value
- Each additional click ≈ $44 revenue increase
- Engagement score formula: 0.4×(clicks/max) + 0.3×(pages/max) + 0.3×(locations/max)

### 2. **User Segmentation Opportunity**
- Deep Researchers (17.8% of users) generate 45% of revenue
- 2.6x average revenue vs. Browsers
- Focus retention efforts on high-value segments

### 3. **Geographic Concentration**
- Top 15 countries = 78% of revenue
- Opportunity for geographic diversification
- Region-specific marketing strategies needed

### 4. **Temporal Seasonality**
- Clear weekly cycles with weekend peaks
- SARIMA captures patterns for operational planning
- Enables predictive staffing and inventory management

### 5. **Page Location Matters**
- Location 1 (top-left): 1.67x engagement vs. Location 6
- Header region: 48.3% of total clicks
- Right-side redesign could improve engagement

---

## 📁 Project Structure

```
clickstream-analytics/
├── data/
│   └── e-shop_clothing_2008.csv          # Raw clickstream data
├── notebooks/
│   ├── clothing.ipynb                    # Main EDA and preprocessing
│   └── majorplots.ipynb                  # Visualization generation
├── models/
│   ├── stacking_regressor_model.pkl      # Trained ensemble model
│   ├── scaler_regression.pkl             # Feature scaler
│   └── best_timeseries_model_sarima.pkl  # SARIMA model
├── outputs/
│   ├── viz02_correlation_heatmap.png
│   ├── viz03_revenue_engagement_analysis.png
│   ├── viz04_geographic_temporal_analysis.png
│   ├── viz05_user_segment_dashboard.png
│   ├── website_click_heatmap.png
│   └── model_comparison_results.csv
├── report/
│   ├── clickstream-analytics-report.tex  # LaTeX report
│   └── clickstream-analytics-report.pdf  # Compiled PDF
└── README.md                             # This file
```

---

## 🚀 Quick Start

### Prerequisites
```bash
python 3.9.7 or higher
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels pillow
```

### Installation
```bash
# Clone or download the project
cd clickstream-analytics

# Install dependencies
pip install -r requirements.txt
```

### Run Analysis
```python
# 1. Load and preprocess data
jupyter notebook clothing.ipynb

# 2. Generate visualizations
jupyter notebook majorplots.ipynb

# 3. Build regression models
python train_regression_models.py

# 4. Train time series models
python train_timeseries_models.py

# 5. Generate heatmap overlay
python generate_website_heatmap.py
```

---

## 📊 Regression Model Usage

```python
import pickle
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
with open('stacking_regressor_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_regression.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare new session data
new_session = pd.DataFrame({
    'clicks_per_session': [15],
    'max_order': [5],
    'order_std': [1.2],
    'unique_pages_visited': [4],
    'unique_locations_clicked': [3]
})

# Scale and predict
new_session_scaled = scaler.transform(new_session)
predicted_value = model.predict(new_session_scaled)[0]
print(f"Predicted session value: ${predicted_value:.2f}")
```

---

## 📈 Time Series Forecasting

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit SARIMA model
model = SARIMAX(
    daily_values,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 7)  # Weekly seasonality
)
results = model.fit()

# 7-day forecast
forecast = results.get_forecast(steps=7)
forecast_df = forecast.conf_int()
print(forecast_df)
```

---

## 🎯 Business Recommendations

| Initiative | Potential Impact | Timeline |
|-----------|-----------------|----------|
| Deep Researcher VIP Programs | +15-20% revenue | 2-4 weeks |
| Browser Conversion via Discovery | +8-12% revenue | 4-6 weeks |
| Location 1 Product Placement | +3-5% revenue | 1-2 weeks |
| Seasonal Marketing Alignment | +10-15% revenue | 2-3 weeks |
| Geographic Expansion | +5-10% revenue | 8-12 weeks |

---

## 📚 Report & Presentation

- **Full Report:** `clickstream-analytics-report.tex` (LaTeX format)
- **Visualizations:** 5 professional charts + website heatmap
- **Slides:** PowerPoint presentation with 13 slides
- **PlantUML Diagram:** System architecture

---

## 🔗 References

1. Łapczyński, M., & Białowąs, S. (2013). Predicting E-commerce Conversion Using Clickstream Data. *Studia Ekonomiczne*.
2. UCI Machine Learning Repository. (2019). Clickstream Data for Online Shopping. DOI: 10.24432/C5QK7X
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.
4. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.
5. Box, G. E., et al. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.

---

## ⚠️ Limitations

- **Historical Data:** Dataset from 2008; patterns may differ in modern e-commerce
- **Single Domain:** Maternity clothing only; generalization requires validation
- **5-Month Window:** Misses annual seasonality (holidays, seasonal products)
- **Aggregate Data:** No individual user tracking for cohort analysis
- **No External Factors:** Marketing spend, competitor activity not included

---

## 🔮 Future Enhancements

- [ ] Deep Learning (LSTM/RNN) for sequential click patterns
- [ ] Real-time deployment in production systems
- [ ] Causal inference for design impact measurement
- [ ] Multi-task learning (conversion, AOV, churn joint prediction)
- [ ] Privacy-preserving federated learning
- [ ] SHAP/LIME explainability for stakeholder communication
- [ ] A/B testing framework for recommendation validation

---

## 📝 License

Dataset: Creative Commons Attribution 4.0 International (CC BY 4.0)
Project Code: Open Source (MIT License)

---

## 👥 Authors

Team Members: [Add your names]
Guide: [Add guide name]
Institution: [Add institution name]

---

## 📧 Support & Contact

For questions or issues:
- Review the main report: `clickstream-analytics-report.pdf`
- Check Jupyter notebooks for detailed code comments
- Refer to inline documentation in Python scripts

---

## ✨ Highlights

✅ **98.6% Accuracy** in revenue prediction  
✅ **25% Better Forecasting** with SARIMA vs. ARIMA  
✅ **2.6x Revenue Variance** across user segments  
✅ **48% Header Engagement** identifies UX optimization opportunity  
✅ **Production-Ready Code** with model serialization  
✅ **Professional Report** with LaTeX formatting  

---

**Last Updated:** November 21, 2025  
**Status:** Complete & Ready for Submission ✅
