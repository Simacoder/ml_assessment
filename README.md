# 🏠 House Prices - Exploratory Data Analysis & Predictive Modeling

A comprehensive end-to-end machine learning project demonstrating exploratory data analysis (EDA), data cleaning, feature engineering, and predictive modeling on the Ames Housing dataset.

**Model Performance:** 99.38% R² | **Average Error:** ±$6,905 | **Dataset:** 1,460 Training + 1,459 Test Records

---

## 📋 Overview

This project performs a complete data science workflow:

✅ **Exploratory Data Analysis** - Understand data structure, distributions, and relationships  
✅ **Data Cleaning** - Handle missing values, outliers, and data quality issues  
✅ **Feature Engineering** - Create new features to improve model performance  
✅ **Categorical Encoding** - Convert categorical variables for modeling  
✅ **Model Training** - Train and compare 3 regression algorithms  
✅ **Test Predictions** - Generate predictions for Kaggle submission  

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Step 1: Install Dependencies
```bash
pip install pandas matplotlib seaborn scikit-learn numpy scipy
```

### Step 2: Download Dataset
Download the dataset from [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data):
- `train.csv` 
- `test.csv`

Create a `data/` folder and place both files inside:
```
project/
├── data/
│   ├── train.csv
│   └── test.csv
└── Housing_Prices_Analysis.ipynb
```

### Step 3: Run the Notebook
```bash
jupyter notebook  house_price_data_analysis.ipynb
```

Click **Cell → Run All** (or press `Ctrl+Shift+Enter`)

---

## 📁 Project Structure

```bash ml_assessment/
├──  house_price_data_analysis.ipynb       # Main notebook (run this!)
├── README.md                            # This file
├── data/
│   ├── train.csv                        # Training data (1,460 records)
│   └── test.csv                         # Test data (1,459 records)
└── predictions.csv                      # Generated predictions (after running)
```

---

## 📊 Notebook Contents

### 1. **Setup & Data Ingestion**
- Import required libraries (pandas, numpy, matplotlib, seaborn, scikit-learn)
- Load training and test datasets
- Display dataset shape and basic information

### 2. **Data Exploration**
- Dataset overview and structure
- Data types identification
- Feature classification (numerical vs categorical)
- Sample records preview

### 3. **Missing Values Analysis**
- Identify missing values in training and test data
- Calculate missing percentages
- Flag problematic columns
- Duplicate record detection

### 4. **Data Cleaning & Preparation**
- Strategic missing value imputation:
  - Categorical features → Mode (most common value)
  - Numerical features → Median (robust to outliers)
  - Absence indicators → 'NA' (for features like PoolQC, GarageType)
- Applied consistently to both training and test data

### 5. **Exploratory Data Analysis (EDA)**
- **Target Variable Analysis (SalePrice)**
  - Descriptive statistics (mean, median, std dev, min, max)
  - Histogram showing right-skewed distribution
  - Q-Q plot for normality assessment
  - Log transformation visualization

- **Feature Distributions**
  - 8 key numerical features analyzed
  - Distribution shapes and patterns identified

- **Correlation Analysis**
  - Correlation matrix for all numerical features
  - Top 15 features ranked by correlation with SalePrice
  - Heatmap visualization showing relationships

- **Feature-Target Relationships**
  - Scatter plots for top 5 features vs SalePrice
  - Trend lines added for visual interpretation
  - Linear relationship patterns identified

- **Categorical Features Analysis**
  - Average prices by neighborhood
  - Building type impact on price
  - House style preferences
  - Quality ratings effect

### 6. **Feature Engineering**
Creates new features to improve model performance:
- **TotalSF** = TotalBsmtSF + 1stFlrSF + 2ndFlrSF (total square footage)
- **TotalPorchSF** = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch
- **Age** = 2024 - YearBuilt (property age in years)
- **YearsRenovated** = YearRemodAdd - YearBuilt (renovation status)

Applied consistently to both training and test datasets.

### 7. **Categorical Encoding**
- One-hot encoding of categorical variables
- Feature alignment between training and test data
- Handling missing categories in test set
- Drop first category to avoid multicollinearity

### 8. **Model Training & Evaluation**
Trains 3 regression models:

#### Linear Regression
- Train R²: 0.9753 | Val R²: -0.0660 | RMSE: $90,423
- ❌ Poor generalization, severe overfitting

#### Random Forest
- Train R²: 0.9994 | Val R²: 0.9907 | RMSE: $8,426
- ✅ Excellent performance, good generalization

#### Gradient Boosting (Selected)
- Train R²: 0.9999 | Val R²: 0.9938 | RMSE: $6,905
- 🏆 Best performance, minimal overfitting, most accurate

### 9. **Test Predictions**
- Generates predictions for 1,459 test properties
- Creates `predictions.csv` for Kaggle submission
- Shows prediction statistics (mean, min, max)

### 10. **Key Insights & Conclusions**
- Top factors driving house prices
- Model performance summary
- Data quality assessment
- Recommendations for improvement

---

## 🎯 Key Findings

### Top Price Drivers (by Correlation)

| Feature | Correlation | Impact |
|---------|-------------|--------|
| Overall Quality | 0.79 | Most important - quality ratings strongly influence price |
| Ground Living Area | 0.71 | Size matters - larger living spaces command higher prices |
| Total Basement SF | 0.61 | Additional space adds value |
| Garage Area | 0.64 | Storage space is valued |
| Year Built | 0.55 | Newer properties preferred |

### Model Performance Comparison

| Model | Train R² | Val R² | RMSE |
|-------|----------|--------|------|
| Linear Regression | 0.9753 | -0.0660 | $90,423 |
| Random Forest | 0.9994 | 0.9907 | $8,426 |
| **Gradient Boosting** | **0.9999** | **0.9938** | **$6,905** |

**Selected Model:** Gradient Boosting
- **Accuracy:** 99.38% of price variance explained
- **Average Error:** ±$6,905 per prediction
- **Overfitting:** Minimal (0.61% train-val gap)

---

## 📈 Visualizations Included

✅ **SalePrice Distribution** - Histogram and Q-Q plot showing right-skewed distribution  
✅ **Log-Transformed SalePrice** - Normalized distribution after transformation  
✅ **Feature Distributions** - 8 key numerical features analyzed  
✅ **Correlation Heatmap** - All feature correlations displayed  
✅ **Scatter Plots** - Top 5 features vs SalePrice with trend lines  
✅ **Categorical Analysis** - Average prices by category (neighborhood, type, style, quality)  

---

## 💡 Project Highlights

### Data Processing
- ✅ Handles 19 missing value patterns
- ✅ Distinguishes between "absence" and "missing" data
- ✅ Robust median imputation for numerical features
- ✅ Mode imputation for categorical features
- ✅ Applied consistently to train and test sets

### Feature Engineering
- ✅ Creates 4 domain-informed features
- ✅ Captures spatial relationships
- ✅ Incorporates time-based information
- ✅ Improves model predictive power

### Model Development
- ✅ Trains multiple algorithms
- ✅ Proper train-validation split (80-20)
- ✅ RobustScaler for feature normalization
- ✅ Comprehensive performance metrics
- ✅ Model comparison and selection

### Output
- ✅ `predictions.csv` - Ready for Kaggle submission
- ✅ 1,459 house price predictions
- ✅ Full reproducibility with documented steps

---

## 🐛 Troubleshooting

### Issue: "FileNotFoundError: data/train.csv not found"
**Solution:** 
- Download datasets from [Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
- Create a `data/` folder
- Place `train.csv` and `test.csv` inside

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Solution:** Install missing packages
```bash
pip install pandas matplotlib seaborn scikit-learn numpy scipy
```

### Issue: Jupyter kernel not found
**Solution:** Install and start Jupyter
```bash
pip install jupyter
jupyter notebook
```

### Issue: Predictions not generated
**Solution:** Ensure both `train.csv` and `test.csv` are in the `data/` folder and notebook runs without errors

---

## 📊 Data Summary

### Training Data (1,460 records)
- **Target Variable:** SalePrice
- **Numerical Features:** 37
- **Categorical Features:** 43
- **Total Features:** 81
- **Missing Values:** 19 patterns handled

### Test Data (1,459 records)
- **No Target Variable:** SalePrice values to predict
- **Same Features:** Aligned with training data
- **Missing Values:** Same patterns as training

### Price Statistics
- **Mean Price:** $180,921
- **Median Price:** $163,000
- **Min Price:** $34,900
- **Max Price:** $755,000
- **Std Deviation:** $79,443

---

## 🎓 Learning Outcomes

After completing this project, you will understand:

✅ **Data Exploration** - How to analyze and understand datasets  
✅ **Data Cleaning** - Handling missing values and inconsistencies  
✅ **EDA Techniques** - Visualization and statistical analysis  
✅ **Feature Engineering** - Creating meaningful features from raw data  
✅ **Model Selection** - Choosing appropriate algorithms  
✅ **Model Evaluation** - Assessing performance with proper metrics  
✅ **Kaggle Workflows** - End-to-end pipeline for competitions  

---

## 📚 Resources

- [Kaggle House Prices Competition](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Matplotlib & Seaborn](https://matplotlib.org/)

---

## 📝 Expected Output

After running the notebook successfully, you should see:

```
✓ Dataset loaded: 1460 training, 1459 test
✓ Missing values analyzed and handled
✓ 10+ visualizations displayed
✓ Features engineered successfully
✓ 3 models trained and compared
✓ Best model: Gradient Boosting
✓ Test predictions generated: 1459 records
✓ predictions.csv created and ready for submission
```

---

## ✨ Key Metrics Summary

| Metric | Value |
|--------|-------|
| **Best Model** | Gradient Boosting |
| **Validation R²** | 0.9938 (99.38%) |
| **Validation RMSE** | $6,905 |
| **Overfitting Gap** | 0.61% |
| **Features Used** | 280+ (after encoding) |
| **Training Records** | 1,160 (80%) |
| **Validation Records** | 290 (20%) |
| **Test Records** | 1,459 |


---

##  Support

**Questions or Issues?**
- Check the troubleshooting section above
- Review notebook comments and markdown explanations
- Verify all required libraries are installed
- Ensure data files are in the correct location

---

##  Assessment Rubric Coverage

This project demonstrates:

✅ **Exploratory Data Analysis** 
- Dataset understanding and exploration
- Data types and structure analysis
- Missing values identification
- Statistical summaries and distributions

✅ **Data Cleaning** 
- Missing value imputation
- Data quality checks
- Feature preparation
- Consistent train/test handling

✅ **Visualization & Reporting** 
- 10+ professional visualizations
- Clear insights and interpretations
- Correlation and relationship analysis
- Distribution and pattern identification

✅ **Modeling & Evaluation** (20%)
- Multiple algorithms trained
- Model comparison and selection
- Performance metrics
- Predictions generated

---

## 📄 License

This project is based on the Kaggle House Prices dataset.  
Original Data Source: [Dean De Cock, Truman State University](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/)

---

**Ready to predict house prices? Let's go! 🎉**

```bash
jupyter notebook  house_price_data_analysis.ipynb
```

Run all cells and generate your predictions!

---

# AUTHOR
- Simanga Mchunu