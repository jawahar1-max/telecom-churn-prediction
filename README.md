# 📶 Telecom Customer Churn Prediction App

A Streamlit-based machine learning application that predicts customer churn for telecom companies using classification models.

## Project Overview

This application demonstrates:
- **Data Generation**: Synthetic telecom customer data creation
- **Model Training**: Multiple classification algorithms (Random Forest, Logistic Regression, Decision Tree, SVM)
- **Visualization**: Churn analysis by age groups
- **Model Evaluation**: Accuracy, ROC AUC, classification reports, and confusion matrices
- **Download Features**: Export dataset and trained models

## Project Structure

```
Churn prediction/
├── app.py                           # Main Streamlit application
├── make_data.py                     # Data generation script
├── requirement.txt                  # Python dependencies
├── README.md                        # This file
└── data/
    └── telecom_churn_sample.csv    # Generated dataset (2000 records)
```

## Installation & Setup

### Step 1: Install Python Dependencies

```bash
pip install -r requirement.txt
```

**Dependencies:**
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning models
- `joblib` - Model serialization
- `streamlit` - Web app framework
- `matplotlib` - Data visualization

### Step 2: Generate Sample Data

```bash
python make_data.py
```

This creates `data/telecom_churn_sample.csv` with 2000 synthetic customer records.

### Step 3: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Features

### Dataset Features
- **customer_id**: Unique customer identifier
- **age**: Customer age (18-80)
- **call_duration**: Monthly call duration (minutes)
- **internet_usage**: Monthly internet usage (GB)
- **complaints**: Number of customer complaints
- **monthly_charges**: Monthly service charges ($)
- **churn**: Target variable (0 = retained, 1 = churned)

### Application Features

#### 1. **Dataset Preview**
   - View first 5 rows of the dataset
   - Display total number of records (2000)

#### 2. **Churn Analysis by Age Groups**
   - Visualize churn rate by age segments (18-29, 30-39, 40-49, 50-59, 60+)
   - Show count of churned customers per age group

#### 3. **Model Training**
   - Choose between 4 classification algorithms:
     - Random Forest (200 estimators)
     - Logistic Regression
     - Decision Tree
     - Support Vector Machine (SVM)
   - Adjustable test set size (10-40%)
   - Random seed control for reproducibility

#### 4. **Model Performance Metrics**
   - **Accuracy**: Overall prediction correctness
   - **ROC AUC**: Area under the ROC curve
   - **Classification Report**: Precision, recall, F1-score per class
   - **Confusion Matrix**: True/False positives and negatives

#### 5. **Model Management**
   - Auto-save trained models to `model/` directory
   - Load pretrained models
   - Download trained models as pickle files

#### 6. **Data Export**
   - Download the full dataset as CSV
   - Download trained models as pickle files

## How to Use the Application

1. **Adjust Settings** (Left Sidebar)
   - Select your preferred machine learning model
   - Set the test set percentage (default: 25%)
   - Set random seed for reproducibility (default: 42)
   - Click "Train model" to retrain with new settings

2. **View Results**
   - Analyze churn distribution by age group
   - Review model performance metrics
   - Check confusion matrix and classification report

3. **Download**
   - Export the dataset for external analysis
   - Download trained model for production use

## Data Generation Details (`make_data.py`)

The synthetic data is generated with realistic patterns:

```python
# Churn probability is influenced by:
# - Age: Slightly increases churn for younger/older customers
# - Call Duration: Positive engagement reduces churn
# - Internet Usage: More usage reduces churn
# - Complaints: Each complaint increases churn probability
# - Monthly Charges: Higher charges slightly increase churn
```

**Generation Process:**
1. Generate 2000 customer records
2. Create features with realistic distributions
3. Calculate churn probability using logistic model
4. Save to CSV format

## Model Details

### Features Used
- age
- call_duration
- internet_usage
- complaints
- monthly_charges

### Train/Test Split
- Default: 75% training, 25% testing
- Stratified sampling to maintain churn ratio

### Model Selection

| Model | Pros | Cons |
|-------|------|------|
| Random Forest | High accuracy, handles non-linearity | Less interpretable |
| Logistic Regression | Fast, interpretable, good baseline | Assumes linearity |
| Decision Tree | Interpretable, fast | Prone to overfitting |
| SVM | Good for binary classification | Slower, harder to tune |

## File Outputs

### Generated Files
- `data/telecom_churn_sample.csv` - Dataset with 2000 records
- `model/model_trained.pkl` - Trained model (after running app)
- `model/model_rf.pkl` - Pretrained Random Forest model (if available)

## Customization

### Add More Features
Edit `make_data.py` to add new customer attributes:
```python
df = pd.DataFrame({
    "customer_id": [...],
    "age": age,
    "new_feature": new_feature,  # Add here
    ...
})
```

### Change Model Parameters
Edit `app.py` in the `train_model()` function:
```python
if model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=300, max_depth=15, random_state=rs)
```

### Modify Age Groups
Edit the bins and labels in `app.py`:
```python
bins = [18, 30, 40, 50, 60, 80]
labels = ["18-29","30-39","40-49","50-59","60+"]
```

## Troubleshooting

### Streamlit Not Found
```bash
pip install streamlit
```

### Import Errors
```bash
pip install -r requirement.txt
```

### Dataset Not Found
```bash
python make_data.py
```

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

## Performance Tips

1. **First Run**: App trains model on startup; subsequent runs use cached data
2. **Large Datasets**: Consider preprocessing data before loading
3. **Model Selection**: Random Forest is most accurate but slowest
4. **Caching**: Streamlit caches data loading with `@st.cache_data`

## Future Enhancements

- Add feature importance analysis
- Implement hyperparameter tuning
- Add cross-validation results
- Create prediction API endpoint
- Add real dataset import functionality
- Implement A/B testing framework

## Author Notes

This is a demonstration app for learning machine learning workflows with Streamlit. The data is synthetic and designed to show realistic churn patterns for educational purposes.

## License

Open source - Feel free to modify and distribute.

---

**Ready to Start?**
```bash
python make_data.py
streamlit run app.py
```
