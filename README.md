# House Price Prediction using K-Nearest Neighbors (KNN)

## Project Description
This machine learning project implements a K-Nearest Neighbors (KNN) regression model to predict house prices based on various features such as location, square footage, number of bedrooms, and other relevant attributes.

## Project Objectives
- Develop an accurate house price prediction model
- Demonstrate KNN algorithm implementation
- Provide insights into real estate price estimation

## Technical Overview

### Methodology
- **Algorithm**: K-Nearest Neighbors Regression
- **Approach**: 
  1. Data preprocessing
  2. Feature scaling
  3. Model training
  4. Hyperparameter tuning
  5. Price prediction

### Key Features
- Feature selection and engineering
- Automated data cleaning
- Cross-validation
- Performance metrics calculation

## Technical Requirements
### Dependencies
- Python 3.7+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### Installation
```bash
# Clone the repository
git clone https://github.com/swaroopchandrashekar/House-price-prediction-using-KNN.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```


## Usage Examples
```python
# Basic prediction workflow
from src.data_preprocessing import load_and_clean_data
from src.model_training import train_knn_model
from src.prediction import predict_house_price

# Load and preprocess data
data = load_and_clean_data('data/raw/house_prices_raw.csv')

# Train KNN model
model = train_knn_model(data)

# Make a prediction
sample_house = {
    'square_feet': 2000,
    'bedrooms': 3,
    'bathrooms': 2,
    'location': 'suburban'
}
predicted_price = predict_house_price(model, sample_house)
print(f"Predicted House Price: ${predicted_price:,.2f}")
```

## Model Performance Metrics
- **Mean Absolute Error (MAE)**: Measures average prediction error
- **Root Mean Squared Error (RMSE)**: Indicates model's prediction accuracy
- **R-squared (R²)**: Explains variance in house prices

## Potential Improvements
- Implement ensemble methods
- Explore advanced feature engineering
- Integrate more complex machine learning algorithms
- Collect more diverse dataset

## Limitations
- Accuracy depends on dataset quality
- May not capture unique property characteristics
- Requires periodic model retraining

## Contributing
1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request


## Contact
- **Name**: Swaroop Chand
- **Email**: swaroopchandrashekar@gmail.com
- **Project Link**: [GitHub Repository](https://github.com/swaroopchandrashekar/House-price-prediction-using-KNN)
