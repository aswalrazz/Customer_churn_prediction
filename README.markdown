# Customer Churn Prediction using Machine Learning

This Python-based machine learning project predicts customer churn (whether a customer will leave a service) using algorithms like Logistic Regression, Random Forest, and XGBoost. It analyzes features such as customer demographics, usage patterns, and subscription details to help businesses identify at-risk customers and implement retention strategies.

## Features
- **Data Preprocessing**: Handles missing values, encodes categorical variables (e.g., one-hot encoding), and scales numerical features.
- **Model Training**: Trains multiple models including Logistic Regression, Random Forest, and XGBoost.
- **Model Evaluation**: Assesses performance with metrics like accuracy, precision, recall, F1-score, and ROC-AUC.
- **Hyperparameter Tuning**: Uses GridSearchCV to optimize model parameters.
- **Visualizations**: Generates confusion matrices, ROC curves, and feature importance plots.
- **Prediction**: Enables predictions on new customer data using saved models.
- **Modular Code**: Structured for easy extension to add new models or features.

## Installation

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook (optional, for running `.ipynb` files)
- Internet connection for installing dependencies

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements
The following dependencies are listed in `requirements.txt`:
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- xgboost==1.7.6
- matplotlib==3.7.2
- seaborn==0.12.2
- jupyter==1.0.0

## Usage
1. **Prepare Data**:
   - Place your dataset (e.g., `churn_data.csv`) in the `data/` folder.
   - Expected columns: `customer_id`, `gender`, `senior_citizen`, `tenure`, `monthly_charges`, `total_charges`, `contract`, `payment_method`, `churn` (Yes/No).
   - Example dataset: Use telecom churn datasets from Kaggle or similar sources.

2. **Run the Code**:
   - **Using Jupyter Notebook**:
     ```bash
     jupyter notebook churn_prediction.ipynb
     ```
     Open the notebook in your browser and run all cells.
   - **Using Python Script**:
     ```bash
     python churn_prediction.py
     ```
     This loads data, preprocesses it, trains models, evaluates performance, and saves the best model (e.g., `best_model.pkl`).

3. **View Results**:
   - Outputs include model accuracies, confusion matrices, ROC curves, and feature importance plots.
   - Example: XGBoost may achieve ~85% accuracy, with `tenure` and `monthly_charges` as top predictors.

4. **Make Predictions**:
   - Load the trained model and predict on new data:
     ```python
     import pickle
     import pandas as pd

     model = pickle.load(open('models/best_model.pkl', 'rb'))
     new_data = pd.DataFrame({
         'tenure': [12], 'monthly_charges': [80.5], 'contract': ['Month-to-month'],
         # Add other required features
     })
     predictions = model.predict(new_data)
     print(predictions)  # e.g., ['Yes'] (churn)
     ```

## Example
- **Input Dataset**: A CSV with columns like `customer_id`, `gender`, `tenure`, `monthly_charges`, `contract`, `churn`.
- **Preprocessing**: One-hot encode categorical variables (e.g., `contract`), scale numerical features (e.g., `tenure`).
- **Training**: Train Logistic Regression, Random Forest, and XGBoost with GridSearchCV for hyperparameter tuning.
- **Output**:
  - Accuracy: ~85% (XGBoost).
  - Feature Importance: `tenure` and `monthly_charges` are top predictors.
  - Visualizations: Confusion matrix and ROC curve saved as plots.
- **Prediction**: For a customer with short `tenure` (12 months) and high `monthly_charges` ($80.5), the model predicts 'Yes' (likely to churn).

## Project Structure
```
customer-churn-prediction/
├── data/
│   └── churn_data.csv     # Dataset (not included in repo)
├── models/
│   └── best_model.pkl     # Trained model
├── churn_prediction.py    # Main Python script
├── churn_prediction.ipynb # Jupyter Notebook version
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your feature"
   ```
4. Push to the branch:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a Pull Request on GitHub.

**Ideas for Improvement**:
- Add more algorithms (e.g., SVM, Neural Networks).
- Incorporate additional features (e.g., customer support interactions).
- Enhance visualizations with interactive plots (e.g., using Plotly).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For issues or suggestions, open an issue on GitHub or email your-email@example.com.

## Acknowledgments
- Built with scikit-learn and XGBoost.
- Inspired by telecom churn datasets from Kaggle.