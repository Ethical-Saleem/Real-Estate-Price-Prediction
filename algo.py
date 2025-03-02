Algorithm: Data Cleaning Process
Input: Raw real estate dataset, denote D
Output: Cleaned dataset, denote D'

1. Remove duplicate entries
   for each property record r in D:
      if r is a duplicate:
         remove r from D

2. Handle missing values
   for each feature f in D:
      if missing_percentage(f) > 30%:
         remove feature f from D
      else:
         if f is numerical:
            impute missing values with median(f)
         else:
            impute missing values with mode(f)

3. Remove outliers
   for each numerical feature f in D:
      calculate Q1 = first quartile of f
      calculate Q3 = third quartile of f
      calculate IQR = Q3 - Q1
      set lower_bound = Q1 - 1.5 * IQR
      set upper_bound = Q3 + 1.5 * IQR
      for each value v in feature f:
         if v < lower_bound or v > upper_bound:
            mark as outlier and handle (remove or transform)

4. Return cleaned dataset D'

Algorithm: Real Estate Price Prediction Model Training
Input: Preprocessed dataset D with features X and target prices Y
Output: Trained prediction model M

1. Split D into training set Dtrain and validation set Dval
2. For each model type m in model_list:
    3. Initialize model m with default hyperparameters
    4. Train model m on Dtrain
    5. Predict prices on Dval
    6. Calculate performance metrics (RMSE, MAE, R²)
    7. Store model and metrics
8. Select top k performing models based on validation metrics
9. For each selected model s:
    10. Perform hyperparameter tuning using grid search or Bayesian optimization
    11. Retrain model s with optimal hyperparameters
    12. Evaluate on Dval
13. Select best performing model or create ensemble
14. Return final model M.

Algorithm: SHAP Value Calculation
Input: Trained model M, instance x, background dataset D
Output: SHAP values for all features

1. For each feature i:
   a. Set S = all possible feature subsets excluding feature i
   b. For each subset s in S:
      i. Create two instances:
         - x_with_i: instance with features in s ∪ {i} from x and remaining features from B
         - x_without_i: instance with features in s from x and remaining features from B
      ii. Calculate marginal contribution:
          v(s ∪ {i}) - v(s) = M(x_with_i) - M(x_without_i)
      iii. Weight by |s|!(n-|s|-1)!/n!, where n is total number of features
   c. Sum weighted marginal contributions to get SHAP value for feature i
2. Return SHAP values for all features

background: #4F4F4F80;
