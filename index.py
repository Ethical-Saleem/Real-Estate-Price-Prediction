import math
import json
import folium
import seaborn as sns

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import skew
from utils import initialize
from scipy.stats import normaltest
from media_utils import display_image
from missingno import matrix as missing
from summarytools import dfSummary as summary

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

%matplotlib inline
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['figure.dpi'] = 144

sns.set()

# 1. Load the data
property_sales_data = pd.read_csv('real_estate_dataset.csv', low_memory=False)

# 2. First look at the data
print("Dataset shape:", property_sales_data.shape)
print("\nFirst 5 rows:")
display(property_sales_data.head())
summary(property_sales_data)
# 3. Check data types and basic info
print("\nData types and non-null counts:")
print(property_sales_data.info())

# 4. Basic statistical summary
print("\nStatistical summary:")
display(property_sales_data.describe().T)
# 1. Check for missing values
missing_values = property_sales_data.isnull().sum()
missing_percentages = (missing_values / len(property_sales_data)) * 100
missing_data = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage (%)': missing_percentages
})
missing_data = missing_data[missing_data['Missing Values'] > 0].sort_values('Percentage (%)', ascending=False)
print("\nMissing values summary:")
print(missing_data)
# 3. Identify feature types
numerical_features = [feat for feat in property_sales_data.columns if property_sales_data[feat].dtypes != 'O']
categorical_features = [feature for feature in property_sales_data.columns if property_sales_data[feature].dtype == 'O']

print(f"\nNumber of numerical features: {len(numerical_features)}")
print(f"Number of categorical features: {len(categorical_features)}")
Number of numerical features: 14
Number of categorical features: 6
for column in missing_data.index:
    missing_pct = missing_data.loc[column, 'Percentage (%)']
    
    if missing_pct > 30:
        print(f"Column {column} has {missing_pct:.2f}% missing values. Consider dropping this column.")
    
    elif column in numerical_features:
        median_value = property_sales_data[column].median()
        property_sales_data[column].fillna(median_value, inplace=True)
        print(f"Imputed missing values in {column} with median: {median_value}")
        
    elif column in categorical_features:
        mode_value = property_sales_data[column].mode()[0]
        property_sales_data[column].fillna(mode_value, inplace=True)
        print(f"Imputed missing values in {column} with mode: {mode_value}")
        
# 4. Check for duplicates
duplicates = property_sales_data.duplicated().sum()
print(f"\nNumber of duplicate entries: {duplicates}")
if duplicates > 0:
    property_sales_data.drop_duplicates(inplace=True)
    print(f"Removed {duplicates} duplicate entries.")

# 1. Basic statistics of the target variable
print("\nSale Price Statistics:")
print(property_sales_data['Sale_price'].describe())
# Log-transformed distribution
plt.subplot(1, 2, 1)
log_sale_price = np.log1p(property_sales_data['Sale_price'])
sns.histplot(log_sale_price, kde=True)
plt.title('Log-Transformed Sale Price', fontsize=8, pad=15)
plt.xlabel('Log(Sale Price)')

# Square root transformed distribution
plt.subplot(1, 2, 2)
sqrt_sale_price = np.sqrt(property_sales_data['Sale_price'])
sns.histplot(sqrt_sale_price, kde=True)
plt.title('Square Root Transformed Sale Price', fontsize=8, pad=15)
plt.xlabel('Sqrt(Sale Price)')

plt.tight_layout(pad=3.0)
plt.show()
# 3. Compare skewness of different transformations
transformations = {
    'Original': property_sales_data['Sale_price'],
    'Log': log_sale_price,
    'Square Root': sqrt_sale_price
}

for name, data in transformations.items():
    print(f"Skewness of {name} Sale Price: {skew(data)}")
    
    # 2. Statistical summary of numerical features
print("\nNumerical features statistics:")
display(property_sales_data[numerical_features].describe().T)
# 3. Analyze distribution and normality of numerical features
features_list = []
test_statistics = []
p_values = []
skewness_values = []

for feature in property_sales_data[numerical_features].columns:
    # Perform normality test
    test_statistic, p_value = normaltest(property_sales_data[feature])

    # Append results to respective lists
    features_list.append(feature)
    test_statistics.append(test_statistic)
    p_values.append(p_value)

    # Calculate skewness
    skewness = skew(property_sales_data[feature])
    skewness_values.append(skewness)

# Create a DataFrame to display the results
results_df = pd.DataFrame({'Feature': features_list,
                           'Test Statistics': test_statistics,
                           'p-value': p_values,
                           'Skew': skewness_values})
print("\nNormality test results:")
display(results_df)
# 4. Visualize distributions of key numerical features
sample_data = property_sales_data.sample(n=10000, random_state=42)

key_features = ['Sale_price', 'FinishedSqft', 'Lotsize', 'Rooms', 'Bdrms', 'Stories']
plt.figure(figsize=(15, 15))
for i, feature in enumerate(key_features):
    plt.subplot(2, 3, i+1)
    sns.histplot(sample_data[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()
# 1. Create a function to detect outliers using box plots
def plot_boxplots(df, features, rows=3, cols=3):
    plt.figure(figsize=(15, 15))
    for i, feature in enumerate(features):
        if i < rows * cols:
            plt.subplot(rows, cols, i+1)
            sns.boxplot(y=df[feature])
            plt.title(f'Boxplot of {feature}')
    plt.tight_layout()
    plt.show()
    # 2. Plot boxplots for numerical features to identify outliers
numerical_features_subset = [f for f in numerical_features if f != 'Sale_price'][:9]
plot_boxplots(property_sales_data, numerical_features_subset)
# 3. Define function to detect outliers using IQR
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    return outliers, lower_bound, upper_bound
  
  # 4. Create outlier summary for all numerical features
outlier_summary = pd.DataFrame(columns=['Feature', 'Number of Outliers', 'Percentage (%)', 'Lower Bound', 'Upper Bound'])

for feature in numerical_features:
    outliers, lower_bound, upper_bound = detect_outliers(property_sales_data, feature)
    percentage = len(outliers) / len(property_sales_data) * 100
    
    outlier_summary = outlier_summary._append({
        'Feature': feature,
        'Number of Outliers': len(outliers),
        'Percentage (%)': percentage,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound
    }, ignore_index=True)

print("\nOutlier summary:")
display(outlier_summary.sort_values('Percentage (%)', ascending=False))
# 5. Define function to treat outliers
def treat_outliers(df, column, method='cap'):
    outliers, lower_bound, upper_bound = detect_outliers(df, column)
    
    if method == 'cap':
        # Cap the outliers at the bounds
        df[column] = np.where(
            df[column] > upper_bound,
            upper_bound,
            np.where(
                df[column] < lower_bound,
                lower_bound,
                df[column]
            )
        )
    elif method == 'remove':
        # Remove the outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df
  # 6. Treat outliers for numerical features with significant outliers (e.g., > 5%)
for feature in outlier_summary[outlier_summary['Percentage (%)'] > 5]['Feature']:
    print(f"Treating outliers in {feature}")
    property_sales_data = treat_outliers(property_sales_data, feature, method='cap')
    # 7. Verify the impact of outlier treatment
for feature in outlier_summary[outlier_summary['Percentage (%)'] > 5]['Feature']:
    outliers_after, _, _ = detect_outliers(property_sales_data, feature)
    print(f"Feature {feature} now has {len(outliers_after)} outliers after treatment.")
    # 1. Compute correlation matrix for all numerical features
correlation_matrix = property_sales_data[numerical_features].corr(method='spearman')
# 3. Focused correlation analysis with key features
key_features_corr = ['Sale_price', 'District', 'FinishedSqft', 'Lotsize', 'nbhd']
correlation_matrix_key = property_sales_data[key_features_corr].corr(method='spearman')
housing_features = ['Sale_price', 'Rooms', 'Fbath', 'Hbath', 'Bdrms', 'Stories']
correlation_matrix_housing = property_sales_data[housing_features].corr(method='spearman')
# 1. Create regression plots for key variables
fig, axes = plt.subplots(4, 1, figsize=(5, 15))
for variable, subplot in zip(['District', 'FinishedSqft', 'Lotsize', 'nbhd'], axes.flatten()):
    sns.regplot(data=property_sales_data, y='Sale_price', x=variable, ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
plt.tight_layout()
plt.show()
# 2. Filter and visualize Lotsize relationship (removing extreme values)
filtered_lotsize_data = property_sales_data[property_sales_data['Lotsize'] < 4*1e6]
plt.figure(figsize=(10, 6))
sns.regplot(data=filtered_lotsize_data, y='Sale_price', x='Lotsize')
plt.title('Sale Price vs Lot Size (Filtered)')
plt.show()
# 3. Visualize relationship with housing features
housing_vars = ['Rooms', 'Fbath', 'Hbath', 'Bdrms', 'Stories']
fig, axes = plt.subplots(len(housing_vars), 1, figsize=(8, 20))
for variable, subplot in zip(housing_vars, axes.flatten()):
    sns.regplot(data=property_sales_data, y='Sale_price', x=variable, ax=subplot)
    subplot.set_title(f'Sale Price vs {variable}')
plt.tight_layout()
plt.show()
# 4. Create pairplot for key features
sample_data_pair = property_sales_data.sample(n=10000, random_state=42)
key_features_pair = ['Sale_price', 'FinishedSqft', 'Lotsize', 'Rooms', 'Bdrms']
sns.pairplot(sample_data_pair[key_features_pair], height=2)
plt.suptitle('Pairwise Relationships Between Key Features', y=1.02)
plt.show()
# 1. Preview categorical features
print("\nCategorical features preview:")
display(property_sales_data[categorical_features].head())
# 3. Analyze each categorical feature
for feature in categorical_features:
    value_counts = property_sales_data[feature].value_counts()
    
    # Only plot if there aren't too many categories
    if len(value_counts) <= 20:
        plt.figure(figsize=(10, 6))
        sns.countplot(y=property_sales_data[feature], order=value_counts.index)
        plt.title(f'Distribution of {feature}')
        plt.tight_layout()
        plt.show()
        
        # Also show the relationship with the target variable
        plt.figure(figsize=(12, 6))
        sns.boxplot(x=feature, y='Sale_price', data=property_sales_data, order=value_counts.index)
        plt.title(f'Sale Price by {feature}')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print(f"{feature} has {len(value_counts)} unique values. Too many to plot.")
        print(f"Top 10 most frequent values: \n{value_counts.head(10)}")
        # 1. Bin continuous features into categories

# Binning nbhd (neighborhood) feature
def bin_area(x):
    if (x > 0) and (x <= 1.09*1e3):
        return '1'
    elif (x > 1.09*1e3) and (x <= 1.28*1e3):
        return '2'
    elif (x > 1.28*1e3) and (x <= 1.638*1e3):
        return '3'
    elif (x > 1.638*1e3):
        return '4'
    else:
        return 'Unknown'

property_sales_data['livable_bin'] = property_sales_data['nbhd'].apply(bin_area)
# Calculate statistics by livable_bin
grouped_sales_stats = property_sales_data.groupby('livable_bin')['Sale_price'].describe()
print("\nSale price statistics by livable area bin:")
display(grouped_sales_stats)
# Visualize median sale price by livable_bin
median_sale_price = grouped_sales_stats['50%'].reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(x='livable_bin', y='50%', data=median_sale_price)
plt.title('Median Sale Price by Livable Area Bin')
plt.xlabel('Livable Area Bin')
plt.ylabel('Median Sale Price ($)')
plt.show()
# 4. Create new derived features
property_sales_data['price_per_sqft'] = property_sales_data['Sale_price'] / property_sales_data['FinishedSqft']
property_sales_data['total_bathrooms'] = property_sales_data['Fbath'] + (0.5 * property_sales_data['Hbath'])
property_sales_data['room_ratio'] = property_sales_data['Bdrms'] / property_sales_data['Rooms']
property_sales_data['rooms_per_square_feet'] = property_sales_data['Rooms'] / property_sales_data['FinishedSqft']
# 5. Log transform skewed numerical features
skewed_features = [
    feature for feature in numerical_features 
    if feature != 'Sale_price' and abs(skew(property_sales_data[feature])) > 0.75
]

for feature in skewed_features:
    property_sales_data[f'{feature}_log'] = np.log1p(property_sales_data[feature])
    print(f"Log-transformed {feature} due to skewness of {skew(property_sales_data[feature]):.2f}")