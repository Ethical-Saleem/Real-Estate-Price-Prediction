
import math
import json
import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, boxcox
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, validation_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Suppress warnings to keep output clean
warnings.filterwarnings('ignore')
# Set pandas to display all columns when printing DataFrames
pd.set_option('display.max_columns', None)
# Set default matplotlib figure size for better visualization
plt.rcParams['figure.figsize'] = (12, 8)

class RandomForestRealEstatePipeline:
    """
    Random Forest pipeline with feature engineering and exploratory data analysis
    
    This class implements a comprehensive machine learning pipeline specifically designed 
    for real estate price prediction. It includes:
    - Advanced data analysis and quality assessment
    - Feature engineering with domain knowledge
    - Intelligent outlier detection and treatment
    - Hyperparameter optimization
    - Comprehensive model evaluation and interpretation
    
    The pipeline is designed to be educational, with comments explaining
    each decision and methodology.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the pipeline with default parameters
        
        Args:
            random_state (int): Random seed for reproducibility
                              This ensures that results are consistent across runs
        """
        self.random_state = random_state
        self.rf_model = None  # Will store the trained Random Forest model
        self.scaler = StandardScaler()  # For feature scaling
        self.label_encoders = {}  # For categorical variable encoding
        self.feature_names = []  # Will store final feature names
        self.feature_importance_df = None  # For feature importance analysis
        self.data_analysis_report = {}  # Stores comprehensive data analysis results
        self.engineered_features = []  # Tracks newly created features
        
    def load_and_preprocess_data(self, filepath):
        """
        Load and preprocess the real estate data with comprehensive analysis
        
        This method is the entry point for data processing. It loads the CSV file
        and performs initial data quality assessment to understand the dataset
        structure and identify potential issues.
        
        Args:
            filepath (str): Path to the CSV file containing real estate data
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("Loading and analyzing data...")
        # Load data with low_memory=False to avoid dtype warnings
        self.df = pd.read_csv(filepath, low_memory=False)
        print(f"Dataset shape: {self.df.shape}")
        
        # Store original data for reference - useful for comparison later
        self.original_df = self.df.copy()
        
        # Perform comprehensive data analysis to understand the dataset
        self._comprehensive_data_analysis()
        
        return self.df
    
    def _comprehensive_data_analysis(self):
        """
        Comprehensive data quality and exploratory data analysis
        
        This method performs a thorough analysis of the dataset to understand:
        - Data structure and types
        - Missing values patterns
        - Distribution characteristics
        - Feature relationships
        
        This analysis is crucial for making informed decisions about preprocessing
        and feature engineering strategies.
        """
        print("\n" + "="*50)
        print("COMPREHENSIVE DATA ANALYSIS REPORT")
        print("="*50)
        
        # Basic statistics - understand dataset size and memory usage
        print(f"Dataset dimensions: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Data types analysis - distinguish between numerical and categorical features
        # This is important because different feature types require different preprocessing
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nFeature types:")
        print(f"  Numerical: {len(numeric_cols)}")
        print(f"  Categorical: {len(categorical_cols)}")
        
        # Missing values analysis - critical for determining imputation strategies
        missing_analysis = self.df.isnull().sum()
        missing_pct = (missing_analysis / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing_Count': missing_analysis,
            'Missing_Percentage': missing_pct
        }).sort_values('Missing_Percentage', ascending=False)
        
        print(f"\nMissing values:")
        print(missing_df[missing_df['Missing_Count'] > 0])
        
        # Cardinality analysis for categorical variables
        # High cardinality features may need special encoding techniques
        if categorical_cols:
            print(f"\nCategorical variables cardinality:")
            for col in categorical_cols:
                unique_count = self.df[col].nunique()
                print(f"  {col}: {unique_count} unique values")
        
        # Numerical variables distribution analysis
        # Skewness and kurtosis help identify features that may need transformation
        if numeric_cols:
            print(f"\nNumerical variables distribution analysis:")
            for col in numeric_cols:
                skewness = skew(self.df[col].dropna())
                kurt = kurtosis(self.df[col].dropna())
                print(f"  {col}: skewness={skewness:.3f}, kurtosis={kurt:.3f}")
        
        # Store analysis results for later use
        self.data_analysis_report = {
            'shape': self.df.shape,
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'missing_analysis': missing_df,
            'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Create visualizations to better understand data patterns
        self._create_data_quality_visualizations()
        
    def _create_data_quality_visualizations(self):
        """
        Create comprehensive data quality visualizations
        
        Visualizations help identify patterns that might not be obvious from
        numerical summaries alone. This includes:
        - Missing data patterns
        - Feature correlations
        - Distribution shapes
        """
        print("\nCreating data quality visualizations...")
        
        # Missing values heatmap - shows patterns in missing data
        # Patterns in missing data can indicate systematic issues
        plt.figure(figsize=(12, 8))
        sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap')
        plt.show()
        
        # Correlation matrix for numerical features
        # High correlations can indicate multicollinearity issues
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(14, 10))
            correlation_matrix = self.df[numeric_cols].corr()
            # Use upper triangle mask to avoid redundant information
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                       center=0, square=True, linewidths=0.5)
            plt.title('Correlation Matrix - Numerical Features')
            plt.show()
        
        # Distribution plots for key numerical features
        # Understanding distributions helps choose appropriate transformations
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:6]  # Top 6
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for i, col in enumerate(numeric_cols):
                if i < 6:
                    sns.histplot(self.df[col].dropna(), kde=True, ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
            
            plt.tight_layout()
            plt.show()
    
    def advanced_outlier_detection(self, methods=['iqr', 'zscore', 'isolation']):
        """
        Advanced outlier detection using multiple methods
        
        Outliers can significantly impact model performance. This method uses
        multiple detection techniques to identify potential outliers:
        - IQR (Interquartile Range): Traditional method, robust to extreme values
        - Z-score: Assumes normal distribution, sensitive to extreme values
        - Modified Z-score: Uses median instead of mean, more robust
        
        Args:
            methods (list): List of outlier detection methods to use
            
        Returns:
            pd.DataFrame: Summary of outliers detected by each method
        """
        print(f"\nAdvanced outlier detection using methods: {methods}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_summary = {}
        
        for col in numeric_cols:
            outlier_summary[col] = {}
            col_data = self.df[col].dropna()
            
            # IQR method - based on quartiles, robust to extreme values
            if 'iqr' in methods:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                # Standard rule: outliers are beyond 1.5 * IQR from quartiles
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                outlier_summary[col]['iqr'] = iqr_outliers
            
            # Z-score method - assumes normal distribution
            if 'zscore' in methods:
                z_scores = np.abs(stats.zscore(col_data))
                # Standard rule: |z-score| > 3 indicates outlier
                zscore_outliers = (z_scores > 3).sum()
                outlier_summary[col]['zscore'] = zscore_outliers
            
            # Modified Z-score method - uses median instead of mean
            if 'modified_zscore' in methods:
                median = np.median(col_data)
                # MAD = Median Absolute Deviation
                mad = np.median(np.abs(col_data - median))
                # 0.6745 is the 75th percentile of the standard normal distribution
                modified_z_scores = 0.6745 * (col_data - median) / mad
                # Modified rule: |modified z-score| > 3.5 indicates outlier
                modified_zscore_outliers = (np.abs(modified_z_scores) > 3.5).sum()
                outlier_summary[col]['modified_zscore'] = modified_zscore_outliers
        
        # Create outlier summary DataFrame for easy analysis
        outlier_df = pd.DataFrame(outlier_summary).T
        outlier_df['total_records'] = len(self.df)
        
        print("Outlier detection summary:")
        print(outlier_df)
        
        return outlier_df
    
    def advanced_feature_engineering(self):
        """
        Feature engineering with domain knowledge
        
        This method creates new features based on domain knowledge of real estate:
        - Property ratios (lot utilization, room efficiency)
        - Derived categories (size categories, age groups)
        - Neighborhood statistics (price comparisons)
        - Interaction features (luxury scores)
        - Statistical transformations (handling skewness)
        
        The goal is to create features that capture meaningful patterns
        that the model can learn from.
        """
        print("\nAdvanced feature engineering...")
        
        # Store original feature count to track new features
        original_features = self.df.columns.tolist()
        
        # 1. PROPERTY SIZE AND SPACE FEATURES
        # These features capture how efficiently space is used
        if 'FinishedSqft' in self.df.columns and 'Lotsize' in self.df.columns:
            # Lot utilization ratio - how much of the lot is used by the house
            self.df['lot_utilization_ratio'] = self.df['FinishedSqft'] / self.df['Lotsize']
            # Handle division by zero or infinity
            self.df['lot_utilization_ratio'] = self.df['lot_utilization_ratio'].replace([np.inf, -np.inf], 0)
            
            # Property size categories - convert continuous to categorical
            # This helps capture non-linear relationships
            self.df['property_size_category'] = pd.cut(
                self.df['FinishedSqft'], 
                bins=[0, 1000, 1500, 2500, 5000, np.inf],
                labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large']
            )
            
            # Lot size categories - similar reasoning
            self.df['lot_size_category'] = pd.cut(
                self.df['Lotsize'],
                bins=[0, 5000, 10000, 20000, 50000, np.inf],
                labels=['Small Lot', 'Medium Lot', 'Large Lot', 'Very Large Lot', 'Estate']
            )
        
        # 2. ROOM AND BATHROOM FEATURES
        # These capture living space efficiency and comfort
        if all(col in self.df.columns for col in ['Rooms', 'Bdrms', 'Fbath', 'Hbath']):
            # Total bathrooms - half baths count as 0.5
            self.df['total_bathrooms'] = self.df['Fbath'] + (0.5 * self.df['Hbath'])
            
            # Bedroom to total room ratio - indicates room allocation
            self.df['bedroom_ratio'] = self.df['Bdrms'] / self.df['Rooms']
            
            # Bathroom to bedroom ratio - indicates convenience level
            self.df['bathroom_bedroom_ratio'] = self.df['total_bathrooms'] / self.df['Bdrms']
            self.df['bathroom_bedroom_ratio'] = self.df['bathroom_bedroom_ratio'].replace([np.inf, -np.inf], 0)
            
            # Room efficiency features - space utilization
            if 'FinishedSqft' in self.df.columns:
                self.df['room_efficiency'] = self.df['Rooms'] / self.df['FinishedSqft']
                self.df['space_per_room'] = self.df['FinishedSqft'] / self.df['Rooms']
        
        # 3. PRICE-RELATED FEATURES
        # These help understand value patterns
        if 'Sale_price' in self.df.columns:
            if 'FinishedSqft' in self.df.columns:
                # Price per square foot - key real estate metric
                self.df['price_per_sqft'] = self.df['Sale_price'] / self.df['FinishedSqft']
                
                # Price per square foot categories - market segments
                self.df['price_per_sqft_category'] = pd.cut(
                    self.df['price_per_sqft'],
                    bins=5,
                    labels=['Budget', 'Economy', 'Mid-range', 'Premium', 'Luxury']
                )
            
            # Log transformation for price - helps with skewed distributions
            # log1p is log(1+x) which handles zero values safely
            self.df['log_sale_price'] = np.log1p(self.df['Sale_price'])
        
        # 4. NEIGHBORHOOD AND LOCATION FEATURES
        # Location is crucial in real estate - these features capture location value
        if 'nbhd' in self.df.columns:
            # Neighborhood price statistics - market positioning
            nbhd_stats = self.df.groupby('nbhd')['Sale_price'].agg(['mean', 'median', 'std']).reset_index()
            nbhd_stats.columns = ['nbhd', 'nbhd_mean_price', 'nbhd_median_price', 'nbhd_price_std']
            self.df = self.df.merge(nbhd_stats, on='nbhd', how='left')
            
            # Price deviation from neighborhood average - relative value
            self.df['price_deviation_from_nbhd'] = (self.df['Sale_price'] - self.df['nbhd_mean_price']) / self.df['nbhd_mean_price']
            
            # Neighborhood ranking based on median price - prestige indicator
            nbhd_ranking = self.df.groupby('nbhd')['Sale_price'].median().rank(ascending=False).reset_index()
            nbhd_ranking.columns = ['nbhd', 'nbhd_price_rank']
            self.df = self.df.merge(nbhd_ranking, on='nbhd', how='left')
        
        # 5. PROPERTY AGE AND CONDITION FEATURES
        # Age affects property value in complex ways
        if 'Built' in self.df.columns:
            current_year = datetime.now().year
            self.df['property_age'] = current_year - self.df['Built']
            
            # Age categories - different age ranges have different market appeal
            self.df['age_category'] = pd.cut(
                self.df['property_age'],
                bins=[0, 5, 15, 30, 50, np.inf],
                labels=['New', 'Modern', 'Established', 'Mature', 'Historic']
            )
            
            # Depreciation factor - exponential decay model
            # Assumes property value decreases exponentially with age
            self.df['depreciation_factor'] = np.exp(-self.df['property_age'] / 50)
        
        # 6. INTERACTION FEATURES
        # These capture complex relationships between features
        if all(col in self.df.columns for col in ['FinishedSqft', 'Lotsize', 'total_bathrooms', 'Rooms']):
            # Property luxury score - composite feature
            # Weighted combination of size, amenities, and space
            self.df['luxury_score'] = (
                (self.df['FinishedSqft'] / self.df['FinishedSqft'].median()) * 0.3 +
                (self.df['total_bathrooms'] / self.df['total_bathrooms'].median()) * 0.2 +
                (self.df['Rooms'] / self.df['Rooms'].median()) * 0.2 +
                (self.df['Lotsize'] / self.df['Lotsize'].median()) * 0.3
            )
        
        # 7. STATISTICAL TRANSFORMATIONS
        # Handle skewed distributions for better model performance
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Sale_price', 'District', 'nbhd'] and col in original_features:
                # Check if feature is highly skewed
                col_skew = skew(self.df[col].dropna())
                if abs(col_skew) > 1.5:  # Rule of thumb: |skewness| > 1.5 is highly skewed
                    # Box-Cox transformation for positive values
                    if (self.df[col] > 0).all():
                        try:
                            self.df[f'{col}_boxcox'], _ = boxcox(self.df[col] + 1)
                            self.engineered_features.append(f'{col}_boxcox')
                        except:
                            # Fall back to log transformation if Box-Cox fails
                            self.df[f'{col}_log'] = np.log1p(self.df[col])
                            self.engineered_features.append(f'{col}_log')
                    else:
                        # Log transformation for features with negative values
                        self.df[f'{col}_log'] = np.log1p(self.df[col] - self.df[col].min() + 1)
                        self.engineered_features.append(f'{col}_log')
        
        # 8. CLUSTERING FEATURES
        # Discover hidden patterns in the data
        self._create_clustering_features()
        
        # 9. POLYNOMIAL FEATURES (selective)
        # Capture non-linear relationships
        self._create_polynomial_features()
        
        # Track engineered features
        new_features = [col for col in self.df.columns if col not in original_features]
        self.engineered_features.extend(new_features)
        
        print(f"Created {len(new_features)} new features:")
        for feature in new_features[:10]:  # Show first 10
            print(f"  - {feature}")
        if len(new_features) > 10:
            print(f"  ... and {len(new_features) - 10} more")
        
        print(f"Total features: {len(self.df.columns)}")
        
        return self.df
    
    def _create_clustering_features(self):
        """
        Create clustering-based features
        
        Clustering helps identify groups of similar properties.
        These clusters can reveal market segments or property types
        that aren't obvious from individual features.
        """
        print("Creating clustering features...")
        
        # Select key features for clustering
        clustering_features = []
        if 'FinishedSqft' in self.df.columns:
            clustering_features.append('FinishedSqft')
        if 'Lotsize' in self.df.columns:
            clustering_features.append('Lotsize')
        if 'total_bathrooms' in self.df.columns:
            clustering_features.append('total_bathrooms')
        if 'Rooms' in self.df.columns:
            clustering_features.append('Rooms')
        
        if len(clustering_features) >= 2:
            # Prepare data for clustering - handle missing values
            cluster_data = self.df[clustering_features].fillna(self.df[clustering_features].median())
            
            # Standardize features - clustering algorithms are sensitive to scale
            scaler = StandardScaler()
            cluster_data_scaled = scaler.fit_transform(cluster_data)
            
            # K-means clustering - 5 clusters for property types
            kmeans = KMeans(n_clusters=5, random_state=self.random_state)
            self.df['property_cluster'] = kmeans.fit_predict(cluster_data_scaled)
            
            # Cluster-based features - add cluster statistics
            cluster_stats = self.df.groupby('property_cluster')['Sale_price'].agg(['mean', 'std']).reset_index()
            cluster_stats.columns = ['property_cluster', 'cluster_mean_price', 'cluster_price_std']
            self.df = self.df.merge(cluster_stats, on='property_cluster', how='left')
    
    def _create_polynomial_features(self):
        """
        Create selective polynomial features
        
        Polynomial features capture non-linear relationships and interactions
        between features. We focus on interaction terms rather than squared terms
        to avoid overfitting while capturing important feature interactions.
        """
        print("Creating polynomial features...")
        
        # Select key features for polynomial transformation
        poly_features = []
        if 'FinishedSqft' in self.df.columns:
            poly_features.append('FinishedSqft')
        if 'Lotsize' in self.df.columns:
            poly_features.append('Lotsize')
        if 'total_bathrooms' in self.df.columns:
            poly_features.append('total_bathrooms')
        
        if len(poly_features) >= 2:
            # Create polynomial features (degree 2, interaction only)
            poly_data = self.df[poly_features].fillna(self.df[poly_features].median())
            
            # Standardize before polynomial transformation to prevent numerical issues
            scaler = StandardScaler()
            poly_data_scaled = scaler.fit_transform(poly_data)
            
            # Create polynomial features - interaction_only=True avoids x^2 terms
            poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            poly_transformed = poly.fit_transform(poly_data_scaled)
            
            # Add only the interaction terms
            feature_names = poly.get_feature_names_out(poly_features)
            interaction_features = [name for name in feature_names if ' ' in name]
            
            # Add selected interaction features (limit to prevent overfitting)
            for i, feature_name in enumerate(interaction_features[:5]):  # Limit to 5 interactions
                clean_name = feature_name.replace(' ', '_x_')
                interaction_idx = list(feature_names).index(feature_name)
                self.df[f'poly_{clean_name}'] = poly_transformed[:, interaction_idx]
    
    def advanced_data_preprocessing(self):
        """
        Advanced preprocessing with intelligent missing value handling
        
        This method implements sophisticated preprocessing strategies:
        - Intelligent missing value imputation based on missing percentage
        - Advanced outlier treatment
        - Data validation and quality checks
        
        The preprocessing strategy adapts to the characteristics of each feature.
        """
        print("\nAdvanced data preprocessing...")
        
        # 1. INTELLIGENT MISSING VALUE IMPUTATION
        # Different strategies based on data characteristics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # For numerical columns - strategy depends on missing percentage
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
                
                if missing_pct > 50:
                    # Drop features with too many missing values
                    print(f"Dropping {col} due to {missing_pct:.1f}% missing values")
                    self.df.drop(col, axis=1, inplace=True)
                elif missing_pct > 10:
                    # Use KNN imputation for moderate missing values
                    # KNN considers relationships between features
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=5)
                    self.df[col] = imputer.fit_transform(self.df[[col]])
                    print(f"KNN imputed {col} ({missing_pct:.1f}% missing)")
                else:
                    # Use median for low missing values (robust to outliers)
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                    print(f"Median imputed {col} ({missing_pct:.1f}% missing)")
        
        # For categorical columns - use mode or create 'Unknown' category
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                missing_pct = (self.df[col].isnull().sum() / len(self.df)) * 100
                
                if missing_pct > 50:
                    print(f"Dropping {col} due to {missing_pct:.1f}% missing values")
                    self.df.drop(col, axis=1, inplace=True)
                else:
                    # Use mode (most frequent value) for categorical variables
                    mode_value = self.df[col].mode()[0] if not self.df[col].mode().empty else 'Unknown'
                    self.df[col].fillna(mode_value, inplace=True)
                    print(f"Mode imputed {col} ({missing_pct:.1f}% missing)")
        
        # 2. OUTLIER TREATMENT
        self._intelligent_outlier_treatment()
        
        # 3. REMOVE DUPLICATES
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            self.df.drop_duplicates(inplace=True)
            print(f"Removed {duplicates} duplicate rows")
        
        # 4. DATA VALIDATION
        self._validate_data_quality()
        
        return self.df
    
    def _intelligent_outlier_treatment(self):
        """
        Intelligent outlier treatment based on feature characteristics
        
        Different outlier treatment strategies based on the percentage of outliers:
        - High percentage (>10%): Cap outliers (winsorizing)
        - Moderate percentage (5-10%): Use percentile-based capping
        - Low percentage (<5%): Keep outliers (might be legitimate extreme values)
        """
        print("Intelligent outlier treatment...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col == 'Sale_price':
                continue  # Don't treat outliers in target variable
            
            # Calculate outlier bounds using IQR method
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers
            outliers_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            outliers_count = outliers_mask.sum()
            outliers_pct = (outliers_count / len(self.df)) * 100
            
            if outliers_pct > 10:
                # Cap outliers for features with many outliers
                self.df[col] = np.where(
                    self.df[col] > upper_bound, upper_bound,
                    np.where(self.df[col] < lower_bound, lower_bound, self.df[col])
                )
                print(f"Capped {outliers_count} outliers in {col} ({outliers_pct:.1f}%)")
            elif outliers_pct > 5:
                # Winsorize for moderate outliers (use 5th and 95th percentiles)
                self.df[col] = np.where(
                    self.df[col] > self.df[col].quantile(0.95), self.df[col].quantile(0.95),
                    np.where(self.df[col] < self.df[col].quantile(0.05), self.df[col].quantile(0.05), self.df[col])
                )
                print(f"Winsorized {outliers_count} outliers in {col} ({outliers_pct:.1f}%)")
                
    def _validate_data_quality(self):
        """Validate data quality after preprocessing"""
        print("Validating data quality...")
        
        # Check for remaining missing values
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            print(f"Warning: {missing_count} missing values remain")
        
        # Check for infinite values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                print(f"Warning: {inf_count} infinite values in {col}")
                self.df[col] = self.df[col].replace([np.inf, -np.inf], self.df[col].median())
        
        # Check for constant features
        constant_features = [col for col in self.df.columns if self.df[col].nunique() <= 1]
        if constant_features:
            print(f"Removing {len(constant_features)} constant features: {constant_features}")
            self.df.drop(constant_features, axis=1, inplace=True)
        
        print(f"Final dataset shape: {self.df.shape}")
    
    def prepare_features_target(self, target_column='Sale_price'):
        """Advanced feature preparation with encoding and scaling"""
        print(f"\nPreparing features and target: {target_column}")
        
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        # Separate features and target
        X = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        
        # Handle categorical variables with advanced encoding
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            unique_count = X[col].nunique()
            
            if unique_count <= 10:
                # Use label encoding for low cardinality
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            else:
                # Use target encoding for high cardinality
                target_mean = self.df.groupby(col)[target_column].mean()
                X[col] = X[col].map(target_mean).fillna(target_mean.mean())
                print(f"Applied target encoding to {col} ({unique_count} categories)")
        
        # Remove highly correlated features
        X = self._remove_highly_correlated_features(X)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Final feature set: {len(self.feature_names)} features")
        print(f"Target variable shape: {y.shape}")
        
        return X, y
    
    def _remove_highly_correlated_features(self, X, threshold=0.95):
        """Remove highly correlated features"""
        print("Removing highly correlated features...")
        
        # Calculate correlation matrix
        corr_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features to drop
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        
        if to_drop:
            X = X.drop(to_drop, axis=1)
            print(f"Removed {len(to_drop)} highly correlated features: {to_drop}")
        
        return X
    
    def train_advanced_random_forest(self, X, y, test_size=0.2, optimize_hyperparameters=True):
        """Train Random Forest with advanced techniques"""
        print("\nTraining Advanced Random Forest...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=None
        )
        
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        
        if optimize_hyperparameters:
            print("Advanced hyperparameter optimization...")
            self.rf_model = self._advanced_hyperparameter_tuning(X_train, y_train)
        else:
            # Use optimized default parameters
            self.rf_model = RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # Train model
        self.rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.rf_model.predict(X_train)
        y_pred_test = self.rf_model.predict(X_test)
        
        self.y_pred_train = y_pred_train
        self.y_pred_test = y_pred_test
        
        # Calculate comprehensive metrics
        train_metrics = self._calculate_comprehensive_metrics(y_train, y_pred_train)
        test_metrics = self._calculate_comprehensive_metrics(y_test, y_pred_test)
        
        print(f"\nAdvanced Training Results:")
        print(f"Train R²: {train_metrics['r2']:.4f}")
        print(f"Test R²: {test_metrics['r2']:.4f}")
        print(f"Train RMSE: ${train_metrics['rmse']:,.2f}")
        print(f"Test RMSE: ${test_metrics['rmse']:,.2f}")
        print(f"Train MAE: ${train_metrics['mae']:,.2f}")
        print(f"Test MAE: ${test_metrics['mae']:,.2f}")
        print(f"Train MAPE: {train_metrics['mape']:.2f}%")
        print(f"Test MAPE: {test_metrics['mape']:.2f}%")
        print(f"Train MSE: ${train_metrics['mse']:,.2f}")
        print(f"Test MSE: ${test_metrics['mse']:,.2f}")
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'model': self.rf_model
        }
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """Optimize Random Forest hyperparameters"""
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        rf_base = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'r2': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred)
        }
    
    def analyze_feature_importance(self, top_n=20):
        """Analyze and visualize feature importance"""
        if self.rf_model is None:
            raise ValueError("Model not trained yet. Call train_random_forest() first.")
        
        print(f"\nAnalyzing feature importance (top {top_n})...")
        
        # Get feature importance
        importance = self.rf_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance_df = feature_importance
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(top_n)
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title(f'Top {top_n} Feature Importance - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        # Permutation importance for more robust analysis
        print("Calculating permutation importance...")
        perm_importance = permutation_importance(
            self.rf_model, self.X_test, self.y_test, 
            n_repeats=10, random_state=self.random_state
        )
        
        perm_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        # Plot permutation importance
        plt.figure(figsize=(12, 8))
        top_perm_features = perm_importance_df.head(top_n)
        plt.barh(range(len(top_perm_features)), top_perm_features['importance_mean'])
        plt.yticks(range(len(top_perm_features)), top_perm_features['feature'])
        plt.xlabel('Permutation Importance')
        plt.title(f'Top {top_n} Permutation Importance - Random Forest')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return feature_importance, perm_importance_df
    
    def cross_validate_model(self, X, y, cv=5):
        """Perform cross-validation"""
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        if self.rf_model is None:
            raise ValueError("Model not trained yet. Call train_random_forest() first.")
        
        cv_scores = cross_val_score(
            self.rf_model, X, y, cv=cv, 
            scoring='neg_mean_squared_error'
        )
        
        cv_rmse = np.sqrt(-cv_scores)
        
        print(f"Cross-validation RMSE: {cv_rmse.mean():.4f} (+/- {cv_rmse.std() * 2:.4f})")
        print(f"Individual fold RMSE: {cv_rmse}")
        
        return cv_rmse
    
    def visualize_predictions(self):
        """Visualize model predictions"""
        if self.rf_model is None:
            raise ValueError("Model not trained yet. Call train_random_forest() first.")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted - Training
        axes[0, 0].scatter(self.y_train, self.y_pred_train, alpha=0.6)
        axes[0, 0].plot([self.y_train.min(), self.y_train.max()], 
                       [self.y_train.min(), self.y_train.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Sale Price')
        axes[0, 0].set_ylabel('Predicted Sale Price')
        axes[0, 0].set_title('Training Set: Actual vs Predicted')
        
        # Actual vs Predicted - Testing
        axes[0, 1].scatter(self.y_test, self.y_pred_test, alpha=0.6)
        axes[0, 1].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 1].set_xlabel('Actual Sale Price')
        axes[0, 1].set_ylabel('Predicted Sale Price')
        axes[0, 1].set_title('Test Set: Actual vs Predicted')
        
        # Residuals - Training
        train_residuals = self.y_train - self.y_pred_train
        axes[1, 0].scatter(self.y_pred_train, train_residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Sale Price')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Training Set: Residuals')
        
        # Residuals - Testing
        test_residuals = self.y_test - self.y_pred_test
        axes[1, 1].scatter(self.y_pred_test, test_residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Sale Price')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Test Set: Residuals')
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model and preprocessing objects"""
        if self.rf_model is None:
            raise ValueError("Model not trained yet. Call train_random_forest() first.")
        
        model_data = {
            'model': self.rf_model,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and preprocessing objects"""
        model_data = joblib.load(filepath)
        self.rf_model = model_data['model']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        print(f"Model loaded from {filepath}")
    
    def predict_new_data(self, new_data):
        """Make predictions on new data"""
        if self.rf_model is None:
            raise ValueError("Model not trained yet. Call train_random_forest() first.")
        
        # Preprocess new data
        new_data_processed = new_data.copy()
        
        # Apply same preprocessing as training data
        categorical_features = [feat for feat in new_data_processed.columns if new_data_processed[feat].dtype == 'O']
        
        for feature in categorical_features:
            if feature in self.label_encoders:
                new_data_processed[feature] = self.label_encoders[feature].transform(new_data_processed[feature].astype(str))
        
        # Make predictions
        predictions = self.rf_model.predict(new_data_processed)
        
        return predictions

# Example usage
def main():
    """Example usage of the Random Forest pipeline"""
    
    # Initialize the pipeline
    rf_pipeline = RandomForestRealEstatePipeline(random_state=42)
    
    # Load and preprocess data
    df = rf_pipeline.load_and_preprocess_data('real_estate_dataset.csv')
    
    # Handle missing values and outliers
    missing_summary = rf_pipeline.handle_missing_values()
    outlier_summary = rf_pipeline.handle_outliers(method='cap')
    
    # Feature engineering
    df_engineered = rf_pipeline.feature_engineering()
    
    # Prepare features and target
    X, y = rf_pipeline.prepare_features_target(target_column='Sale_price')
    
    # Train Random Forest
    results = rf_pipeline.train_random_forest(X, y, optimize_hyperparameters=True)
    
    # Analyze feature importance
    feature_imp, perm_imp = rf_pipeline.analyze_feature_importance(top_n=20)
    
    # Cross-validate
    cv_scores = rf_pipeline.cross_validate_model(X, y, cv=5)
    
    # Visualize predictions
    rf_pipeline.visualize_predictions()
    
    # Save model
    # rf_pipeline.save_model('random_forest_real_estate_model.pkl')
    
    print("\nRandom Forest Real Estate Pipeline completed successfully!")

if __name__ == "__main__":
    main()