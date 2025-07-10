
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
            # Clean data first: remove inf/nan values for binning
            finishedsqft_clean = self.df['FinishedSqft'].replace([np.inf, -np.inf], np.nan)
            self.df['property_size_category'] = pd.cut(
                finishedsqft_clean, 
                bins=[0, 1000, 1500, 2500, 5000, finishedsqft_clean.max() + 1],
                labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'],
                include_lowest=True
            )
            
            # Lot size categories - similar reasoning
            # Clean data first: remove inf/nan values for binning
            lotsize_clean = self.df['Lotsize'].replace([np.inf, -np.inf], np.nan)
            self.df['lot_size_category'] = pd.cut(
                lotsize_clean,
                bins=[0, 5000, 10000, 20000, 50000, lotsize_clean.max() + 1],
                labels=['Small Lot', 'Medium Lot', 'Large Lot', 'Very Large Lot', 'Estate'],
                include_lowest=True
            )
        
        # 2. ROOM AND BATHROOM FEATURES
        # These capture living space efficiency and comfort
        if all(col in self.df.columns for col in ['Rooms', 'Bdrms', 'Fbath', 'Hbath']):
            # Total bathrooms - half baths count as 0.5
            self.df['total_bathrooms'] = self.df['Fbath'] + (0.5 * self.df['Hbath'])
            
            # Bedroom to total room ratio - indicates room allocation
            # Handle division by zero
            self.df['bedroom_ratio'] = np.where(
                self.df['Rooms'] > 0, 
                self.df['Bdrms'] / self.df['Rooms'], 
                0
            )
            
            # Bathroom to bedroom ratio - indicates convenience level
            self.df['bathroom_bedroom_ratio'] = np.where(
                self.df['Bdrms'] > 0,
                self.df['total_bathrooms'] / self.df['Bdrms'],
                0
            )
            
            # Room efficiency features - space utilization
            if 'FinishedSqft' in self.df.columns:
                self.df['room_efficiency'] = np.where(
                    self.df['FinishedSqft'] > 0,
                    self.df['Rooms'] / self.df['FinishedSqft'],
                    0
                )
                self.df['space_per_room'] = np.where(
                    self.df['Rooms'] > 0,
                    self.df['FinishedSqft'] / self.df['Rooms'],
                    0
                )
        
        # 3. PRICE-RELATED FEATURES
        # These help understand value patterns
        if 'Sale_price' in self.df.columns:
            if 'FinishedSqft' in self.df.columns:
                # Price per square foot - key real estate metric
                self.df['price_per_sqft'] = np.where(
                    self.df['FinishedSqft'] > 0,
                    self.df['Sale_price'] / self.df['FinishedSqft'],
                    0
                )
                
                # Price per square foot categories - market segments
                # Clean data first: remove inf/nan values for binning
                price_per_sqft_clean = self.df['price_per_sqft'].replace([np.inf, -np.inf], np.nan)
                if price_per_sqft_clean.notna().sum() > 0:  # Only if we have valid data
                    self.df['price_per_sqft_category'] = pd.cut(
                        price_per_sqft_clean,
                        bins=5,
                        labels=['Budget', 'Economy', 'Mid-range', 'Premium', 'Luxury']
                    )
            
            # Log transformation for price - helps with skewed distributions
            # log1p is log(1+x) which handles zero values safely
            self.df['log_sale_price'] = np.log1p(self.df['Sale_price'])
        
        # 4. NEIGHBORHOOD AND LOCATION FEATURES
        # Location is crucial in real estate - these features capture location value
        if 'nbhd' in self.df.columns and 'Sale_price' in self.df.columns:
            # Neighborhood price statistics - market positioning
            nbhd_stats = self.df.groupby('nbhd')['Sale_price'].agg(['mean', 'median', 'std']).reset_index()
            nbhd_stats.columns = ['nbhd', 'nbhd_mean_price', 'nbhd_median_price', 'nbhd_price_std']
            self.df = self.df.merge(nbhd_stats, on='nbhd', how='left')
            
            # Price deviation from neighborhood average - relative value
            self.df['price_deviation_from_nbhd'] = np.where(
                self.df['nbhd_mean_price'] > 0,
                (self.df['Sale_price'] - self.df['nbhd_mean_price']) / self.df['nbhd_mean_price'],
                0
            )
            
            # Neighborhood ranking based on median price - prestige indicator
            nbhd_ranking = self.df.groupby('nbhd')['Sale_price'].median().rank(ascending=False).reset_index()
            nbhd_ranking.columns = ['nbhd', 'nbhd_price_rank']
            self.df = self.df.merge(nbhd_ranking, on='nbhd', how='left')
        
        # 5. PROPERTY AGE AND CONDITION FEATURES
        # Age affects property value in complex ways
        if 'Year_Built' in self.df.columns:  # Fixed: should be Year_Built, not Built
            current_year = datetime.now().year
            self.df['property_age'] = current_year - self.df['Year_Built']
            
            # Handle negative ages (future dates) and inf values
            self.df['property_age'] = np.where(
                self.df['property_age'] < 0,
                0,
                self.df['property_age']
            )
            
            # Age categories - different age ranges have different market appeal
            # Clean data first: remove inf/nan values for binning
            property_age_clean = self.df['property_age'].replace([np.inf, -np.inf], np.nan)
            if property_age_clean.notna().sum() > 0:  # Only if we have valid data
                max_age = property_age_clean.max()
                self.df['age_category'] = pd.cut(
                    property_age_clean,
                    bins=[0, 5, 15, 30, 50, max_age + 1],
                    labels=['New', 'Modern', 'Established', 'Mature', 'Historic'],
                    include_lowest=True
                )
            
            # Depreciation factor - exponential decay model
            # Assumes property value decreases exponentially with age
            self.df['depreciation_factor'] = np.exp(-self.df['property_age'] / 50)
        
        # 6. INTERACTION FEATURES
        # These capture complex relationships between features
        if all(col in self.df.columns for col in ['FinishedSqft', 'Lotsize', 'total_bathrooms', 'Rooms']):
            # Property luxury score - composite feature
            # Weighted combination of size, amenities, and space
            # Use safe median calculation to avoid division by zero
            finishedsqft_median = self.df['FinishedSqft'].median()
            bathrooms_median = self.df['total_bathrooms'].median()
            rooms_median = self.df['Rooms'].median()
            lotsize_median = self.df['Lotsize'].median()
            
            # Only calculate if medians are valid
            if all(x > 0 for x in [finishedsqft_median, bathrooms_median, rooms_median, lotsize_median]):
                self.df['luxury_score'] = (
                    (self.df['FinishedSqft'] / finishedsqft_median) * 0.3 +
                    (self.df['total_bathrooms'] / bathrooms_median) * 0.2 +
                    (self.df['Rooms'] / rooms_median) * 0.2 +
                    (self.df['Lotsize'] / lotsize_median) * 0.3
                )
                # Handle any remaining inf values
                self.df['luxury_score'] = self.df['luxury_score'].replace([np.inf, -np.inf], 0)
        
        # 7. STATISTICAL TRANSFORMATIONS
        # Handle skewed distributions for better model performance
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Sale_price', 'District', 'nbhd'] and col in original_features:
                # Skip if column has inf values that can't be handled
                if np.isinf(self.df[col]).any():
                    continue
                    
                # Check if feature is highly skewed
                col_data = self.df[col].dropna()
                if len(col_data) == 0:
                    continue
                    
                col_skew = skew(col_data)
                if abs(col_skew) > 1.5:  # Rule of thumb: |skewness| > 1.5 is highly skewed
                    # Box-Cox transformation for positive values
                    if (col_data > 0).all():
                        try:
                            self.df[f'{col}_boxcox'], _ = boxcox(col_data + 1)
                            self.engineered_features.append(f'{col}_boxcox')
                        except:
                            # Fall back to log transformation if Box-Cox fails
                            self.df[f'{col}_log'] = np.log1p(col_data)
                            self.engineered_features.append(f'{col}_log')
                    else:
                        # Log transformation for features with negative values
                        min_val = col_data.min()
                        self.df[f'{col}_log'] = np.log1p(col_data - min_val + 1)
                        self.engineered_features.append(f'{col}_log')
        
        # 8. CLUSTERING FEATURES
        # Discover hidden patterns in the data
        if hasattr(self, '_create_clustering_features'):
            self._create_clustering_features()
        
        # 9. POLYNOMIAL FEATURES (selective)
        # Capture non-linear relationships
        if hasattr(self, '_create_polynomial_features'):
            self._create_polynomial_features()
        
        # Track engineered features
        new_features = [col for col in self.df.columns if col not in original_features]
        if not hasattr(self, 'engineered_features'):
            self.engineered_features = []
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
        """
        Validate data quality after preprocessing
        
        Final quality checks to ensure data is ready for modeling:
        - Check for remaining missing values
        - Validate data types
        - Check for infinite values
        - Verify target variable distribution
        - Feature correlation analysis (only for numeric columns)
        """
        print("Validating data quality...")
        
        # 1. CHECK FOR REMAINING MISSING VALUES
        missing_values = self.df.isnull().sum()
        if missing_values.sum() > 0:
            print("Warning: Remaining missing values found:")
            print(missing_values[missing_values > 0])
            
            # Handle remaining missing values
            for col in missing_values[missing_values > 0].index:
                if self.df[col].dtype == 'object' or self.df[col].dtype.name == 'category':
                  # Convert categorical to object to avoid category restrictions
                  if self.df[col].dtype.name == 'category':
                    self.df[col] = self.df[col].astype('object')
                    
                  # For categorical/object columns, use mode or 'Unknown'
                  if self.df[col].mode().empty:
                    self.df[col] = self.df[col].fillna('Unknown')
                    print(f"  Filled {col} with 'Unknown'")
                  else:
                    mode_val = self.df[col].mode()[0]
                    self.df[col] = self.df[col].fillna(mode_val)
                    print(f"  Filled {col} with mode: {mode_val}")
                else:
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                    print(f"  Filled {col} with median: {median_val}")
        else:
            print("✓ No missing values found")
        
        # 2. CHECK FOR INFINITE VALUES
        print("Alert: checking for infinite values:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_values = {}
        for col in numeric_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                inf_values[col] = inf_count
                # Replace infinite values with column median (more robust)
                finite_values = self.df[col].replace([np.inf, -np.inf], np.nan)
                median_val = finite_values.median()
                self.df[col] = self.df[col].replace([np.inf, -np.inf], median_val)
        
        if inf_values:
            print(f"Warning: Found and replaced infinite values in: {list(inf_values.keys())}")
        else:
            print("✓ No infinite values found")
        
        # 3. VALIDATE TARGET VARIABLE
        if 'Sale_price' in self.df.columns:
            target_stats = {
                'count': self.df['Sale_price'].count(),
                'mean': self.df['Sale_price'].mean(),
                'std': self.df['Sale_price'].std(),
                'min': self.df['Sale_price'].min(),
                'max': self.df['Sale_price'].max(),
                'skewness': skew(self.df['Sale_price']),
                'zero_values': (self.df['Sale_price'] == 0).sum(),
                'negative_values': (self.df['Sale_price'] < 0).sum()
            }
            
            print(f"Target variable (Sale_price) statistics:")
            for key, value in target_stats.items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
            
            # Check for problematic target values
            if target_stats['zero_values'] > 0:
                print(f"Warning: {target_stats['zero_values']} zero values in target variable")
            if target_stats['negative_values'] > 0:
                print(f"Warning: {target_stats['negative_values']} negative values in target variable")
        
        # 4. CHECK FEATURE CORRELATION WITH TARGET (ONLY NUMERIC COLUMNS)
        if 'Sale_price' in self.df.columns:
            # Only use numeric columns for correlation analysis
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                numeric_df = self.df[numeric_cols]
                correlations = numeric_df.corr()['Sale_price'].abs().sort_values(ascending=False)
                print(f"\nTop 10 numeric features correlated with Sale_price:")
                print(correlations.head(10))
                
                # Identify features with very low correlation
                low_corr_features = correlations[correlations < 0.01].index.tolist()
                if 'Sale_price' in low_corr_features:
                    low_corr_features.remove('Sale_price')
                if low_corr_features:
                    print(f"\nNumeric features with very low correlation (<0.01): {len(low_corr_features)}")
            else:
                print("Not enough numeric columns for correlation analysis")
        
        # 5. CHECK FOR MULTICOLLINEARITY (ONLY NUMERIC FEATURES)
        feature_numeric_cols = [col for col in numeric_cols if col != 'Sale_price']
        
        if len(feature_numeric_cols) > 1:
            correlation_matrix = self.df[feature_numeric_cols].corr()
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.95:
                        high_corr_pairs.append((
                            correlation_matrix.columns[i],
                            correlation_matrix.columns[j],
                            correlation_matrix.iloc[i, j]
                        ))
            
            if high_corr_pairs:
                print(f"\nWarning: Found {len(high_corr_pairs)} highly correlated feature pairs (>0.95):")
                for pair in high_corr_pairs[:5]:  # Show first 5
                    print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")
            else:
                print("✓ No severe multicollinearity detected")
        else:
            print("Not enough numeric features for multicollinearity analysis")
        
        # 6. DATA TYPE SUMMARY
        print(f"\nData type summary:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")
        
        # 7. CATEGORICAL VARIABLE SUMMARY
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"\nCategorical variables found: {len(categorical_cols)}")
            for col in categorical_cols:
                unique_count = self.df[col].nunique()
                print(f"  {col}: {unique_count} unique values")
        else:
            print("✓ No categorical variables found")
        
        # 8. FINAL DATA SHAPE AND MEMORY USAGE
        print(f"\nFinal dataset shape: {self.df.shape}")
        print(f"Memory usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return True
    
    def prepare_features_and_target(self, target_column='Sale_price'):
        """
        Prepare features and target variable for modeling
        
        This method separates features from target variable and handles
        categorical encoding for machine learning algorithms.
        
        Args:
            target_column (str): Name of the target variable column
            
        Returns:
            tuple: (X_features, y_target) ready for modeling
        """
        print(f"\nPreparing features and target variable...")
        
        # Separate target variable
        if target_column not in self.df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        y = self.df[target_column].copy()
        X = self.df.drop([target_column], axis=1)
        
        # Handle categorical variables
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        print(f"Encoding {len(categorical_columns)} categorical features...")
        
        for col in categorical_columns:
            # Use LabelEncoder for categorical variables
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
            print(f"  Encoded {col}: {len(le.classes_)} categories")
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        print(f"Final feature set: {len(self.feature_names)} features")
        print(f"Target variable: {len(y)} samples")
        
        return X, y
    
    def train_random_forest_model(self, X, y, test_size=0.2, optimize_hyperparameters=True):
        """
        Train Random Forest model with comprehensive evaluation
        
        This method implements the complete machine learning workflow:
        - Train/validation/test split
        - Feature scaling
        - Hyperparameter optimization
        - Model training and evaluation
        - Feature importance analysis
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            optimize_hyperparameters (bool): Whether to perform hyperparameter tuning
            
        Returns:
            dict: Comprehensive model performance metrics
        """
        print(f"\nTraining Random Forest model...")
        
        # 1. TRAIN-TEST SPLIT
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=None
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # 2. FEATURE SCALING
        # Scale features to improve model performance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Convert back to DataFrame for easier handling
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
        
        # 3. HYPERPARAMETER OPTIMIZATION
        if optimize_hyperparameters:
            print("Optimizing hyperparameters...")
            best_params = self._optimize_hyperparameters(X_train_scaled, y_train)
            self.rf_model = RandomForestRegressor(random_state=self.random_state, **best_params)
        else:
            # Use default parameters with some sensible choices
            self.rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        # 4. MODEL TRAINING
        print("Training model...")
        self.rf_model.fit(X_train_scaled, y_train)
        
        # 5. MODEL EVALUATION
        evaluation_results = self._comprehensive_model_evaluation(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
        
        # 6. FEATURE IMPORTANCE ANALYSIS
        self._analyze_feature_importance(X_train_scaled)
        
        # 7. CROSS-VALIDATION
        cv_scores = cross_val_score(
            self.rf_model, X_train_scaled, y_train, 
            cv=5, scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        evaluation_results['cv_rmse_mean'] = np.sqrt(-cv_scores.mean())
        evaluation_results['cv_rmse_std'] = np.sqrt(cv_scores.std())
        
        print(f"Cross-validation RMSE: {evaluation_results['cv_rmse_mean']:.2f} ± {evaluation_results['cv_rmse_std']:.2f}")
        
        # Store data for later use
        self.X_train, self.X_test = X_train_scaled, X_test_scaled
        self.y_train, self.y_test = y_train, y_test
        
        return evaluation_results
    
    def _optimize_hyperparameters(self, X_train, y_train):
        """
        Optimize Random Forest hyperparameters using GridSearchCV
        
        This method performs systematic hyperparameter optimization to find
        the best combination of parameters for the Random Forest model.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            
        Returns:
            dict: Best hyperparameters found
        """
        # Define parameter grid for optimization
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Use a subset of data for faster hyperparameter optimization
        if len(X_train) > 5000:
            X_sample = X_train.sample(n=5000, random_state=self.random_state)
            y_sample = y_train.loc[X_sample.index]
        else:
            X_sample, y_sample = X_train, y_train
        
        # Perform grid search
        rf_base = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_sample, y_sample)
        
        print(f"Best parameters found: {grid_search.best_params_}")
        print(f"Best cross-validation score: {np.sqrt(-grid_search.best_score_):.2f}")
        
        return grid_search.best_params_
    
    def _comprehensive_model_evaluation(self, X_train, X_test, y_train, y_test):
        """
        Comprehensive model evaluation with multiple metrics
        
        This method evaluates the model performance using various metrics
        and provides detailed analysis of prediction quality.
        
        Args:
            X_train, X_test (pd.DataFrame): Training and test features
            y_train, y_test (pd.Series): Training and test targets
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        print("\nEvaluating model performance...")
        
        # Make predictions
        y_train_pred = self.rf_model.predict(X_train)
        y_test_pred = self.rf_model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'train_mape': mean_absolute_percentage_error(y_train, y_train_pred),
            'test_mape': mean_absolute_percentage_error(y_test, y_test_pred)
        }
        
        # Print evaluation results
        print("\nModel Performance Metrics:")
        print("-" * 40)
        print(f"Training RMSE: {metrics['train_rmse']:,.2f}")
        print(f"Test RMSE: {metrics['test_rmse']:,.2f}")
        print(f"Training R²: {metrics['train_r2']:.4f}")
        print(f"Test R²: {metrics['test_r2']:.4f}")
        print(f"Training MAE: {metrics['train_mae']:,.2f}")
        print(f"Test MAE: {metrics['test_mae']:,.2f}")
        print(f"Training MAPE: {metrics['train_mape']:.2f}%")
        print(f"Test MAPE: {metrics['test_mape']:.2f}%")
        
        # Check for overfitting
        rmse_diff = metrics['test_rmse'] - metrics['train_rmse']
        r2_diff = metrics['train_r2'] - metrics['test_r2']
        
        print(f"\nOverfitting Analysis:")
        print(f"RMSE difference (test - train): {rmse_diff:,.2f}")
        print(f"R² difference (train - test): {r2_diff:.4f}")
        
        if rmse_diff > 0.2 * metrics['train_rmse']:
            print("Warning: Possible overfitting detected (high RMSE difference)")
        if r2_diff > 0.1:
            print("Warning: Possible overfitting detected (high R² difference)")
        
        # Create prediction plots
        self._create_prediction_plots(y_test, y_test_pred)
        
        return metrics
    
    def _create_prediction_plots(self, y_true, y_pred):
        """
        Create comprehensive prediction analysis plots
        
        Visualizations help understand model performance and identify
        areas where the model performs well or poorly.
        
        Args:
            y_true (pd.Series): True values
            y_pred (np.array): Predicted values
        """
        print("\nCreating prediction analysis plots...")
        
        # 1. Actual vs Predicted scatter plot
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.grid(True, alpha=0.3)
        
        # 2. Residuals plot
        plt.subplot(1, 3, 2)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        
        # 3. Residuals histogram
        plt.subplot(1, 3, 3)
        plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 4. Prediction error distribution by price range
        plt.figure(figsize=(12, 6))
        
        # Create price ranges
        price_ranges = pd.cut(y_true, bins=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
        error_by_range = pd.DataFrame({
            'price_range': price_ranges,
            'absolute_error': np.abs(residuals)
        })
        
        plt.subplot(1, 2, 1)
        sns.boxplot(data=error_by_range, x='price_range', y='absolute_error')
        plt.title('Prediction Error by Price Range')
        plt.ylabel('Absolute Error')
        plt.xticks(rotation=45)
        
        # 5. Percentage error by price range
        plt.subplot(1, 2, 2)
        percentage_error = (np.abs(residuals) / y_true) * 100
        error_by_range['percentage_error'] = percentage_error
        sns.boxplot(data=error_by_range, x='price_range', y='percentage_error')
        plt.title('Percentage Error by Price Range')
        plt.ylabel('Percentage Error (%)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _analyze_feature_importance(self, X_train):
        """
        Analyze and visualize feature importance
        
        Feature importance analysis helps understand which features
        contribute most to the model's predictions.
        
        Args:
            X_train (pd.DataFrame): Training features
        """
        print("\nAnalyzing feature importance...")
        
        # Get feature importance from the model
        feature_importance = self.rf_model.feature_importances_
        
        # Create feature importance DataFrame
        self.feature_importance_df = pd.DataFrame({
            'feature': X_train.columns,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        # Print top 20 features
        print("Top 20 most important features:")
        print(self.feature_importance_df.head(20))
        
        # Create visualizations
        plt.figure(figsize=(15, 10))
        
        # 1. Top 20 features bar plot
        plt.subplot(2, 2, 1)
        top_20 = self.feature_importance_df.head(20)
        sns.barplot(data=top_20, x='importance', y='feature')
        plt.title('Top 20 Feature Importance')
        plt.xlabel('Importance')
        
        # 2. Feature importance distribution
        plt.subplot(2, 2, 2)
        plt.hist(feature_importance, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Feature Importance')
        plt.ylabel('Frequency')
        plt.title('Feature Importance Distribution')
        
        # 3. Cumulative importance
        plt.subplot(2, 2, 3)
        cumulative_importance = np.cumsum(self.feature_importance_df['importance'])
        plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance)
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Importance')
        plt.title('Cumulative Feature Importance')
        plt.grid(True, alpha=0.3)
        
        # 4. Feature importance by category
        plt.subplot(2, 2, 4)
        # Categorize features
        categories = []
        for feature in self.feature_importance_df['feature']:
            if any(keyword in feature.lower() for keyword in ['price', 'cost', 'value']):
                categories.append('Price-related')
            elif any(keyword in feature.lower() for keyword in ['sqft', 'size', 'area']):
                categories.append('Size-related')
            elif any(keyword in feature.lower() for keyword in ['room', 'bed', 'bath']):
                categories.append('Room-related')
            elif any(keyword in feature.lower() for keyword in ['nbhd', 'neighborhood', 'location']):
                categories.append('Location-related')
            elif any(keyword in feature.lower() for keyword in ['age', 'built', 'year']):
                categories.append('Age-related')
            else:
                categories.append('Other')
        
        self.feature_importance_df['category'] = categories
        category_importance = self.feature_importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
        
        plt.pie(category_importance.values, labels=category_importance.index, autopct='%1.1f%%')
        plt.title('Feature Importance by Category')
        
        plt.tight_layout()
        plt.show()
        
        # Permutation importance for more robust analysis
        print("\nCalculating permutation importance...")
        try:
            perm_importance = permutation_importance(
                self.rf_model, X_train, self.y_train,
                n_repeats=5, random_state=self.random_state, n_jobs=-1
            )
            
            # Create permutation importance DataFrame
            perm_importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance_mean': perm_importance.importances_mean,
                'importance_std': perm_importance.importances_std
            }).sort_values('importance_mean', ascending=False)
            
            print("Top 10 features by permutation importance:")
            print(perm_importance_df.head(10))
            
        except Exception as e:
            print(f"Could not calculate permutation importance: {e}")
    
    def save_model(self, filepath):
        """
        Save the trained model and preprocessing components
        
        This method saves all necessary components for model deployment:
        - Trained Random Forest model
        - Feature scaler
        - Label encoders
        - Feature names
        
        Args:
            filepath (str): Path to save the model (without extension)
        """
        if self.rf_model is None:
            raise ValueError("No model trained yet. Please train a model first.")
        
        print(f"\nSaving model to {filepath}...")
        
        # Create model package
        model_package = {
            'model': self.rf_model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_df,
            'engineered_features': self.engineered_features,
            'model_metadata': {
                'training_date': datetime.now().isoformat(),
                'model_type': 'RandomForestRegressor',
                'n_features': len(self.feature_names),
                'random_state': self.random_state
            }
        }
        
        # Save model package
        joblib.dump(model_package, f"{filepath}.pkl")
        
        print(f"Model saved successfully to {filepath}.pkl")
        
        # Save feature importance as CSV
        if self.feature_importance_df is not None:
            self.feature_importance_df.to_csv(f"{filepath}_feature_importance.csv", index=False)
            print(f"Feature importance saved to {filepath}_feature_importance.csv")
    
    def load_model(self, filepath):
        """
        Load a previously saved model
        
        Args:
            filepath (str): Path to the saved model file
        """
        print(f"\nLoading model from {filepath}...")
        
        # Load model package
        model_package = joblib.load(filepath)
        
        # Restore components
        self.rf_model = model_package['model']
        self.scaler = model_package['scaler']
        self.label_encoders = model_package['label_encoders']
        self.feature_names = model_package['feature_names']
        self.feature_importance_df = model_package.get('feature_importance')
        self.engineered_features = model_package.get('engineered_features', [])
        
        print("Model loaded successfully!")
        print(f"Model type: {model_package['model_metadata']['model_type']}")
        print(f"Number of features: {model_package['model_metadata']['n_features']}")
        print(f"Training date: {model_package['model_metadata']['training_date']}")
    
    def predict(self, X_new):
        """
        Make predictions on new data
        
        Args:
            X_new (pd.DataFrame): New data for prediction
            
        Returns:
            np.array: Predicted values
        """
        if self.rf_model is None:
            raise ValueError("No model trained yet. Please train a model first.")
        
        print(f"\nMaking predictions for {len(X_new)} samples...")
        
        # Ensure feature consistency
        X_processed = X_new.copy()
        
        # Handle categorical variables
        for col, encoder in self.label_encoders.items():
            if col in X_processed.columns:
                X_processed[col] = encoder.transform(X_processed[col].astype(str))
        
        # Ensure all required features are present
        missing_features = set(self.feature_names) - set(X_processed.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features
        X_processed = X_processed[self.feature_names]
        
        # Scale features
        X_scaled = self.scaler.transform(X_processed)
        
        # Make predictions
        predictions = self.rf_model.predict(X_scaled)
        
        return predictions
    
    def run_complete_pipeline(self, filepath, target_column='Sale_price'):
        """
        Run the complete machine learning pipeline
        
        This method orchestrates the entire pipeline from data loading
        to model training and evaluation.
        
        Args:
            filepath (str): Path to the dataset
            target_column (str): Name of the target variable
            
        Returns:
            dict: Complete pipeline results
        """
        print("="*60)
        print("RUNNING COMPLETE RANDOM FOREST PIPELINE")
        print("="*60)
        
        pipeline_results = {}
        
        try:
            # 1. Load and analyze data
            start_time = datetime.now()
            self.load_and_preprocess_data(filepath)
            pipeline_results['data_loading'] = 'Success'
            
            # 2. Feature engineering
            self.advanced_feature_engineering()
            pipeline_results['feature_engineering'] = 'Success'
            
            # 3. Data preprocessing
            self.advanced_data_preprocessing()
            pipeline_results['preprocessing'] = 'Success'
            
            # 4. Prepare features and target
            X, y = self.prepare_features_and_target(target_column)
            pipeline_results['feature_preparation'] = 'Success'
            
            # 5. Train model
            evaluation_results = self.train_random_forest_model(X, y)
            pipeline_results['model_training'] = 'Success'
            pipeline_results['evaluation_metrics'] = evaluation_results
            
            # 6. Pipeline completion
            end_time = datetime.now()
            pipeline_results['total_duration'] = str(end_time - start_time)
            pipeline_results['completion_status'] = 'Success'
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60)
            print(f"Total duration: {pipeline_results['total_duration']}")
            print(f"Final dataset shape: {self.df.shape}")
            print(f"Number of features: {len(self.feature_names)}")
            print(f"Test R² score: {evaluation_results['test_r2']:.4f}")
            print(f"Test RMSE: {evaluation_results['test_rmse']:,.2f}")
            
            return pipeline_results
            
        except Exception as e:
            pipeline_results['error'] = str(e)
            pipeline_results['completion_status'] = 'Failed'
            print(f"\nPipeline failed with error: {e}")
            return pipeline_results

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RandomForestRealEstatePipeline(random_state=42)
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline('real_estate_dataset.csv')
    
    # Save the trained model
    #pipeline.save_model('trained_rf_model')
    
    # Making predictions on new data
    # new_data = pd.read_csv('new_properties.csv')
    # predictions = pipeline.predict(new_data)