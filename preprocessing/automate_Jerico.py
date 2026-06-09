import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def run_preprocessing(input_path, output_dir):
    print("Memulai proses preprocessing otomatis...")
    
    # 1. Load Data
    df = pd.read_csv(input_path)
    
    # 2. Cleaning Outlier (Sesuai notebook)
    df_clean = df[(df['person_age'] <= 100) & (df['person_age'].notnull())]
    df_clean = df_clean[(df_clean['person_emp_length'].fillna(0) <= 60)]
    
    # 3. Split Data
    X = df_clean.drop('loan_status', axis=1)
    y = df_clean['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Pipeline
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # sparse_output=False penting agar mudah diubah kembali ke DataFrame
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
    ])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])
    
    # 5. Transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Mendapatkan nama kolom baru setelah OneHotEncoding
    cat_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(cat_cols).tolist()
    feature_names = num_cols.tolist() + cat_feature_names
    
    # 6. Save Data Siap Latih
    os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(X_train_processed, columns=feature_names).to_csv(os.path.join(output_dir, 'X_train_clean.csv'), index=False)
    pd.DataFrame(X_test_processed, columns=feature_names).to_csv(os.path.join(output_dir, 'X_test_clean.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train_clean.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test_clean.csv'), index=False)
    
    print(f" Preprocessing selesai! Data siap latih disimpan di folder: {output_dir}")

if __name__ == "__main__":
    run_preprocessing(
        input_path="../credit_risk_dataset.csv", 
        output_dir="dataset_preprocessing"
    )