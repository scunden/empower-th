import pandas as pd
import numpy as np
import random
from scripts.utils import variables as var
from scripts.utils import functions as f
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, StandardScaler
from sklearn.decomposition import PCA

class PreProcessor():
    def __init__(self, seed=42) -> None:
        """Initilization of class instance."""
        
        self.df_raw = pd.read_excel(var.DATA_PATH)
        self.default_drop = list(set(var.REDUNDANT_COLS+var.DROP_COLS+var.SINGLE_VALUE_COLS))
        self.default_impute = var.IMPUTE_NULLS_COLS
        self.categorical = var.CAT_COLS
        self.seed=seed
    
    def split(self, df, features):
        """Create train, test and val splits for modelling."""
        
        X, y = df[features], df[var.TARGET_VARIABLE]
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=self.seed)
        X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.2, random_state=self.seed)
        
        print(f"Train: {X_train.shape[0]} | {y_train.mean()}")
        print(f"Val: {X_val.shape[0]} | {y_val.mean()}")
        print(f"Test: {X_test.shape[0]} | {y_test.mean()}")
        
        return X_test, X_train, X_val, y_test, y_train, y_val
    
    def drop_variables(self, df, variables=None):
        """Drop redundant variables."""
        
        variables = variables if variables is not None else self.default_drop
        return df.drop(columns=variables)
    
    def resample(self, df):
        """Resample dataset to provide balanced incidence rate."""
        
        positive_samples = df.loc[df[var.TARGET_VARIABLE]==1]
        positive_sample_size = positive_samples.shape[0]
        negative_samples_frac = positive_sample_size/df.loc[df[var.TARGET_VARIABLE]==0].shape[0]
        negative_samples = df.loc[df[var.TARGET_VARIABLE]==0].sample(frac=negative_samples_frac, random_state = self.seed)

        df_resampled = pd.concat([positive_samples, negative_samples]).sample(frac=1)
        print(f"{df_resampled.shape[0]} | {df_resampled[var.TARGET_VARIABLE].mean()}")
        
        return df_resampled
    
    def impute(self, df, impute=None):
        """Impute mean value for select variables."""
        
        impute = impute if impute is not None else self.default_impute
        for col in impute:
            if col in var.FLOAT_COLS:
                df[col] = df[col].fillna((df[col].mean()))
            elif col in var.INT_COLS or col in var.BINARY_COLS:
                df[col] = df[col].fillna((df[col].mode()))
            else:
                df[col] = df[col].fillna("N/A")
                
        return df
    
    def encode_features(self, Xs, y_train):
        """Encode the categorical variables."""
        
        X_test, X_train, X_val = Xs
        self.te = TargetEncoder()
        
        X_train[self.categorical] = self.te.fit_transform(X_train[self.categorical], y_train)
        X_test[self.categorical] = self.te.transform(X_test[self.categorical])
        X_val[self.categorical] = self.te.transform(X_val[self.categorical])
        
        return X_test, X_train, X_val
    
    def scale_features(self, Xs):
        """Scale the features in the dataframe."""
        
        X_test, X_train, X_val = Xs
        self.scaler = StandardScaler()

        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), index = X_train.index, columns = X_train.columns)
        X_test = pd.DataFrame(self.scaler.transform(X_test), index = X_test.index, columns = X_test.columns)
        X_val = pd.DataFrame(self.scaler.transform(X_val), index = X_val.index, columns = X_val.columns)
        
        return X_test, X_train, X_val
    
    def reduce_features(self, Xs, n_components=0.8):
        """Apply a PCA feature reduction to reduce cardinality."""
        
        self.pca = PCA(n_components=n_components)
        X_test, X_train, X_val = Xs

        X_train= pd.DataFrame(self.pca.fit_transform(X_train), index = X_train.index)
        X_test = pd.DataFrame(self.pca.transform(X_test), index = X_test.index)
        X_val = pd.DataFrame(self.pca.transform(X_val), index = X_val.index)
        
        return X_test, X_train, X_val
    
    def run_processes(self, drop=True, impute=True, resample=True, scale=True, reduce=True):
        """Initilizate processes pipeline."""
        
        df = self.df_raw.copy()
        if drop:
            df = self.drop_variables(df)
        if impute:
            df = self.impute(df)
        if resample:
            df = self.resample(df)
        
        features = [x for x in df.columns if x != var.TARGET_VARIABLE]
        X_test, X_train, X_val, y_test, y_train, y_val = self.split(df, features)
        
        Xs = X_test, X_train, X_val
        ys = y_test, y_train, y_val
        
        Xs = self.encode_features(Xs, y_train)
        if scale:
            Xs = self.scale_features(Xs)
        if reduce:
            Xs = self.reduce_features(Xs)
            
        return Xs, ys