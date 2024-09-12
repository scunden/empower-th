
import pandas as pd
import numpy as np
from scripts.preprocessing import PreProcessor
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score, recall_score, accuracy_score, f1_score, roc_curve
from scripts.utils import variables as var
import os
import shap
from scripts.utils import functions as f

sns.set_theme()


class ModelFramework():
    def __init__(self, data_process=None, hps=None, seed=42, explain=False, save=False, logger=None):
        self.logger = f.create_logger() if logger is None else logger
        self.logger.debug('Initializing modeling framework')
        self.seed = seed
        self.data_process = data_process if data_process is not None else var.DEFAULT_DATA_PROCESS
        self.hps = hps if hps is not None else var.DEFAULT_HPS 
        self.preprocessor = PreProcessor(seed=self.seed, logger=self.logger)
        self.explain=explain
        self.save = save
    
    def generate_data(self):
        
        Xs, ys = self.preprocessor.run_processes(**self.data_process)

        return Xs, ys
    
    def tune(self, X_train, y_train):
        
        self.logger.debug('Training and tuning model')
        model = xgb.XGBClassifier(objective = 'binary:logistic', random_state=self.seed)

        grid_search = GridSearchCV(
            estimator = model,
            param_grid = self.hps,
            scoring='roc_auc',
            cv=3,
            )

        grid_search.fit(X_train, y_train, verbose=0)
        self.logger.debug(f'Best gridsearch hyperparameters: {grid_search.best_params_}')
        self.logger.debug(f'Best gridsearch training AUC: {grid_search.best_score_}')
        
        return grid_search
    
    def validate(self, grid_search, X_train, y_train, X_val, y_val):
        
        self.logger.debug('Validating best model')
        best_model = xgb.XGBClassifier(
            objective = 'binary:logistic', 
            random_state=self.seed,
            max_depth = grid_search.best_params_['max_depth'],
            learning_rate = grid_search.best_params_['learning_rate'],
            n_estimators = grid_search.best_params_['n_estimators'],
            eval_metric='auc'
            
            )
        
        best_model.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_val, y_val)], verbose=0)
        train_auc = best_model.evals_result()['validation_0']['auc']
        val_auc = best_model.evals_result()['validation_1']['auc']
        
        self.logger.debug(f'Best training AUC: {np.mean(train_auc)}')
        self.logger.debug(f'Best validation AUC: {np.mean(val_auc)}')
        
        iterations = range(0, len(val_auc))
        self.generate_learning_curves(iterations, train_auc, val_auc)
        self.generate_roc_curves(best_model, X_val, y_val)
        t, precision, recall = self.generate_pr_curve(best_model, X_val, y_val)
        optimal_t = self.optimal_thresholds(t, precision, recall)
        
        return val_auc, best_model, optimal_t
    
    def generate_learning_curves(self, iterations, train_auc, val_auc):
        if self.save:
            self.logger.debug(f'Generating learning curves...')
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(iterations, train_auc, label='training')
            ax.plot(iterations, val_auc, label='validation')
            ax.set_title('Learning Curve')
            ax.set_xlabel('Iterations')
            ax.set_ylabel('AUC')
            ax.legend()
            plt.savefig('reports/images/learning-curves.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_roc_curves(self, model, X_val, y_val):
        y_val_preds_proba = model.predict_proba(X_val)[:,1]
        fpr, tpr, _ = roc_curve(y_val, y_val_preds_proba)
        auc = roc_auc_score(y_val, y_val_preds_proba)
            
        if self.save:
            
            self.logger.debug(f'Generating ROC curves...')
            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(fpr, tpr, label=f'ROC Curve {auc}')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.plot([0,1],[0,1], color='red', linestyle='--')
            fig.legend()
        
            plt.savefig('reports/images/roc-curves.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return auc
    
    def generate_pr_curve(self, model, X_val, y_val):
        y_val_preds_proba = model.predict_proba(X_val)[:,1]
        precision_val, recall_val, t_val = precision_recall_curve(y_val, y_val_preds_proba)
            
        if self.save:
            
            self.logger.debug(f'Generating Precision-Recall curves...')
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(t_val, precision_val[:-1], label='Precision')
            ax.plot(t_val, recall_val[:-1], label='Recall')
            ax.set_title('PR Curve')
            ax.set_xlabel('Thresholds')
            ax.set_ylabel('Metrics')
            ax.legend()
            plt.savefig('reports/images/pr-curves.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        return t_val, precision_val[:-1], recall_val[:-1]
    
    def optimal_thresholds(self, t, precision, recall):
        
        diff = precision - recall
        crossing_index = np.where(np.diff(np.sign(diff)))[0]
        optimal_t = np.mean(t[crossing_index])
        self.logger.debug(f'Optimal threshold for prediction: {optimal_t}')
        
        return optimal_t

    
    def test(self, model, X_test, y_test, optimal_t=0.3):
        y_test_preds_proba = model.predict_proba(X_test)[:,1]
        y_test_preds = (y_test_preds_proba>optimal_t).astype(int)

        p = precision_score(y_test, y_test_preds)
        r = recall_score(y_test, y_test_preds)
        auc = roc_auc_score(y_test, y_test_preds)
        self.logger.debug(f'Test result: AUC {auc:.2f} |  Precision {p:.2f} |  Recall {r:.2f} | ')
        
        return auc
    
    def run(self):
        
        Xs, ys = self.generate_data()
        X_test, X_train, X_val = Xs
        y_test, y_train, y_val = ys
        
        grid_search = self.tune(X_train, y_train)
        val_auc, best_model, optimal_t = self.validate(grid_search, X_train, y_train, X_val, y_val)
        test_auc = self.test(best_model, X_test, y_test, optimal_t=optimal_t)
        if self.explain:
            self.create_explanations(best_model, X_test, y_test)
        
        return test_auc
    
    def create_explanations(self, model, X_test, y_test):
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(X_test)
        
        if self.save:
            self.logger.debug(f'Generating Shap curves...')
            plt.figure()  
            shap.summary_plot(shap_values, features=X_test, feature_names=X_test.columns, plot_type="bar", show=False)
            plt.savefig('reports/images/mean-shap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
            self.shap_example(model, X_test, y_test)
        
    def shap_example(self, model, X_test, y_test):
        y_test_preds_proba = model.predict_proba(X_test)[:,1]
        res = y_test.to_frame()
        res['pred'] = y_test_preds_proba
        loc_idx = res.sort_values('pred', ascending=False).iloc[5].name
        iloc_idx = X_test.index.get_loc(loc_idx)
        
        if self.save:
            explainer = shap.Explainer(model)
            shap_values = explainer(X_test)
            plt.figure()  
            shap.plots.waterfall(shap_values[iloc_idx], show=False)
            plt.savefig('reports/images/shap-example.png', dpi=300, bbox_inches='tight')
            plt.close()


def main():
    framework = ModelFramework()
    _ = framework.run()

if __name__ == "__main__":
    main()
    