from scripts.framework import ModelFramework

def main():
    
    data_process={
    'drop':True, 
    'impute':True, 
    'resample':True, 
    'scale':True, 
    'reduce':False
    }
    framework = ModelFramework(
        data_process = data_process,
        explain=True,
        save=True
    )
    Xs, ys, model, optimal_t, val_auc, test_auc = framework.run()
    print(f"Test AUC {test_auc:.2f} | Optimal Threshold {optimal_t:.2f}")

if __name__ == "__main__":
    main()
    