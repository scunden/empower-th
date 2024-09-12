from scripts.framework import ModelFramework
import numpy as np

def main():
    
    all_processes = {
    'drop':True, 
    'impute':True, 
    'resample':True, 
    'scale':True, 
    'reduce':True
    }
    
    approaches={
        "all":all_processes,
        "all_but_resample":{k: (v if k not in ['resample'] else False) for k, v in all_processes.items()},
        "all_but_scale":{k: (v if k not in ['scale'] else False) for k, v in all_processes.items()},
        "all_but_reduce":{k: (v if k not in ['reduce'] else False) for k, v in all_processes.items()},
        "no_processes":{k: (v if k not in ['resample', 'scale', 'reduce'] else False) for k, v in all_processes.items()}
    }
    
    N = 10
    auc_by_approach={}
    for approach, processes in approaches.items():
        auc=[]
        for _ in range(N):
            framework = ModelFramework(data_process = processes, explain=False, save=False)
            Xs, ys, model, optimal_t, val_auc, test_auc = framework.run()
            auc.append(test_auc)
        auc_by_approach[approach] = np.mean(auc)
    
    print(auc_by_approach)

if __name__ == "__main__":
    main()
    
