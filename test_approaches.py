from scripts.framework import ModelFramework
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.utils import functions as f

sns.set_theme()

def plot_results(results, logger):
    
    logger.info(f'Generating bar plot...')
    categories = list(results.keys())
    values = list(results.values())
    
    fig, ax = plt.subplots()

    bars = ax.bar(categories, values, color=plt.cm.Paired.colors[:len(categories)])

    ax.set_xticklabels(categories, rotation=45, ha='right')

    ax.set_ylabel('AUC')
    ax.set_xlabel('Processes')
    ax.set_title('Average AUC by Pre Processing Pipeline')

    plt.tight_layout() 
    plt.savefig('reports/images/approach-test.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    
    # Initialize logger
    logger = f.create_logger(level='info')
    
    # Define processes
    all_processes = {
    'drop':True, 
    'impute':True, 
    'resample':True, 
    'scale':True, 
    'reduce':True
    }
    
    # Define the specific approaches
    approaches={
        "all":all_processes,
        "all_but_resample":{k: (v if k not in ['resample'] else False) for k, v in all_processes.items()},
        "all_but_scale":{k: (v if k not in ['scale'] else False) for k, v in all_processes.items()},
        "all_but_reduce":{k: (v if k not in ['reduce'] else False) for k, v in all_processes.items()},
        "no_processes":{k: (v if k not in ['resample', 'scale', 'reduce'] else False) for k, v in all_processes.items()}
    }
    
    # For every approach, run the model framework N times, and average the AUC
    N = 5
    auc_by_approach={}
    for approach, processes in approaches.items():
        logger.info(f'Starting modeling framework for: {approach}')
        auc=[]
        for _ in tqdm(range(N)):
            framework = ModelFramework(data_process = processes, explain=False, save=False, logger=logger)
            test_auc = framework.run()
            auc.append(test_auc)
        auc_by_approach[approach] = np.mean(auc)
    
    # Plot the AUC by approach
    plot_results(auc_by_approach, logger)

if __name__ == "__main__":
    main()
    
