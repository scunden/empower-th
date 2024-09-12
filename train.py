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
    test_auc = framework.run()

if __name__ == "__main__":
    main()
    