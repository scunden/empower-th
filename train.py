from scripts.framework import ModelFramework

def main():
    
    # Initialize the data processing pipeline
    data_process={
    'drop':True, 
    'impute':True, 
    'resample':False, 
    'scale':True, 
    'reduce':False
    }
    
    # Initialize the model framework
    framework = ModelFramework(
        data_process = data_process,
        explain=True,
        save=True
    )
    
    # Run the framework
    test_auc = framework.run()

if __name__ == "__main__":
    main()
    