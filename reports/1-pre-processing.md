# Summary

This phase was simply about creating a process to act on the main findings from the initial exploration. The framework I built allowed me to toggle each process on or off seamlessly and programmatically. This makes testing the impact of each process on the final model very easy. This test has been done in phase 3.

These processes are:

* Data Cleaning
* Imputing
* Resampling
* Encoding
* Scaling
* Feature Reduction
* Splitting

In the future, more processes can be added to this class, allowing for reproduceable and flexible data pre processing.

# Detailed Processes

## Data Cleaning

During the exploration stage, we have identified redundant variables to drop for the analysis. Additionally, some columns needed to be re categorized. This part of the process is fairly straightforward where it allows the user to drop and properly categorize certain columns prior to the analysis

## Imputing

This process will impute values for features that have nulls. For categorical features, this imputation will simply be creating a new category called "N/A." For an integer variable, it will impute the mode. For a float variable it will impute the median.

## Resampling

Resampling involves rebalancing the dataset, so that the incidence rate of our target variable becomes closer to 50%. In our case, the incidence rate is roughly 22%, which is low, but not enough that resampling becomes a must.

The framework that I built allows me to toggle on or off the resampling process.

## Encoding and Scaling

Encoding allows categorical to be converted to a numeric format that is more digestible for modeling while retaining information about the original categories, through the encoding type. Scaling refers to a transformation process applied to numeric variables, that resets the scale of the variable to a more standard one.

The framework I built allows me to toggle each process on or off easily, while also choosing what kind of scaling/encoding I want to apply to my variables.

## Feature Reduction

Feature reduction refers to the process of minimizing the number of features by condensing them into a smaller subset of feature, that carries similar information. This is especially useful in reducing the cardinality of the dataset.

The framework I built allows me to toggle on or off feature reduction seamlessly. Note that the current feature reduction process uses a PCA approach which will retain enough components to keep 80% of variance.

## Splitting

This process is about splitting the dataset into a train set, a validation set and a test set. This process is not toggleable as it is required for the modelling piece of the broader analysis.