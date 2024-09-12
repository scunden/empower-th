"""
This file will contain all the static variables used throughout the analysis.
Most of these determinations come from the initial exploration. 
Some of them can be programatically defined, but in order to get reproduceable results
I have decided to hard code some of the categorization based on the analysis that I made

In a productionized envionrment, I would give more thought to a programmatic, or more flexible approach
"""

DATA_PATH = 'data\Project_Interview_Advance_Data.xlsx'

TARGET_VARIABLE='DefaultedAdvances'

SINGLE_VALUE_COLS = ['AdvanceAmount', 'SuspiciousTransactionCount']
DROP_COLS = ['NumberOfMatches', 'ErrorRate', 'IsNameBased']
REDUNDANT_COLS=['CreditAccountCount','LatefeesCount']
IMPUTE_NULLS_COLS=['CurrentBalance']

BINARY_COLS=['IsNameBased', 'HasEmpowerBanking']
INT_COLS=[
    'AdvanceAmount',
    'LatefeesTotalCount',
    'CreditAccounts',
    'OverdraftCount',
    'OverdraftTotal',
    'LatefeesCount',
    'CheckingAccountCount',
    'CreditAccountCount',
    'SavingsAccountCount',
    'SuspiciousTransactionCount',
    'NegativeBalanceCount',
    'Bal4100',
    'Bal3100',
    'Bal2100',
    'Bal450',
    'Bal350',
    'Bal250',
    'BalanceAbove100L30Count',
    'NumberOfMatches'
    ]

FLOAT_COLS=[
    'CurrentBalance',
    'LastRepaymentAmount',
    'AverageMonthlySpend',
    'BalanceAverage',
    'BalanceMin',
    'TotalAssets',
    'AverageNumberOfTransactionsADay',
    'TotalCash',
    'Paycheck',
    'TotalHistoryInDays',
    'AverageMonthlyIncome',
    'AverageMonthlyDiscretionarySpend',
    'OutstandingCreditDebtWherePayingInterest',
    'AverageNumberOfTransactionsADayPrimaryChecking',
    'ErrorRate',
    'AveragePotentialMonthlyIncome'
    ]

CAT_COLS=['PaycheckModelUsed']


DEFAULT_XGBOOST_HPS = {
    'max_depth': [3, 10],  # Maximum depth of a tree
    'learning_rate': [0.001, 0.01, 0.1],  # Step size shrinkage
    'n_estimators': [100, 1000],  # Number of boosting rounds
    'min_child_weight': [1, 10],  # Minimum sum of instance weight (hessian) needed in a child
    'gamma': [0, 0.1, 1],  # Minimum loss reduction required to make a further partition on a leaf node
    'subsample': [0.5, 1.0],  # Fraction of training instances used to grow trees
    'colsample_bytree': [1.0],  # Fraction of features used to train each tree
    'colsample_bylevel': [1.0],  # Fraction of features for each level in each tree
    'lambda': [0.01, 1],  # L2 regularization term on weights
    'alpha': [0.01, 1],  # L1 regularization term on weights
    'scale_pos_weight': [1, 10],  # Control for imbalanced classes
}

