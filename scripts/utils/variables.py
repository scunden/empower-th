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


DEFAULT_DATA_PROCESS={
    'drop':True, 
    'impute':True, 
    'resample':False, 
    'scale':False, 
    'reduce':False
}
DEFAULT_HPS = {
    'max_depth': [3, 5, 10],  # Maximum depth of a tree
    'learning_rate': [0.01, 0.1, 1.0],  # Step size shrinkage
    'n_estimators': [10, 100],  # Number of boosting rounds
}