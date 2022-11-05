import numpy as np

#implement onconditional coverage
def unconditional_coverage(PIs, data_test):
    indicator = []
    for i in range(len(PIs)):
        if PIs[i][0] < data_test[i] < PIs[i][1]:
            indicator.append(1)
        else:
            indicator.append(0)
    uc = np.sum(indicator)/len(indicator)
    return uc

#implement Winkler's score
def winlers_score(PIs, data_test, tau):
    indicators = []
    for i in range(len(PIs)):
        if PIs[i][0] < data_test[i] < PIs[i][1]:
            score = PIs[i][1]-PIs[i][0]
            indicators.append(score)
        elif data_test[i] < PIs[i][0]:
            score = (PIs[i][1]-PIs[i][0]) + 2 / tau * (PIs[i][0]-data_test[i])
            indicators.append(score)
        elif PIs[i][1] < data_test[i]:
            score = (PIs[i][1]-PIs[i][0]) + 2 / tau * (data_test[i]-PIs[i][1])
            indicators.append(score)
    wc = np.sum(indicators)/len(indicators)
    return wc
