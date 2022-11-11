import numpy as np
from typing import List


def make_tabular_ts(items: List[float], time_lag=4):
    # should only use for linear regression and stuff
    output = []
    
    for i in range(len(items) - time_lag - 1):
        output.append(items[i:i + time_lag + 1])
        
    return np.array(output).astype(np.float32)