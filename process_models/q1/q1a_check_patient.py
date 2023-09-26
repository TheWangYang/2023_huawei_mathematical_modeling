import numpy as np

# 超过33的为发生血肿
def up_relative33(data):
    if data[0]==0:
        print("data is zero, please check")
        return (np.array([], dtype=np.int64), )
    
    return np.where(((data - data[0])/data[0])>=0.33)
