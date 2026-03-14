import numpy as np
import pandas as pd
"""
This function will help build all 16 components of the KS
"""
def build_HAR_components(df):
    ret = df["ret"].values.copy()
    ret_num = len(ret)

    RV = ret ** 2 #RV: Squared Daily return

    abs_ret = np.abs(ret)#absolute return
    BPV = np.concatenate([[0.0], abs_ret[:-1]*abs_ret[1:]]) #multiply consecutive returns, NOTE: we append 0 at the front because the product would lead to n-1 values instead
    
    BPV_std = pd.Series.rolling(21, min_periods=5).std().values #computes std of BPV over 21 day windoe for each day
    BPV_std = np.nan_to_num(BPV_std, nan=1e-8) #any values with Nan in BPV_std can be replaced with a negligable number for data processing 
    BPV_std = np.where(BPV_std == 0, 1e-8, BPV_std) #any values with 0 in BPV_std can be replaced with a negligable number for data processing
    #we ensure std is not 0 or Nan as we dividde in jumps

    ABD_jump = np.maximum(RV-BPV, 0.0) #if RV is bigger than BPV, then a jump else if similar then 0
    ABD_CSP = RV - ABD_jump #RV is in two parts, the ABD_CSP and ABD_jump, CSP is the smooth variance after removing the jump


    BNS_jump = np.where(RV > 3.0 * BPV_std, ABD_jump,0.0) #only use ABD jump if RV exceeds 3 std of BPV else 0
    BNS_CSP = RV - BNS_jump #RV is in two parts, the BNS_CSP and BNS_jump, CSP is the smooth variance after removing the jump

    # Do Note: difference here is that we check ret instead of RV as proposed in the paper
    Jo_jump = np.where(np.abs(ret)>2.0*BPV_std, ABD_jump, 0.0) #only use ABD jump if absolute return exceeds 2 std of BPV else 0    
    Jo_CSP = RV-Jo_jump #RV is in two parts, the Jo_CSP and Jo_jump, CSP is the smooth variance after removing the jump

    RS_positive = np.where(ret>=0, RV, 0.0) #Realised semi variance, note that the addition of both negative and positive semi variance gets the RV
    RS_negative = np.where(ret<0, RV, 0.0)

    SJ = RS_positive - RS_negative #Signed jump, the difference between positive and negative semi variance, if positive then more positive jumps than negative jumps and vice versa
    SJ_positive = np.where(SJ>0, SJ, 0.0)
    SJ_negative = np.where(SJ<0, SJ, 0.0)

    negative_RV = np.where(ret<0, RV, 0.0)#where daily retun is negative, the negative RV exists

    TQ = np.abs(ret) **(4.0/3.0) #Tripower Quarticity, a measure of estimating var of RV 

    KS_Components = pd.DataFrame({
        "RV": RV,
        "BPV": BPV,
        "ABD_jump": ABD_jump,
        "ABD_CSP": ABD_CSP,
        "BNS_jump": BNS_jump,
        "BNS_CSP": BNS_CSP,
        "Jo_jump": Jo_jump,
        "Jo_CSP": Jo_CSP,
        "RS_positive": RS_positive,
        "RS_negative": RS_negative,
        "ret": ret,
        "SJ": SJ,
        "SJ_positive": SJ_positive,
        "SJ_negative": SJ_negative,
        "negative_RV": negative_RV,
        "TQ": TQ
    }, index = df.index)
    return KS_Components

components_inorder = ["RV", "BPV", "ABD_jump", "ABD_CSP", "BNS_jump", "BNS_CSP", "Jo_jump", "Jo_CSP", "RS_positive", "RS_negative", "ret", "SJ", "SJ_positive", "SJ_negative", "negative_RV", "TQ "]#rows of the 16x16 image

#This is what the CNN will refer to later when processing 16x16 image
def build_labels(KS_components):
    RV = KS_components["RV"].values

    label = np.where(np.roll(RV, -1) < RV, 1, 0) #looks at tomorrow and today and if threshold met, then produces 1 or 0
    label[-1] = 0 #last day has no tomorrow 
    return label 

#note that the image is 16x16 where 16 rows ar e the HAR components above and the 16 columns are the components over different time windows

def compute_rolling_window(series, lags): #lag is time horizon, series is one column of one of the KS components
    length = len(series) 
    arr = np.zeros((length, len(lags))) #creates empty array of n rows(days) and 16 columns
    #we use 16 different horizons, and length n for now 
    #16 time horizons means 16 columns
    for i, lag in enumerate(lags):
        if lag == 1:
            arr[:, i] = series #window of 1 requires doing nothing, use the raw
        else:
            arr[:,i] = pd.Series(series).rolling(lag, min_periods=1).mean().values #for window of 2,3,5,10,21, we compute the rolling mean over the window size and fill in the array column by column

    return arr #essentially, go through every day per lag and do rolling window mean stuff and repeat for every lag 

