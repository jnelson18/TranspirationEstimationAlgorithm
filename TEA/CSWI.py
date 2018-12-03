"""
Created on Fri Aug 26 17:32:13 2016

@author: Jacob A. Nelson and Martin Jung
"""

# Uses the numpy package for matrix operations
import numpy as np

def CSWI(precip,ET,s0=5):
    '''CSWI(precip,ET,s0=5,ConvertET=True)

    Modified Water Balance (CSWI)

    CSWI forces a positive water water storage for any time-step with precipitation, yet has a maximum storage of s0.


    Parameters
    ----------
    precip : list or list like
        precipitation in mm per half hour
    ET : list or list like
        Evapotranspiration in mm per half hour
    s0 : value, float or int
        The maximum storage of the water balance

     Returns
    -------
    array
        The modified water balance
    '''
    # insure that datasets are one dimensional
    precip=precip.reshape(-1)
    ET=ET.reshape(-1)
    # extract the data in case that precip has an associated mask
    precipcalc=np.ma.getdata(precip)

    # in case of an associated mask, fill all gap values with the maximum water storage
    if np.ma.is_masked(precip):
        precipcalc[np.ma.getmask(precip)]=s0
    else:
    # in case of any negative values in precipitation fill with the maximum water storage
        precipcalc[(precip<0)]=s0
    # fill any other missing values with the maximum water storage
    precipcalc[np.isnan(precip)]=s0
    precipcalc[np.isinf(precip)]=s0

    # create an array of zeros the same shape as the precipitation data to hold the CSWI
    CSWI=np.zeros(precip.shape)
    # set the initial value of CSWI to the max storage capacity
    CSWI[0]=s0

    # set any missing values in ET data to a number
    ET[np.isnan(ET)]=-9999
    ET[np.isinf(ET)]=-9999

    # loop through each timestep, skipping the inital condition
    for j in range(precip.shape[0]-1):
        k=j+1
        # stepVal give either the current water balance or s0, causing s0 to be a ceiling
        stepVal=min(CSWI[j]+precipcalc[k]-ET[k],s0)
        # in case of a positive precip value, the current CSWI is the max between the previous
        # CSWI and either the value of the precip or the s0 depending on which is smaller
        if precipcalc[k]>0:
            CSWI[k]=max(stepVal,min(precipcalc[j],s0))
        # if there is no precip, the CSWI is according to the stepVal,
        # causing simple water balance behaviour
        else:
            CSWI[k]=stepVal
    return(CSWI)
