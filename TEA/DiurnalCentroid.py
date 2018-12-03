# Uses the numpy package for matrix operations
import numpy as np

def DiurnalCentroid(flux,nStepsPerDay=48):
    '''DiurnalCentroid(flux)

    Diurnal centroid of sub-daily fluxes

    Calculates the daily flux weighted time of a sub-daily flux.


    Parameters
    ----------
    flux : list or list like
        sub-daily flux that must be condinuous and regular
    nStepsPerDay : integer
        frequency of the sub-daily measurements, 48 for half hourly measurements

    Returns
    -------
    array
        The diurnal centroid, in the same units as nStepsPerDay, at a daily frequency
    '''

    # calculate the total number of days
    days,UPD=flux.reshape(-1,nStepsPerDay).shape
    # create a 2D matrix providing a UPD time series for each day, used in the matrix operations.
    hours=np.tile(np.arange(UPD),days).reshape(days,UPD)
    # calculate the diurnal centroid
    C=np.sum(hours*flux.reshape(-1,nStepsPerDay),axis=1)/np.sum(flux.reshape(-1,nStepsPerDay),axis=1)
    C=C*(24/nStepsPerDay)
    return(C)
