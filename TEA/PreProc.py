# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 15:28:46 2018

@author: jnelson
"""

import xarray as xr
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import fmin

from sklearn.ensemble import RandomForestRegressor

from TEA.CSWI import CSWI
from TEA.DiurnalCentroid import DiurnalCentroid
import TEA.DWCI as DWCI
from TEA.TEA import partition

def quantreg(x,y,PolyDeg=1,rho=0.95,weights=None):
    '''quantreg(x,y,PolyDeg=1,rho=0.95)

    Quantile regression

    Fits a polynomial function (of degree PolyDeg) using quantile regression based on a percentile (rho).
    Based on script by Dr. Phillip M. Feldman, and based on method by Koenker, Roger, and
    Gilbert Bassett Jr. “Regression Quantiles.” Econometrica: Journal of
    the Econometric Society, 1978, 33–50.


    Parameters
    ----------
    x : list or list like
        independent variable
    y : list or list like
        dependent variable
    PolyDeg : int
        The degree of the polynomial function
    rho : float between 0-1
        The percentile to fit to, must be between 0-1
    weights : list or list like
        Vector to weight each point, must be same size as x

     Returns
    -------
    list
        The resulting parameters in order of degree from low to high
    '''
    def model(x, beta):
       """
       This example defines the model as a polynomial, where the coefficients of the
       polynomial are passed via `beta`.
       """
       if PolyDeg == 0:
           return x*beta
       else:
           return np.polyval(x, beta)

    N_coefficients=PolyDeg+1

    def tilted_abs(rho, x, weights):
       """
       OVERVIEW

       The tilted absolute value function is used in quantile regression.


       INPUTS

       rho: This parameter is a probability, and thus takes values between 0 and 1.

       x: This parameter represents a value of the independent variable, and in
       general takes any real value (float) or NumPy array of floats.
       """

       return weights * x * (rho - (x < 0))

    def objective(beta, rho, weights):
       """
       The objective function to be minimized is the sum of the tilted absolute
       values of the differences between the observations and the model.
       """
       return tilted_abs(rho, y - model(x, beta), weights).sum()

    # Build weights if they don't exits:
    if weights is None:
        weights=np.ones(x.shape)

    # Define starting point for optimization:
    beta_0= np.zeros(N_coefficients)
    if N_coefficients >= 2:
       beta_0[1]= 1.0

    # `beta_hat[i]` will store the parameter estimates for the quantile
    # corresponding to `fractions[i]`:
    beta_hat= []

    #for i, fraction in enumerate(fractions):
    beta_hat.append( fmin(objective, x0=beta_0, args=(rho,weights), xtol=1e-8,
      disp=False, maxiter=3000) )
    return(beta_hat)


def QuantRegDetector(x,y,TrainingMask=None,strictness=3):
    '''QuantRegDetector(x,y,TrainingMask=None,strictness=3)

    quantile regression outlier detectior

    Uses a mixture of quantile regression and quantiles of residuals to identify outliers in y with respect to x.


    Parameters
    ----------
    x: list or list like
        predictor variable
    y: list or list like
        response variable
    TrainingMask: list or list like
        boolean mask of data to ignore
    strictness: integer
        number of inter quartile ranges to use

     Returns
    -------
    bool array
        outliers detected
    '''
    x     = np.array(x)
    y     = np.array(y)
    index = np.arange(y.size)
    if TrainingMask is None:
        TrainingMask=np.ones(x.shape).astype(bool)

    if np.any(np.isnan(x)):
        raise RuntimeError("NaN values detected in x variable")
    if np.any(np.isnan(y)):
        raise RuntimeError("NaN values detected in y variable")

    si75 = quantreg(x[TrainingMask],y[TrainingMask],PolyDeg=0,rho=0.75)[0]
    si25 = quantreg(x[TrainingMask],y[TrainingMask],PolyDeg=0,rho=0.25)[0]
    si50 = quantreg(x[TrainingMask],y[TrainingMask],PolyDeg=0,rho=0.5)[0]

    Q75  = x*si75[0] #+si75[0]
    Q25  = x*si25[0] #+si25[0]

    IQrange      = np.subtract(Q75,Q25)
    LowerBound   = Q25-strictness*IQrange #1.5
    UpperBound   = Q75+strictness*IQrange #1.5
    resid        = x*si50[0] - y
    IQresid       = np.diff(np.percentile(resid,[25,75]))
    Lowerresid   = x*si50[0]-strictness*IQresid #1.5
    Upperresid   = x*si50[0]+strictness*IQresid #1.5
    Outliers     = ~np.logical_or(np.logical_and(y<=UpperBound,y>=LowerBound),np.logical_and(resid<=Upperresid,resid>=Lowerresid))
    return(~Outliers)

def outlierCorrection(ds, n_jobs=1):
    for percentile in ds.percentiles:
        for CSWIlim in ds.CSWIlims:
            T        = ds['TEA_T'].sel(percentiles=percentile,CSWIlims=CSWIlim).values
            nanMask  = ~np.isnan(T) & ~np.isnan(ds.Rg.values)
            xVars    = ds.features.split(',')
            for var in xVars:
                nanMask[np.isnan(ds[var])] = False
                nanMask[ds[var]<-9000] = False
            RFxs     = np.asanyarray([ds[v].values for v in xVars])
            RFxs     = np.matrix(RFxs).T
            Outliers = QuantRegDetector(ds.Rg.values,T,TrainingMask=nanMask)
            if np.all(Outliers):
                continue
            else:
                Forest   = RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=n_jobs, verbose=0, warm_start=False)
                Forest.fit(RFxs[Outliers & nanMask],T[Outliers & nanMask])
                T[~Outliers & nanMask] = Forest.predict(RFxs[~Outliers & nanMask])
                ds['TEA_T'].sel(percentiles=percentile,CSWIlims=CSWIlim).values=T

def tempFlag(Tair):
    '''tempFlag(Tair)

    Tair limit flag

    Calculates the air temperature limit flag with limits of 5 deg C.
    True indicates the time period should be included


    Parameters
    ----------
    Tair: list or list like
        air temperature (deg C)

     Returns
    -------
    bool array
        tempFlag
    '''
    tempdaymin=5
    return(Tair>tempdaymin)

def GPPFlag(GPP,nStepsPerDay):
    '''GPPFlag(GPP)

    GPP limit flag

    Calculates the GPP limit flag with limits of 2 gC m-1 d-1 and 0.05 umol m-2 s-1.
    True indicates the time period should be included


    Parameters
    ----------
    GPP : list or list like
        gross primary productivity (umol m-2 s-1)

     Returns
    -------
    bool array
        GPPFlag
    '''
    GPPdaymin=0.5
    GPPmin=0.05
    GPPFlag=np.repeat((umolC_per_s_to_gC_per_d(GPP,nStepsPerDay)>GPPdaymin),nStepsPerDay)
    GPPFlag=GPPFlag*(GPP>GPPmin)
    return(GPPFlag)

def umolC_per_s_to_gC_per_d(GPP,nStepsPerDay):
    '''umolC_per_s_to_gC_per_d(GPP)

    convert umolC s-1 to gC d-1

    Convert umolC s-1 to gC d-1 returning an array of the same resolution


    Parameters
    ----------
    GPP : list or list like
        gross primary productivity (umol m-2 s-1)

     Returns
    -------
    array
        daily GPP in gC d-1
    '''
    GPPnew=GPP*12.011 # ug per umol
    GPPnew=GPPnew/1000000 # ug to g
    GPPnew=GPPnew*1800 # s per hh
    GPPday=np.sum(GPPnew.reshape(-1,nStepsPerDay),axis=1)
    return(GPPday)

def GPPgrad(GPP,nStepsPerDay):
    '''GPPgrad(GPP)

    GPP gradient

    Calculates the seasonal GPP gradiant from smoothed GPP.


    Parameters
    ----------
    GPP : list or list like
        gross primary productivity (umol m-2 s-1)

     Returns
    -------
    array
        GPPgrad
    '''
    gradGPP=GPP
    gradGPP[np.isnan(gradGPP)]=0
    gradGPP[(gradGPP<-9000)]=0
    GPPgrad=np.repeat(np.gradient(gaussian_filter(gradGPP.reshape(-1,nStepsPerDay).mean(axis=1),sigma=[20])),nStepsPerDay)
    GPPgrad[(GPP<-9000)]=np.nan
    GPPgrad[np.isnan(GPP)]=np.nan
    GPPgrad[0]=0
    return(GPPgrad)

def build_dataset(timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u, OtherVars=None):
    """build_dataset(timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u, OtherVars=None)

    build dataset for partitioning

    Uses the imput variables to build an xarray dataset which will be used in the partitioning.


    Parameters
    ----------
    timestamp : datetime64[ns] array
        timestamp of all variables in
    ET: list or list like
        evapotranspiration (mm hh-1)
    GPP: list or list like
        gross primary productivity (umol C m-2 s-1)
    Tair: list or list like
        air temperature (deg C)
    RH: list or list like
        relative humidity (%)
    VPD: list or list like
        vapor pressure deficit (hPa)
    precip: list or list like
        precipitation (mm hh-1)
    Rg: list or list like
        incoming shortwave radiation (W m-2)
    Rg_pot: list or list like
        potential incoming shortwave radiation (W m-2)
    u: list or list like
        wind speed (m s-1)
    OtherVars: dictionary
        dictionary of other variables to include in dataset (optional)

    Recommended Parameters (add with OtherVars)
    ----------
    NEE: list or list like
        net ecosystem exchange (umol C m-2 s-1)
    NEE_sd: list or list like
        uncertainty of net ecosystem exchange (umol C m-2 s-1)
    ET_sd: list or list like
        uncertainty of evapotranspiration (mm hh-1)
    ET_fall: list or list like
        ET with all periods filled as gaps (mm hh-1)
    NEE_fall: list or list like
        NEE with all periods filled as gaps (mm hh-1)

     Returns
    -------
    xarray Dataset
        dataset used to pass to preprocess
    """

    UnitAttDic = {'ET':{'Units':'mm hh-1','long name':'evapotranspiration'},
              'GPP':{'Units':'umol C m-2 s-1','long name':'gross primary productivity'},
              'Tair':{'Units':'deg C','long name':'air temperature'},
              'VPD':{'Units':'hPa','long name':'vapor pressure deficit'},
              'RH':{'Units':'%','long name':'relative humidity'},
              'Rg':{'Units':'W m-2','long name':'incoming shortwave radiation'},
              'Rg_pot':{'Units':'W m-2','long name':'potential incoming shortwave radiation'},
              'precip':{'Units':'mm hh-1','long name':'precipitation'},
              'u':{'Units':'m s-1','long name':'wind speed'},
                }

    CoreVars = {'ET':ET, 'GPP':GPP, 'RH':RH, 'Rg':Rg,
                'Rg_pot':Rg_pot, 'Tair':Tair, 'VPD':VPD, 'precip':precip, 'u':u}
    InputDic = {}
    for var in CoreVars:
        InputDic[var] = (['timestamp'], CoreVars[var])
    if OtherVars is not None:
        for var in OtherVars.keys():
            InputDic[var] = (['timestamp'], OtherVars[var])
    ds = xr.Dataset(InputDic,coords={'timestamp':timestamp})
    for var in UnitAttDic.keys():
        ds[var] = ds[var].assign_attrs(UnitAttDic[var])
    return(ds)

def preprocess(ds,nStepsPerDay=48):
    '''preprocess(ds)

    preprocess for partitioning

    Builds all the derived variables used in the partitioning such as CSWI,
    DWCI, etc., as well as filters which remove night time, low air temp/GPP, etc. periods.
    input dataset (ds) must contain at least timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u


    Parameters
    ----------
    ds : xarray dataset
        must contain variables which are inputs for build_dataset
    nStepsPerDay : int
        the number of time steps within a day, 48 for half-hourly files.

     Returns
    -------
    xarray Dataset
        dataset used to partition
    '''
    # calculate the CWSI
    ds['CSWI'] = (('timestamp'), CSWI(ds.precip.values.copy(),ds.ET.values.copy()))
    ds['CSWI'] = ds['CSWI'].assign_attrs({'long name':'Conservative Surface Water Index','Units':'mm hh-1'})

    # convert a half hourly resolution Rg_pot into a daily Rg_pot
    ds['Rg_pot_daily'] = (('timestamp'),np.repeat(ds.Rg_pot.values.reshape(-1,nStepsPerDay).sum(axis=1),nStepsPerDay)*((3600*(24/nStepsPerDay))/1000000))
    ds['Rg_pot_daily'] = ds['Rg_pot_daily'].assign_attrs({'long name':'daily potential radiation','Units':'MJ m-2 d-1'})

    # extract year to be used as a predictor variable
    ds['year'] = (('timestamp'),ds.timestamp.values.astype('datetime64[Y]').astype(int) + 1970)

    # calculates the diurnal centroids of Rg and ET, giving returning the difference of the two
    C_ET          = np.repeat(DiurnalCentroid(ds.ET.copy().values,nStepsPerDay),nStepsPerDay)
    C_Rg          = np.repeat(DiurnalCentroid(ds.Rg.copy().values,nStepsPerDay),nStepsPerDay)
    ds['C_Rg_ET'] = (('timestamp'),C_ET-C_Rg)
    ds['C_Rg_ET'] = ds['C_Rg_ET'].assign_attrs({'long name':'normalized diurnal centroid','Units':'hours'})

    # caluculates a DWCI depending on variables in the dataset.
    try:
        # first case if is both the standard deviation and fully gap-filled variables (_fall) from
        # the eddy covariance partitioning are present, which will take into account both
        # uncertanty in GPP (from NEE) and ET, as well as correlation structure between the two
        if 'ET_sd' in ds.data_vars and 'NEE_sd' in ds.data_vars and 'NEE' in ds.data_vars and 'ET_fall' in ds.data_vars and 'NEE_fall' in ds.data_vars:
            ds['DWCI']=(('timestamp'),np.repeat(DWCI.DWCIcalc(ds.Rg_pot.copy().values,ds.ET.copy().values,ds.GPP.copy().values,ds.VPD.copy().values,ds.NEE.copy().values,ds.ET_sd.copy().values,ds.GPP_sd.copy().values,ds.NEE_fall.copy().values,ds.ET_fall.copy().values,nStepsPerDay),nStepsPerDay))
        # second case will only take into account the uncertanties because the fully gap-filled
        # variables are not in the dataset
        elif 'ET_sd' in ds.data_vars and 'NEE_sd' in ds.data_vars and 'NEE' in ds.data_vars:
            ds['DWCI']=(('timestamp'),np.repeat(DWCI.DWCIcalcSimple(ds.Rg_pot.copy().values,ds.ET.copy().values,ds.GPP.copy().values,ds.VPD.copy().values,ds.NEE.copy().values,ds.ET_sd.copy().values,ds.GPP_sd.copy().values,nStepsPerDay),nStepsPerDay))
        # final case is the simplest, using only GPP, ET, and VPD. This is the case used when partitioning
        # models as there is no uncertanty
        else:
            GPPxsqrtVPD = ds.VPD.copy().values
            GPPxsqrtVPD[GPPxsqrtVPD<0]=0
            GPPxsqrtVPD = ds.GPP.values * np.sqrt(GPPxsqrtVPD)
            ds['DWCI']=(('timestamp'),np.repeat(DWCI.daily_corr(ds.ET.copy().values, GPPxsqrtVPD, ds.Rg_pot.copy().values,nStepsPerDay),nStepsPerDay))
    except ValueError:
        # returns a nan array if all other methods are unsuccessful
        ds['DWCI']=(('timestamp'),np.zeros(ds.Rg_pot.shape)*np.nan)
    ds['DWCI'] = ds['DWCI'].assign_attrs({'long name':'diurnal water:carbon index','Units':'unitless'})

    # calculate the daily smoothed GPP gradient, which gives and indication of phenology
    ds['GPPgrad']=(('timestamp'),GPPgrad(ds.GPP.copy().values,nStepsPerDay))
    ds['GPPgrad'] = ds['GPPgrad'].assign_attrs({'long name':'smoothed daily GPP gradient','Units':'umol C m-2 s-1 d-1'})

    # calculate the daily smoothed Rg_pot gradient, which gives an indication of time of year (spring/summer)
    try:
        ds['Rgpotgrad']=(('timestamp'),np.gradient(ds.Rg_pot))
        ds['Rgpotgrad_day']=(('timestamp'),np.repeat(np.gradient(ds.Rg_pot.values.reshape(-1,nStepsPerDay).mean(axis=1) ),nStepsPerDay))
    except ValueError:
        ds['Rgpotgrad']=(('timestamp'),np.ones(ds.ET.size)*np.nan)
        ds['Rgpotgrad_day']=(('timestamp'),np.ones(ds.ET.size)*np.nan)
    ds['Rgpotgrad'] = ds['Rgpotgrad'].assign_attrs({'long name':'smoother Rg_pot gradient','Units':'W m-2 hh-1'})
    ds['Rgpotgrad_day'] = ds['Rgpotgrad_day'].assign_attrs({'long name':'smoothed daily GPP gradient','Units':'W m-2 d-1'})

    # calcualte all flags used to build the training dataset
    def flags(ds):
        ds['DayNightFlag']=(ds['Rg_pot']>0)
        ds['posFlag']=(('timestamp'),(ds.GPP.values>0) & (ds.ET.values>0))
        ds['tempFlag']=(('timestamp'),tempFlag(ds.Tair.values))
        ds['GPPFlag']=(('timestamp'),GPPFlag(ds.GPP.values,nStepsPerDay))
        ds['seasonFlag']=(('timestamp'),ds.tempFlag.values*ds.GPPFlag.values)
        return(ds)

    # calculate the instant WUE (inst_WUE) which is used as the target variable
    ds['inst_WUE']=ds.ET.copy()
    ds['inst_WUE'][ds.ET.values<=0] = 0
    ds['inst_WUE'][ds.ET.values>0] = ds.GPP[ds.ET.values>0]/ds.inst_WUE[ds.ET.values>0]
    ds['inst_WUE']=ds['inst_WUE']*((12*1800)/1000)
    ds['inst_WUE'] = ds['inst_WUE'].assign_attrs({'long name':'instant water use efficiency (GPP/ET)','Units':'g C per kg H2O'})

    # builds a quality flag if none is supplied. quality flag should remove all values not
    # to be used in the training dataset, such as when GPP or ET values are gap filled.
    if 'qualityFlag' not in list(ds.variables):
        ds['qualityFlag'] = (('timestamp'),np.ones(ds.ET.size).astype(bool))
    ds = flags(ds)

    return(ds)

def simplePartition(timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u):
    '''simplePartition(ds)

    Performs a basic partitioning using default parameters with basic inputs.


    Parameters
    ----------
    timestamp : datetime64[ns] array
        timestamp of all variables in
    ET: list or list like
        evapotranspiration (mm hh-1)
    GPP: list or list like
        gross primary productivity (umol m-2 s-1)
    Tair: list or list like
        air temperature (deg C)
    RH: list or list like
        relative humidity (%)
    VPD: list or list like
        vapor pressure deficit (hPa)
    precip: list or list like
        precipitation (mm hh-1)
    Rg: list or list like
        incoming shortwave radiation (W m-2)
    Rg_pot: list or list like
        potential incoming shortwave radiation (W m-2)
    u: list or list like
        wind speed (m s-1)
    OtherVars: dictionary
        dictionary of other variables to include in dataset (optional)
     Returns
    -------
    TEA_T: array
        transpiration from TEA
    TEA_E: array
        evaporation from TEA
    TEA_WUE: array
        water use efficiency from TEA
    '''
    ds   = build_dataset(timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u)
    ds   = preprocess(ds)

    RFmod_vars=['Rg','Tair','RH','u','Rg_pot_daily',
            'Rgpotgrad','year','GPPgrad','DWCI','C_Rg_ET','CSWI']
    ds=partition(ds,
           percs=np.array([75]),
           CSWIlims=np.array([-0.5]),
           RFmod_vars=RFmod_vars
           )
    TEA_T   = ds.TEA_T.values.ravel()
    TEA_E   = ds.TEA_E.values.ravel()
    TEA_WUE = ds.TEA_WUE.values.ravel()
    return(TEA_T,TEA_E,TEA_WUE)
