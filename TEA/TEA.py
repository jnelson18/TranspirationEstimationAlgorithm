# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:54:06 2017

@author: jTEA
"""

import numpy as np
import os
import datetime
from pyquantrf import QuantileRandomForestRegressor
from TEA.PreProc import build_dataset, preprocess

def partition(ds,
        percs=np.linspace(50,100,11),CSWIlims=np.array([-1]),
        RFmod_vars=['Rg','Tair','RH','u','Rg_pot_daily','Rgpotgrad','year','GPPgrad','DWCI','C_Rg_ET','CSWI'],
        RandomForestRegressor_kwargs={'n_estimators':100, 'oob_score':True, 'max_features':"n/3", 'verbose':0, 'warm_start':False, 'n_jobs':1}):
    """partition(ds,
        percs=np.linspace(50,100,11), CSWIlims=np.array([-1]),
        RFmod_vars=['Rg','Tair','RH','u','Rg_pot_daily','Rgpotgrad','year','GPPgrad','DWCI','C_Rg_ET','CSWI'],
        RandomForestRegressor_kwargs={'n_estimators':100, 'oob_score':True, 'max_features':"n/3", 'verbose':0, 'warm_start':False, 'n_jobs':1})

    Runs TEA partitioning

    The function for running the TEA partitioning given a preprocessed dataset (ds), the predictor variables (RFmod_vars), and the Random Forest configuration variables (RandomForestRegressor_kwargs).


    Parameters
    ----------
    ds : xarray Dataset 
        Output from the preprocess function
    percs : numpy array
        The percentiles to output. Generally the 75th percentile is used. e.g. np.array([75])
    CSWIlims : numpy array
        The limits to use for the CSWI filter. Generally -1 is used. e.g. np.array([-1])
    RFmod_vars : list of strings
        A list of stings refering to the variables to be used as predictors (X) in the random forest regressor.
    RandomForestRegressor_kwargs : dict
        Arguments to be passed on to the RandomForestRegressor. Defaults are: {'n_estimators':100, 'oob_score':True, 'max_features':"n/3", 'verbose':0, 'warm_start':False, 'n_jobs':1}.
        See https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html


     Returns
    -------
    xarray Dataset 
        The resulting partitioned dataset.
    """
        
    Default_RandomForestRegressor_kwargs = {'n_estimators':100, 'oob_score':True, 'max_features':"n/3", 'verbose':0, 'warm_start':False, 'n_jobs':1}
    
    for key in Default_RandomForestRegressor_kwargs.keys():
        if key not in RandomForestRegressor_kwargs.keys():
            RandomForestRegressor_kwargs[key] = Default_RandomForestRegressor_kwargs[key]
    
    RFxs = [ds[x].values for x in RFmod_vars]
    RFxs = np.asarray(RFxs).T
    RFxs[RFxs==-9999] = np.nan

    ds.attrs['features']=','.join(RFmod_vars)
    
    ds.coords['percentiles']=percs
    ds.coords['CSWIlims']=CSWIlims
    ds.coords['RFmod_vars']=RFmod_vars
    ds['nanflag']=(('timestamp'),(np.isfinite(ds.ET.values) & np.isfinite(ds.GPP.values)).astype(bool))
    
    ds['TEA_WUE']=(('timestamp','percentiles','CSWIlims'),np.zeros((ds.timestamp.size,percs.size,CSWIlims.size))-9999)
    ds['TEA_T']=(('timestamp','percentiles','CSWIlims'),np.zeros((ds.timestamp.size,percs.size,CSWIlims.size))-9999)
    ds['TEA_E']=(('timestamp','percentiles','CSWIlims'),np.zeros((ds.timestamp.size,percs.size,CSWIlims.size))-9999)

    ds['RFflag']=(('timestamp','CSWIlims'),np.zeros((ds.timestamp.size,CSWIlims.size))-9999)
    ds['feature_importances']=(('RFmod_vars','CSWIlims'),np.zeros((len(RFmod_vars),CSWIlims.size))-9999)
    ds['NumForestPoints']=(('CSWIlims'),np.zeros((CSWIlims.size))-9999)
    ds['oob_scores']=(('CSWIlims'),np.zeros((CSWIlims.size))-9999)

    for var in RFmod_vars:
        ds['nanflag'][np.isnan(ds[var])] = False
        ds['nanflag'][ds[var]<-9000] = False
        
    ds['Baseflag']=ds.DayNightFlag & ds.qualityFlag & ds.seasonFlag & ds.posFlag & ds.nanflag
    
    if RandomForestRegressor_kwargs['max_features']=='n/3':
        RandomForestRegressor_kwargs['max_features']=int(np.ceil(len(RFmod_vars)/3))

    for l in range(CSWIlims.size):
        CurFlag=ds.Baseflag.values*(ds.CSWI.values<ds.coords['CSWIlims'].values[l])
        ds['RFflag'][:,l]=CurFlag
        ds['NumForestPoints'][l]=CurFlag.sum()
        
        if CurFlag.sum()>240:
            qrf = QuantileRandomForestRegressor(nthreads = RandomForestRegressor_kwargs["n_jobs"],
                                    **RandomForestRegressor_kwargs)
            qrf.fit(RFxs[CurFlag],ds.inst_WUE.values[CurFlag])
    
            if RandomForestRegressor_kwargs["oob_score"]:
                ds['oob_scores'][l] = qrf.forest.oob_score_

            for j in range(len(RFmod_vars)):
                ds.feature_importances.sel(RFmod_vars=RFmod_vars[j])[l] = qrf.forest.feature_importances_[j]

            ds['TEA_WUE'][:,:,l][~ds['DayNightFlag'].values] = 0
            ds['TEA_WUE'][:,:,l][ds['nanflag'].values & ds['DayNightFlag'].values]= qrf.predict(RFxs[ds['nanflag'].values & ds['DayNightFlag'].values], percs/100)

            ds['TEA_T'][:,:,l]=ds.GPP/(ds['TEA_WUE'][:,:,l]*(1000/(12*1800)))
            ds['TEA_T'][:,:,l][~ds['DayNightFlag'].values] = 0
            ds['TEA_E'][:,:,l]=ds.ET-ds['TEA_T'][:,:,l]
            
            ds['TEA_T'] = ds['TEA_T'].assign_attrs({'long name':'TEA transpiration','Units':'mm hh-1'})
            ds['TEA_E'] = ds['TEA_E'].assign_attrs({'long name':'TEA evaporation','Units':'mm hh-1'})
            ds['TEA_WUE'] = ds['TEA_WUE'].assign_attrs({'long name':'TEA water use efficiency','Units':'g C per kg H2O'})

    ds['TEA_T'].values.reshape(-1)[(ds['TEA_WUE'].values.reshape(-1)==-9999)] = np.nan
    ds['TEA_E'].values.reshape(-1)[(ds['TEA_WUE'].values.reshape(-1)==-9999)] = np.nan
    ds['TEA_WUE'].values.reshape(-1)[(ds['TEA_WUE'].values.reshape(-1)==-9999)] = np.nan  
    
    ds.attrs['PartRunTimestamp']=datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    try:
        ds.attrs['PartType']=os.path.basename(__file__)
    except NameError:
        ds.attrs['PartType']='manual'
    return(ds)


def simplePartition(timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u, qualityFlag=None):
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
    ds   = build_dataset(timestamp, ET, GPP, RH, Rg, Rg_pot, Tair, VPD, precip, u, qualityFlag=qualityFlag)
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