# -*- coding: utf-8 -*-
"""
Created on Mon May 29 15:54:06 2017

@author: jnelson
"""

import numpy as np
import os
import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from JMaths.Regs import RFPercentilePrediction

def partition(ds,percs=np.linspace(20,60,5),lims=np.array([-1]),n_jobs=47,RFmod_vars=['Rg','Tair','RH','u','month','year','GPPgrad','DWCI','C_Rg_LE','MWB']):
    RFxs=[ds[x].values for x in RFmod_vars]
    RFxs=np.matrix(RFxs).T
    RFxs[np.isnan(RFxs)]=-9999

    ds['Baseflag']=ds.DayNightFlag & ds.qualityFlag & ds.seasonFlag & ds.posFlag
    ds.attrs['features']=','.join(RFmod_vars)
    
    ds.coords['percentiles']=percs
    ds.coords['MWBlims']=lims
    ds.coords['RFmod_vars']=RFmod_vars
    ds['nanflag']=(('timestamp'),((~np.isnan(ds.ET)) & (~np.isnan(ds.GPP))).astype(bool))
    
    ds['Nelson_WUE']=(('timestamp','percentiles','MWBlims'),np.zeros((ds.timestamp.size,percs.size,lims.size))-9999)
    ds['Nelson_T']=(('timestamp','percentiles','MWBlims'),np.zeros((ds.timestamp.size,percs.size,lims.size))-9999)
    ds['Nelson_E']=(('timestamp','percentiles','MWBlims'),np.zeros((ds.timestamp.size,percs.size,lims.size))-9999)

    ds['RFflag']=(('timestamp','MWBlims'),np.zeros((ds.timestamp.size,lims.size))-9999)
    ds['feature_importances']=(('RFmod_vars','MWBlims'),np.zeros((len(RFmod_vars),lims.size))-9999)
    ds['NumForestPoints']=(('MWBlims'),np.zeros((lims.size))-9999)
    ds['oob_scores']=(('MWBlims'),np.zeros((lims.size))-9999)

    for var in RFmod_vars:
        if np.any(np.isnan(ds[var][ ds['nanflag']])) or np.any(ds[var][ ds['nanflag']]<-9000):
            RFmod_vars.remove(var)

    for l in range(lims.size):
        CurFlag=ds.Baseflag.values*(ds.MWB.values<ds.coords['MWBlims'].values[l])
        ds['RFflag'][:,l]=CurFlag
        ds['NumForestPoints'][l]=CurFlag.sum()
        
        if CurFlag.sum()>240:
            Forest=RandomForestRegressor(n_estimators=100, oob_score=True, n_jobs=n_jobs, verbose=0, warm_start=False)
            Forest.fit(RFxs[CurFlag],ds.instET_WUE.values[CurFlag])
    
            ds['oob_scores'][l]=Forest.oob_score_

            for j in range(len(RFmod_vars)):
                ds.feature_importances.sel(RFmod_vars=RFmod_vars[j])[l]=Forest.feature_importances_[j]

            ds['Nelson_WUE'][:,:,l][ds['nanflag'].values]=RFPercentilePrediction(Forest,RFxs[CurFlag],
                                                                ds.instET_WUE.values[CurFlag],
                                                                RFxs[ds['nanflag'].values],
                                                                ds.percentiles.values,n_jobs=n_jobs)#.flatten()

            ds['Nelson_T'][:,:,l]=ds['Nelson_WUE'][:,:,l]*ds.GPP
            ds['Nelson_E'][:,:,l]=ds.ET-ds['Nelson_WUE'][:,:,l]    
    
    ds.attrs['PartRunTimestamp']=datetime.datetime.now().strftime("%A, %d. %B %Y %I:%M%p")
    try:
        ds.attrs['PartType']=os.path.basename(__file__)
    except NameError:
        ds.attrs['PartType']='manual'
    return(ds)
