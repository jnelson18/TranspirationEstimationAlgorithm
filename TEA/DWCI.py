import numpy as np

def daily_corr(x, y, Rg_pot,nStepsPerDay):
    '''daily_corr(x, y, Rg_pot)

    Daily correlation coefficient

    Calculates a daily correlation coefficient between two sub-daily timeseries


    Parameters
    ----------
    x : list or list like
        x variable
    y : list or list like
        y variable
    Rg_pot : list or list like
        potential radiation
     Returns
    -------
    array
        correlation coefficents at daily timescale
    '''
    x=x.reshape(-1,nStepsPerDay)
    y=y.reshape(-1,nStepsPerDay)
    Rg_pot=Rg_pot.reshape(-1,nStepsPerDay)
    mask=Rg_pot<=0
    x=np.ma.MaskedArray(x,mask=mask)
    y=np.ma.MaskedArray(y,mask=mask)
    x=x/x.max(axis=1)[:,None]
    y=y/y.max(axis=1)[:,None]
    mx = x.mean(axis=1)
    my = y.mean(axis=1)
    xm, ym = x - mx[..., None], y - my[..., None]
    r_num = np.ma.add.reduce(xm * ym, axis=1)
    r_den = np.ma.sqrt(np.ma.sum(xm**2, axis=1) * np.ma.sum(ym**2, axis=1))
    r = r_num / r_den
    r = np.ma.getdata(r)
    return(r**2)

def DWCIcalcSimple(Rg_pot,ET,GPP,VPD,NEE,ET_sd,GPP_sd,nStepsPerDay):
    '''DWCIcalc(Rg_pot,ET,GPP,ET_sd,GPP_sd,NEE_fall,ET_fall)

    Simplified diurnal water:carbon index (DWCI)

    DWCI measures the probability that the carbon and water are coupled within
    a given day. Simplified version does not take into account the correlation
    between ET and GPP due to missing NEE_fall and/or ET_fall.
    This correlation is then compare to a distribution of correlations
    between artificial datasets built from the signal of potential radiation
    and the uncertainty in the ET and GPP.

    Parameters
    ----------
    Rg_pot : list or list like
        Sub-daily timeseries of potential radiation
    ET : list or list like
        Sub-daily timeseries of evapotranspiration or latent energy
    GPP : list or list like
        Sub-daily timeseries of gross primary productivity
    VPD : list or list like
        Sub-daily timeseries of vapor pressure deficit
    NEE : list or list like
        Sub-daily timeseries of net ecosystem exchange
    ET_sd : list or list like
        Sub-daily estimation of the uncertainty of ET
    GPP_sd : list or list like
        Sub-daily estimation of the uncertainty of GPP

     Returns
    -------
    array
        The simplified diurnal water:carbon index (DWCI)
    '''
    repeats=100
    days=int(ET.size/nStepsPerDay)
    StN=np.zeros([repeats,days])
    for rep in range(repeats):
        GPP_noise=np.random.normal(scale=GPP_sd)
        ET_noise=np.random.normal(scale=ET_sd)
        GPP_DayCycle=(Rg_pot.reshape(-1,nStepsPerDay)*(GPP.reshape(-1,nStepsPerDay).mean(axis=1)/Rg_pot.reshape(-1,nStepsPerDay).mean(axis=1))[:,None]+GPP_noise.reshape(-1,nStepsPerDay)).reshape(-1)
        ET_DayCycle=(Rg_pot.reshape(-1,nStepsPerDay)*(ET.reshape(-1,nStepsPerDay).mean(axis=1)/Rg_pot.reshape(-1,nStepsPerDay).mean(axis=1))[:,None]+ET_noise.reshape(-1,nStepsPerDay)).reshape(-1)
        StN[rep]=daily_corr(ET_DayCycle,GPP_DayCycle,Rg_pot,nStepsPerDay)
    pwc=daily_corr(ET,GPP*np.sqrt(VPD),Rg_pot,nStepsPerDay)

    DWCI=(StN<np.tile(pwc,repeats).reshape(repeats,-1)).sum(axis=0)
    DWCI[np.isnan(StN).prod(axis=0).astype(bool)]=-9999

    return(DWCI)

def DWCIcalc(Rg_pot,ET,GPP,VPD,NEE,ET_sd,GPP_sd,NEE_fall,ET_fall):
    '''DWCIcalc(Rg_pot,ET,GPP,ET_sd,GPP_sd,NEE_fall,ET_fall)

    Diurnal water:carbon index (DWCI)

    DWCI measures the probability that the carbon and water are coupled within a given day. Method takes the correlation between evapotranspiration (ET) and gross primary productivity (GPP) and calculated the correlation within each day. This correlation is then compare to a distribution of correlations between artificial datasets built from the signal of potential radiation and the uncertainty in the ET and GPP.

    Parameters
    ----------
    Rg_pot : list or list like
        Sub-daily timeseries of potential radiation
    ET : list or list like
        Sub-daily timeseries of evapotranspiration or latent energy
    GPP : list or list like
        Sub-daily timeseries of gross primary productivity
    VPD : list or list like
        Sub-daily timeseries of vapor pressure deficit
    NEE : list or list like
        Sub-daily timeseries of net ecosystem exchange
    ET_sd : list or list like
        Sub-daily estimation of the uncertainty of ET
    GPP_sd : list or list like
        Sub-daily estimation of the uncertainty of GPP
    NEE_fall : list or list like
        Modeled sub-daily timeseries of net ecosystem exchange i.e. no noise
    ET_fall : list or list like
        Modeled sub-daily timeseries of evapotranspiration or latent energy i.e. no noise

     Returns
    -------
    array
        The diurnal water:carbon index (DWCI)
    '''

    # reshape all variables as number of days by number of half hours
    varList=[Rg_pot,ET,GPP,VPD,NEE,ET_sd,GPP_sd,NEE_fall,ET_fall]
    for j in range(len(varList)):
        varList[j]=varList[j].reshape(-1,nStepsPerDay)
    Rg_pot,ET,GPP,VPD,NEE,ET_sd,GPP_sd,NEE_fall,ET_fall=varList
    # the number of artificial datasets to construct
    repeats=100
    # the number of days in the timeseries. Assumes data is half hourly
    days=int(ET.shape[0])
    # creates an empty 2D dataset to hold the artificial distributions
    StN=np.zeros([repeats,days])*np.nan
    corrDev=np.zeros([days,2,2])#*np.nan

    # create the daily cycle by dividing Rg_pot by the daily mean
    daily_cycle=Rg_pot/Rg_pot.mean(axis=1)[:,None]
    mean_GPP=GPP.mean(axis=1)
    mean_ET=ET.mean(axis=1)

    # Isolate the error of the carbon and water fluxes.
    NEE_err=NEE_fall-NEE
    ET_err=ET_fall-ET

    # loops through each day to generate an artificial dataset and calculate the associate correlation
    for d in range(days):#days
        if np.isnan(mean_GPP[d]) or np.isnan(mean_ET[d]):
            continue
        if np.isnan(NEE_err[d]).sum()>0 or np.isnan(ET_err[d]).sum()>0 or np.isnan(GPP_sd[d]).sum()>0 or np.isnan(ET_sd[d]).sum()>0:
            continue
        # find the correlation structure of the uncertanties to pass onto the artificial datasets

        if np.all(ET_err[d]==0) or np.all(NEE_err[d]==0):
            corrDev[d]=np.identity(2)
        else:
            corrDev[d] = np.corrcoef(-(NEE_err[d]),ET_err[d])

        # create our synthetic GPP and ET values for the current day
        synGPP  = np.zeros((repeats,nStepsPerDay))*np.nan
        synET   = np.zeros((repeats,nStepsPerDay))*np.nan

        # this loop builds the artificial dataset using the covariance matrix between NEE and ET
        for i in range(nStepsPerDay):
            # compute the covariance matrix (s) for this half hour
            m   = [GPP_sd[d,i],ET_sd[d,i]]
            s   = np.zeros((2,2))*np.nan
            for j in range(2):
                for k in range(2):
                    s[j,k]    = corrDev[d,j,k]*m[j]*m[k]

            Noise    = np.random.multivariate_normal([0,0],s,100) # generate random 100 values with the std of this half hour and the correlation between ET and GPP
            synGPP[:,i] = daily_cycle[d,i]*mean_GPP[d]+Noise[:,0]    # synthetic gpp
            synET[:,i]  = daily_cycle[d,i]*mean_ET[d]+Noise[:,1]     # synthetic le

        # calculate the 100 artificial correlation coefficients for the day
        StN[:,d]=daily_corr(synGPP, synET, np.tile(daily_cycle[d],100).reshape(-1,nStepsPerDay),nStepsPerDay)

    # calculate the real correlation array
    pwc=daily_corr(ET,GPP*np.sqrt(VPD),Rg_pot,nStepsPerDay)

    # calculate the rank of the real array within the artificial dataset giving DWCI
    DWCI=(StN<np.tile(pwc,repeats).reshape(repeats,-1)).sum(axis=0)
    DWCI[np.isnan(StN).prod(axis=0).astype(bool)]=-9999

    return(DWCI)
