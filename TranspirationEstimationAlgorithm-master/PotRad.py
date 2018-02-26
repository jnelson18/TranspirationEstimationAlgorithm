# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:04:22 2017

@author: jnelson
"""
import numpy as np
import warnings
from datetime import date

from datetime import datetime, time, timedelta
from math import pi, cos, sin

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++ Solar radiation properties
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def timeconv(val):
    '''timeconv(val)

    Converts timestamp as YYYYMMDD.decimaltime into day of year (DoY)
    and decimal hour
 
    Parameters
    ----------
    val : float
        timestamp as YYYYMMDD.decimaltime

    Returns
    -------
    DoY : int
        day of year
    decHour : float
        decimal hour of day  
    '''
    year=int(str(val)[:4])
    month=int(str(val)[4:6])
    day=int(str(val)[6:8])
    decHour=float(str(val)[8:])*24
    DoY=date(year,month,day).timetuple().tm_yday
    return(DoY,decHour)

def solar_time(dt, longit):
    gamma = 2 * pi / 365 * (dt.timetuple().tm_yday - 1 + float(dt.hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * cos(gamma) - 0.032077 * sin(gamma) \
             - 0.014615 * cos(2 * gamma) - 0.040849 * sin(2 * gamma))
    decl = 0.006918 - 0.399912 * cos(gamma) + 0.070257 * sin(gamma) \
           - 0.006758 * cos(2 * gamma) + 0.000907 * sin(2 * gamma) \
           - 0.002697 * cos(3 * gamma) + 0.00148 * sin(3 * gamma)
    time_offset = eqtime + 4 * longit
    tst = dt.hour * 60 + dt.minute + dt.second / 60 + time_offset
    solar_time = datetime.combine(dt.date(), time(0)) + timedelta(minutes=tst)
    return(solar_time)

def CalcSunPosition(DoY,Hour,Lat_deg,Long_deg,TimeZone_h,useSolartime=True):
    '''CalcSunPosition(DoY,Hour,Lat_deg,Long_deg,TimeZone_h,useSolartime=True)
    
    Calculate the position of the sun.
    
    Calculate the position of the sun for use in calculating potential radiation.
    useSolartime by default corrects hour (given in local winter time) for
    latitude to solar time where noon is exactly at 12:00.
    

    Parameters
    ----------
    DoY : list or list like
        Data vector with day of year (DoY)
    Hour : list or list like
        Data vector with time as decimal hour
    Lat_deg : float
        Latitude in (decimal) degrees
    Long_deg : float
        Longitude in (decimal) degrees
    TimeZone_h : float
        Time zone (in hours)
    useSolartime : bool
        Set this to FALSE to compare to code that uses local winter time

    Returns
    -------
    SolTime_h : numpy array
        Solar time (SolTime, hours)
    SolDecl_rad : numpy array
        Solar declination (SolDecl, rad)
    SolElev_rad : numpy array
        Solar elevation with 0 at horizon (SolElev, rad)
    SolAzim_rad : numpy array
        Solar azimuth with 0 at North (SolAzim, rad)
    '''

    DoY = np.asanyarray(DoY)
    Hour = np.asanyarray(Hour)

    # Fractional year in radians
    FracYear_rad = 2 * np.pi * (DoY-1) / 365.24
    
    # Equation of time in hours, accounting for changes in the time of solar noon
    EqTime_h = ( 0.0072*np.cos(FracYear_rad) - 0.0528*np.cos(2*FracYear_rad)
                    - 0.0012*np.cos(3*FracYear_rad) - 0.1229*np.sin(FracYear_rad) 
                    - 0.1565*np.sin(2*FracYear_rad) - 0.0041*np.sin(3*FracYear_rad) )
    
    # Local time in hours
    LocTime_h = (Long_deg/15 - TimeZone_h)
    
    ##details<< 
    ## This code assumes that Hour is given in local winter time zone, and corrects it by longitude to 
    ## solar time (where noon is exactly at 12:00).
    ## Note: This is different form reference PVWave-code, 
    ## that does not account for solar time and uses winter time zone. 
    ## Set argument \code{useSolartime.b} to FALSE to use the local winter time instead.
    
    # Solar time
    # Correction for local time and equation of time
    if useSolartime:
        # Correction for local time and equation of time
        SolTime_h = Hour + LocTime_h + EqTime_h
    else:
        #! Note: For reproducing values close to Fluxnet Rg_pot which is without local time and eq of time correction
        #! (CEIP is even different)
        warnings.warn('Solar position calculated without correction for local time and equation of time.', RuntimeWarning)
        SolTime_h = Hour
    
    # Conversion to radians
    SolTime_rad = (SolTime_h - 12) * np.pi / 12.0
    # Correction for solar time < -pi to positive, important for SolAzim_rad below
    SolTime_rad[(SolTime_rad < -np.pi)]=SolTime_rad[(SolTime_rad < -np.pi)]+2*np.pi
    
    #Solar declination in radians, accounting for the earth axis tilt
    SolDecl_rad = ( (0.33281-22.984*np.cos(FracYear_rad) - 0.34990*np.cos(2*FracYear_rad)
                    - 0.13980*np.cos(3*FracYear_rad) + 3.7872*np.sin(FracYear_rad)
                    + 0.03205*np.sin(2*FracYear_rad) + 0.07187*np.sin(3*FracYear_rad))/180*np.pi )
    
    # Solar elevation (vertical, zenithal angle) in radians with zero for horizon
    SolElev_rad =  np.arcsin(np.sin(SolDecl_rad) * np.sin(Lat_deg/180*np.pi)
                               + np.cos(SolDecl_rad) * np.cos(Lat_deg/180*np.pi)
                               * np.cos(SolTime_rad))
    
    # Solar azimuth (horizontal angle) with zero for North
    SolAzim_cos = ( ( np.cos(SolDecl_rad) * np.cos(SolTime_rad) - np.sin(SolElev_rad)
                    * np.cos(Lat_deg/180*np.pi) ) / ( np.sin(Lat_deg/180*np.pi) * np.cos(SolElev_rad) ) )
    # Correction if off edge values
    SolAzim_cos[(SolAzim_cos > +1)] = 1
    SolAzim_cos[(SolAzim_cos < -1)] = 1
    # Conversion to radians
    SolAzim_rad = np.arccos(SolAzim_cos)
    # Determine if solar azimuth is East or West depending on solar time
    SolAzim_rad[(SolTime_rad < 0)] = np.pi - SolAzim_rad[(SolTime_rad < 0)]
    SolAzim_rad[(SolTime_rad >= 0)] = np.pi + SolAzim_rad[(SolTime_rad >= 0)]
    
    return(SolTime_h,SolDecl_rad,SolElev_rad,SolAzim_rad)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def CalcExtRadiation(DoY):
    '''CalcExtRadiation(DoY)
    
    Calculate the extraterrestrial solar radiation.
    
    Calculate the extraterrestrial solar radiation with the eccentricity correction
    after Lanini, 2010 (Master thesis, Bern University).
    

    Parameters
    ----------
    DoY : list or list like
        Data vector with day of year (DoY)

    Returns
    -------
    ExtRadiation : numpy array
        Extraterrestrial radiation (ExtRad, W_m-2)
    '''  
    # Fractional year in radians
    FracYear_rad = 2*np.pi*(DoY-1) /365.24
    # Total solar irradiance
    SolarIrr_Wm2 = 1366.1 #W/m-2
    #Eccentricity correction
    ExtRadiation = SolarIrr_Wm2 * (1.00011 + 0.034221*np.cos(FracYear_rad) + 0.00128*np.sin(FracYear_rad)
                                          + 0.000719*np.cos(2*FracYear_rad) + 0.000077*np.sin(2*FracYear_rad))
    return(ExtRadiation)


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def CalcPotRadiation(DoY,Hour,Lat_deg,Long_deg,TimeZone_h,useSolartime=True):
    '''CalcSunPosition(DoY,Hour,Lat_deg,Long_deg,TimeZone_h,useSolartime=True)
    
    Calculate the potential radiation.
    
    Calculate potential radiation from solar elevation and extraterrestrial
    solar radiation. useSolartime by default corrects hour (given in local winter time) for
    latitude to solar time where noon is exactly at 12:00.
    

    Parameters
    ----------
    DoY : list or list like
        Data vector with day of year (DoY)
    Hour : list or list like
        Data vector with time as decimal hour
    Lat_deg : float
        Latitude in (decimal) degrees
    Long_deg : float
        Longitude in (decimal) degrees
    TimeZone_h : float
        Time zone (in hours)
    useSolartime : bool
        Set this to FALSE to compare to code that uses local winter time

    Returns
    -------
    PotRadiation : numpy array
        Potential radiation (PotRad, W_m-2)
    '''
    # Calculate potential radiation from solar elevation and extraterrestrial solar radiation
    SolTime_h,SolDecl_rad,SolElev_rad,SolAzim_rad = CalcSunPosition(DoY, Hour, Lat_deg, Long_deg, TimeZone_h, useSolartime=useSolartime)
    ExtRadiation = CalcExtRadiation(DoY)
    PotRadiation = np.zeros(DoY.shape)
    PotRadiation[(SolElev_rad > 0)] = ExtRadiation[(SolElev_rad > 0)] * np.sin(SolElev_rad[(SolElev_rad > 0)])
    return(PotRadiation)

