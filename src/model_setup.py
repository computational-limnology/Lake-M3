#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 17:48:21 2026

@author: emmamarchisin
"""


import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from ancillary_functions import calc_cc


def get_secview(secchifile):
    if secchifile is not None:
        secview0 = pd.read_csv(secchifile)
        secview0['sampledate'] = pd.to_datetime(secview0['sampledate'])
        secview = secview0.loc[secview0['sampledate'] >= startDate]
        if secview['sampledate'].min() >= startDate:
          firstVal = secview.loc[secview['sampledate'] == secview['sampledate'].min(), 'secnview'].values[0]
          firstRow = pd.DataFrame(data={'sampledate': [startDate], 'secnview':[firstVal]})
          secview = pd.concat([firstRow, secview], ignore_index=True)
      
          
        secview['dt'] = (secview['sampledate'] - secview['sampledate'][0]).astype('timedelta64[s]') + 1
        secview['kd'] = 1.7 / secview['secnview']
        secview['kd'] = secview.set_index('sampledate')['kd'].interpolate(method="linear").values
    else:
        secview = None
    
    return(secview)

def get_data_start(row):
    for column_name, value in row.items():
        try:
            if isinstance(float(value), float):  # checks if value is numeric
                return column_name
        except:
            continue
    return None
      
def get_num_data_columns(csv_path, row_name):
    # read CSV without forcing any index
    df = pd.read_csv(csv_path, index_col=0)

    if row_name not in df.index:
        raise ValueError(f"Row '{row_name}' not found in CSV index.")

    # get the row by name
    row = df.loc[row_name]

    # find the first numeric column in that row
    start_col = get_data_start(row)
    if start_col is None:
        return 0  # no numeric data found

    # count columns from the first numeric column to the end
    start_idx = df.columns.get_loc(start_col)
    num_columns = len(df.columns[start_idx:])

    return num_columns
    
#get a the configuration for a given row
def get_lake_config (lake_config_file, iteration = 1):
    df = pd.read_csv(lake_config_file, index_col = 0)
    first_column = get_data_start(df.loc["Zmax"])
    col_index = df.columns.get_loc(first_column)
    
    col = df.iloc[:, col_index + iteration - 1]
    def try_cast(x):
        try:
            return float(x) if '.' in str(x) or 'e' in str(x).lower() else int(x)
        except:
            return x

    col = col.apply(try_cast)
    return col

def get_model_params(model_params_file, iteration=1):
    df = pd.read_csv(model_params_file, index_col=0)
    df = df.replace('None', np.nan) 
    df.replace({np.nan: None})
    first_column = get_data_start(df.loc["km"])
    col_index = df.columns.get_loc(first_column)

    col = df.iloc[:, col_index + iteration - 1]

    def try_cast(x):
        
        
        
        try:
            return float(x) if '.' in str(x) or 'e' in str(x).lower() else int(x)
        except:
            return x

    return col.apply(try_cast)


def get_run_config (run_config_file, iteration = 1):
    df = pd.read_csv(run_config_file, index_col = 0)
    first_column = get_data_start(df.loc["nx"])
    col_index = df.columns.get_loc(first_column)
    col = df.iloc[:, col_index + iteration - 1]
    def try_cast(x):
        try:
            return float(x) if '.' in str(x) or 'e' in str(x).lower() else int(x)
        except:
            return x

    col = col.apply(try_cast)
    return col

def get_ice_and_snow (ice_and_snow_file, iteration = 1):
    df = pd.read_csv(ice_and_snow_file, index_col = 0)
    first_column = get_data_start(df.loc["Hi"])
    col_index = df.columns.get_loc(first_column)
    
    col = df.iloc[:, col_index + iteration - 1]
    def try_cast(x):
        try:
            return float(x) if '.' in str(x) or 'e' in str(x).lower() else int(x)
        except:
            return x

    col = col.apply(try_cast)
    return col

def get_hypsography(hypsofile, dx, nx, outflow_depth=None):
#def get_hypsography(hypsofile, dx, nx):
  hyps = pd.read_csv(hypsofile)
  out_depths = np.linspace(0, nx*dx, nx+1)
  area_fun = interp1d(hyps.Depth_meter.values, hyps.Area_meterSquared.values)
  area = area_fun(out_depths)
  area[-1] = area[-2] - 1 # TODO: confirm this is correct
  depth = np.linspace(0, nx*dx, nx+1)
  
  volume = area * 1000
  # volume = 0.5 * (area[:-1] + area[1:]) * np.diff(depth)
  # volume = (area[:-1] + area[1:]) * np.diff(depth)
  for d in range(0, (len(depth)-1)):
      volume[d] = np.abs(sum(area[0:(d+1)] * dx) - sum(area[0:d] * dx))

  # volume = (area[:-1] - area[1:]) * np.diff(depth)
  # volume = np.append(volume, 1000)
  
  volume = volume[:-1]
  depth = 1/2 * (depth[:-1] + depth[1:])
  area = 1/2 * (area[:-1] + area[1:])
  
  if outflow_depth is not None:
      mask=depth<=(outflow_depth+0.25) #depths off set by 0.25m, corrects for difference
      hypso_weight= np.zeros_like(volume)
      total_volume=np.sum(volume[mask])
      hypso_weight[mask]=volume[mask]/total_volume
  else: 
        total_volume=np.sum(volume)
        hypso_weight=volume/total_volume
  
  return([area, depth, volume, hypso_weight])

def provide_meteorology(meteofile, windfactor, lat, lon, elev, startDate, endDate):

    meteo = pd.read_csv(meteofile)
    daily_meteo = meteo

    daily_meteo['date'] = pd.to_datetime(daily_meteo['datetime'])

    daily_meteo = daily_meteo.loc[
    (daily_meteo['date'] >= startDate) & (daily_meteo['date']<= endDate)]
    
    
    daily_meteo['ditt'] = abs(daily_meteo['date'] - startDate)


    
    daily_meteo['Cloud_Cover'] = calc_cc(date = daily_meteo['date'],
                                                airt = daily_meteo['Air_Temperature_celsius'],
                                                relh = daily_meteo['Relative_Humidity_percent'],
                                                swr = daily_meteo['Shortwave_Radiation_Downwelling_wattPerMeterSquared'],
                                                lat =  lat, lon = lon,
                                                elev = elev)

    
    #daily_meteo['dt'] = (daily_meteo['date'] - daily_meteo['date'][0]).astype('timedelta64[s]') + 1
    # time_diff = daily_meteo['date'] - daily_meteo['date'].iloc[0]
    # daily_meteo['dt'] = time_diff.dt.total_seconds() + 1.0

    daily_meteo.reset_index(drop=True, inplace=True)

    time_diff = (daily_meteo['date'] -  daily_meteo['date'][0]).astype('timedelta64[s]') ##RL init cond change
    daily_meteo['dt'] =time_diff.dt.total_seconds() +1
    daily_meteo['ea'] = (daily_meteo['Relative_Humidity_percent'] * 
      (4.596 * np.exp((17.27*(daily_meteo['Air_Temperature_celsius'])) /
      (237.3 + (daily_meteo['Air_Temperature_celsius']) ))) / 100)
    daily_meteo['ea'] = ((101.325 * np.exp(13.3185 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15))) -
      1.976 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15)))**2 -
      0.6445 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15)))**3 -
      0.1229 * (1 - (373.15 / (daily_meteo['Air_Temperature_celsius'] + 273.15)))**4)) * daily_meteo['Relative_Humidity_percent']/100)
    daily_meteo['ea'] = (daily_meteo['Relative_Humidity_percent']/100) * 10**(9.28603523 - 2322.37885/(daily_meteo['Air_Temperature_celsius'] + 273.15))
    startDate = pd.to_datetime(daily_meteo.loc[0, 'date']) 
    
    ## calibration parameters
    daily_meteo['Shortwave_Radiation_Downwelling_wattPerMeterSquared'] = daily_meteo['Shortwave_Radiation_Downwelling_wattPerMeterSquared'] 
    daily_meteo['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'] = daily_meteo['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'] * windfactor # wind speed multiplier
    
    date_time = daily_meteo.date

    daily_meteo['day_of_year_list'] = [t.timetuple().tm_yday for t in date_time]
    daily_meteo['time_of_day_list'] = [t.hour for t in date_time]
    ## light
    # Package ID: knb-lter-ntl.31.30 Cataloging System:https://pasta.edirepository.org.
    # Data set title: North Temperate Lakes LTER: Secchi Disk Depth; Other Auxiliary Base Crew Sample Data 1981 - current.
    
    
    return(daily_meteo)

def provide_phosphorus(tpfile, startingDate, startTime):
    phos = pd.read_csv(tpfile)

    daily_tp = phos
    daily_tp['date'] = pd.to_datetime(daily_tp['datetime'])
    
    daily_tp['ditt'] = abs(daily_tp['date'] - startingDate)
    daily_tp = daily_tp.loc[daily_tp['date'] >= startingDate]
    if startingDate < daily_tp['date'].min():
        daily_tp.loc[-1] = [startingDate, 'epi', daily_tp['tp'].iloc[0], startingDate, daily_tp['ditt'].iloc[0]]  # adding a row
        daily_tp.index = daily_tp.index + 1  # shifting index
        daily_tp.sort_index(inplace=True) 
        #daily_tp['dt'] = (daily_tp['date'] - daily_tp['date'].iloc[0]).astype('timedelta64[s]') + startTime 
    # time_diff = daily_tp['date'] - daily_tp['date'].iloc[0]
    # daily_tp['dt'] = time_diff.dt.total_seconds() + startTime
    time_diff = (daily_tp['date'] - daily_tp['date'].iloc[0]).astype('timedelta64[s]') #RL init cond change
    daily_tp['dt'] =time_diff.dt.total_seconds() + startTime
    return(daily_tp)


def provide_carbon(ocloadfile, startingDate, startTime):

    # Read Daily OC load input file
    oc_load = pd.read_csv(ocloadfile)
    oc_load['datetime'] = pd.to_datetime(oc_load['datetime'])
    oc_load["date"] = oc_load["datetime"]
    full_range = pd.date_range(
        start = oc_load['date'].min(), 
        end = oc_load['date'].max(),
        freq = "H"
    )
    expanded = pd.DataFrame({"datetime": full_range})
    expanded = expanded.merge(oc_load, on = "datetime", how = "left")
    expanded = expanded.ffill()
    oc_load = expanded

    # Filter data starting from model start date
    daily_oc = oc_load.loc[
    (oc_load['date'] >= startingDate) ]
    daily_oc['ditt'] = abs(daily_oc['date'] - startingDate)

   

    # If startingDate precedes data, insert an initial row
    if startingDate < daily_oc['date'].min():
        first_row = {
            'datetime': startingDate,
            'discharge': oc_load['discharge'].iloc[0],
            'oc': oc_load['oc'].iloc[0],
            'date': startingDate,
            'ditt': (daily_oc['date'].iloc[0] - startingDate)
        }
        daily_oc = pd.concat([pd.DataFrame([first_row]), daily_oc], ignore_index=True)

    # Compute time offset from simulation start
    # print("Unique datetimes in daily_oc:")
    # print(daily_oc['date'].unique())
    # print("Shape:", daily_oc.shape)
    #daily_oc['dt'] = (daily_oc['date'] - daily_oc['date'].iloc[0]).dt.total_seconds() + startTime
    # time_diff = daily_oc['date'] - daily_oc['date'].iloc[0]
    # daily_oc['dt'] = time_diff.dt.total_seconds() + startTime
    time_diff = (daily_oc['datetime'] - daily_oc['datetime'][0]).astype('timedelta64[s]') ##RL init cond change
    daily_oc['dt'] =time_diff.dt.total_seconds() + 1
    #daily_oc['dt'] = (daily_oc['date'] - daily_oc['date'].iloc[0]).dt.total_seconds() + startTime
    #compute total carbon load as oc_mgl * discharge
    daily_oc['total_carbon'] = daily_oc['oc'] * daily_oc['discharge']
    daily_oc['hourly_carbon']=daily_oc['total_carbon']/24


    #fill in hourly times

    return daily_oc

def initial_profile(initfile, nx, dx, depth, startDate):
  #meteo = processed_meteo
  #startDate = meteo['date'].min()
  obs = pd.read_csv(initfile)
  obs['datetime'] = pd.to_datetime(obs['datetime'])
  obs['ditt'] = abs(obs['datetime'] - startDate)
  init_df = obs.loc[obs['ditt'] == obs['ditt'].min()]
  if max(depth) > init_df.Depth_meter.max():
    lastRow = init_df.loc[init_df.Depth_meter == init_df.Depth_meter.max()]
    init_df = pd.concat([init_df, lastRow], ignore_index=True)
    init_df.loc[init_df.index[-1], 'Depth_meter'] = max(depth)
  print("Selected initial profile date:", init_df['datetime'].iloc[0])
  print("Max depth in profile:", init_df['Depth_meter'].max())
  print("Lake max depth:", max(depth))

  profile_fun = interp1d(init_df.Depth_meter.values, init_df.Water_Temperature_celsius.values)
  out_depths = depth # these aren't actually at the 0, 1, 2, ... values, actually increment by 1.0412; make sure okay
  u = profile_fun(out_depths)
  
  # TODO implement warning about profile vs. met start date
  
  return(u)

def wq_initial_profile(initfile, nx, dx, depth, volume, startDate):
  #meteo = processed_meteo
  #startDate = meteo['date'].min()
  obs = pd.read_csv(initfile)
  obs['datetime'] = pd.to_datetime(obs['datetime'])
  
  do_obs = obs.loc[obs['variable'] == 'do']
  do_obs['ditt'] = abs(do_obs['datetime'] - startDate)
  init_df = do_obs.loc[do_obs['ditt'] == do_obs['ditt'].min()]
  if max(depth) > init_df.depth.max():
    lastRow = init_df.loc[init_df.depth == init_df.depth.max()]
    init_df = pd.concat([init_df, lastRow], ignore_index=True)
    init_df.loc[init_df.index[-1], 'depth'] = max(depth)
    
  profile_fun = interp1d(init_df.depth.values, init_df.observation.values)
  out_depths =depth# these aren't actually at the 0, 1, 2, ... values, actually increment by 1.0412; make sure okay
  do = profile_fun(out_depths)
  
  doc_obs = obs.loc[obs['variable'] == 'doc']
  doc_obs['ditt'] = abs(doc_obs['datetime'] - startDate)
  init_df = doc_obs.loc[doc_obs['ditt'] == doc_obs['ditt'].min()]
  # if max(depth) > init_df.depth.max():
  #   lastRow = init_df.loc[init_df.depth == init_df.depth.max()]
  #   init_df = pd.concat([init_df, lastRow], ignore_index=True)
  #   init_df.loc[init_df.index[-1], 'depth'] = max(depth)
    
  if init_df.depth.min()>0: #assumed mixed epi, if no 0m available then it pulls from shallowest option
    shallowest = init_df.loc[init_df.depth == init_df.depth.min()].copy()
    shallowest.loc[:, 'depth'] = 0.0  # set depth to 0
    init_df = pd.concat([shallowest, init_df], ignore_index=True)  
  if max(depth) > init_df.depth.max():
    lastRow = init_df.loc[init_df.depth == init_df.depth.max()]
    init_df = pd.concat([init_df, lastRow], ignore_index=True)
    init_df.loc[init_df.index[-1], 'depth'] = max(depth)
  profile_fun = interp1d(init_df.depth.values, init_df.observation.values)
  out_depths = depth# these aren't actually at the 0, 1, 2, ... values, actually increment by 1.0412; make sure okay
  doc = profile_fun(out_depths)
  
  u = np.vstack((do * volume, doc * volume))
  
  #print(u)
  # TODO implement warning about profile vs. met start date
  
  return(u)
