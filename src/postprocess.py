#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 12 20:25:44 2026

@author: emmamarchisin
"""

# #To do:
# -load in observations
# -observed in doc plots
# -reformat plots (legends)/ colors
# -check integrated gpp and r for volume, maybe no *dx
# - depth labels
#-test multiple columns


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates



def is_on(config, lake_key, key):
    return str(config.loc[key, lake_key]).lower() == "yes"

def get_value(config, lake_key, key):
    return config.loc[key, lake_key]


def depth_to_index(depth_array, target_depth):
    depth_array = np.asarray(depth_array)
    return np.abs(depth_array - target_depth).argmin()


def save_fig(fig, output_dir, lake_key, name):
    fig.tight_layout()
    fig.savefig(output_dir / f"{lake_key}_{name}.png", dpi=300)
    plt.close(fig)
    
def load_observations(observations_dir, config, lake_key, startDate, endDate):
    obs_file = str(get_value(config, lake_key, "obs_file"))

    if obs_file.lower() == "no":
        return None
    obs_path = observations_dir / obs_file
    if not obs_path.exists():
        print(f"Warning: observation file not found: {obs_path}")
        return None
    df = pd.read_csv(obs_path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    df = df[(df["datetime"] >= startDate) & (df["datetime"] <= endDate)]
    return df

def calc_dens(wtemp):
    dens = (999.842594 + (6.793952 * 1e-2 * wtemp) - (9.095290 * 1e-3 *wtemp**2) +
      (1.001685 * 1e-4 * wtemp**3) - (1.120083 * 1e-6* wtemp**4) + 
      (6.536336 * 1e-9 * wtemp**5))
    return dens

def melt_var(var, times, depth, name):
    df = pd.DataFrame(var.T, index=times)
    df["datetime"] = times

    df_long = df.melt(
        id_vars="datetime",
        var_name="depth_index",
        value_name=name
    )

    df_long["depth"] = df_long["depth_index"].apply(lambda i: depth[int(i)])
    df_long = df_long.drop(columns=["depth_index"])

    return df_long

def get_plot_depths(config, lake_key, varname):
    surf_key=f"surf_depth_{varname}"
    deep_key=f"deep_depth_{varname}"
    
    if surf_key not in config.index:
        raise KeyError(f"Missing config key: {surf_key}")

    if deep_key not in config.index:
        raise KeyError(f"Missing config key: {deep_key}")

    surf_depth = float(config.loc[surf_key, lake_key])
    deep_depth = float(config.loc[deep_key, lake_key])

    return surf_depth, deep_depth
    
    
def post_process(
    res,
    times,
    volume,
    area,
    dx,
    lake_output_dir,
    postprocess_config,
    lake_key,
    driver_dir,
    observations_dir,
    startDate,
    depth,
    endDate,
    meteo_all):
    
    temp = res["temp"]
    o2 = res["o2"]
    docl = res["docl"]
    docr = res["docr"]
    pocl = res["pocl"]
    pocr = res["pocr"]
    npp = res["npp"]
    atm_flux = res["atm_flux_output"]
    docl_resp = res["docl_respiration"]
    docr_resp = res["docr_respiration"]
    poc_resp = res["poc_respiration"]
    secchi = res["secchi"]
    
    print(f"=======Post Process {lake_key}=======")

    doc_total = docl + docr
    poc_total = pocl + pocr
    
    # surf_depth = float(get_value(postprocess_config, lake_key, "surf_depth")) #***
    # deep_depth = float(get_value(postprocess_config, lake_key, "deep_depth"))


    # surf_ix = depth_to_index(depth, surf_depth)
    # deep_ix = depth_to_index(depth, deep_depth)

    df_obs = load_observations(
        observations_dir, postprocess_config, lake_key,
        startDate, endDate)
    
    def heatmap_plot(data, temp, name=None, ax=None, vmin=None, vmax=None):

        created_fig = False
    
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 5))
            created_fig = True
        else:
            fig = ax.figure
    
        sns.heatmap(
            data,
            cmap=plt.cm.get_cmap('Spectral_r'),
            xticklabels=1000,
            yticklabels=2,
            ax=ax,
            vmin=vmin,
            vmax=vmax
        )
    
        dens = calc_dens(temp)
        ax.contour(
            np.arange(.5, temp.shape[1]),
            np.arange(.5, temp.shape[0]),
            dens,
            levels=[999],
            colors='black',
            linestyles='dotted'
        )
    
        ax.set_ylabel("Depth (m)", fontsize=15)
    
        
        step = 192   # days between labels
        xticks_ix = np.arange(0, len(times), step)
        ax.set_xticks(xticks_ix+0.5)
        ax.set_xticklabels(
            times[xticks_ix].strftime("%m/%d/%Y"),
            rotation=45,
            ha='right'
        )
    
        n_depth = data.shape[0]
        step = int(round(1 / dx))
    
        tick_positions = np.arange(0, n_depth, step)
        tick_labels = np.round(tick_positions * dx, 0)
    
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, rotation=0)
    
        # Save only if standalone plot
        if created_fig and name is not None:
            plt.tight_layout()
            save_fig(fig, lake_output_dir, lake_key, f"{name}.png")
            plt.close(fig)
    
        return ax


#O2

    if is_on(postprocess_config, lake_key, "o2_heat"):
        heatmap_plot(o2 / volume[:, None], res["temp"], "o2_heat", vmin=0, vmax=20)

    if is_on(postprocess_config, lake_key, "o2_line"):
        #Get depth for graphing
        surf_depth, deep_depth = get_plot_depths(postprocess_config, lake_key, "do")
        surf_ix = depth_to_index(depth, surf_depth)
        deep_ix = depth_to_index(depth, deep_depth)
        
        #Plotting
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(times, o2[surf_ix,:]/volume[surf_ix],color='blue',linestyle='solid',label=f"{int(surf_depth)}m Modeled DO")
        ax.plot(times, o2[deep_ix,:]/volume[deep_ix],color='blue',linestyle='dashed',label=f"{int(deep_depth)}m Modeled DO")

        if df_obs is not None:
            df_obs_surf = df_obs[(df_obs["variable"]=="do") &
                          (df_obs["depth"]==surf_depth)]
            df_obs_deep = df_obs[(df_obs["variable"]=="do") &
                          (df_obs["depth"]==deep_depth)]
            if not df_obs_surf.empty:
                ax.plot(
                    df_obs_surf["datetime"],
                    df_obs_surf["observation"],
                    color='red',
                    linestyle='solid',
                    marker='o',
                    zorder=5,
                    label=f"{int(surf_depth)}m Observed DO")

            if not df_obs_deep.empty:
                ax.plot(
                    df_obs_deep["datetime"],
                    df_obs_deep["observation"],
                    color='red',
                    linestyle='dashed',
                    marker='o',
                    zorder=5,
                    label=f"{int(deep_depth)}m Observed DO")

        ax.set_ylabel("DO (mg/L)", fontsize=15)
        ax.set_xlabel("Time", fontsize=15)
        ax.legend(loc="best")
        plt.tight_layout()
        save_fig(fig,lake_output_dir, lake_key, "o2_line")
        plt.close(fig)

#Water Temp

    if is_on(postprocess_config, lake_key, "wtemp_heat"):
        heatmap_plot(res["temp"], res["temp"], "wtemp_heat", vmin=0, vmax=30)
    
    # if is_on(postprocess_config, lake_key, "wtemp_line"):
    #     fig, ax = plt.subplots(figsize=(10,5))
    #     ax.plot(times, temp[surf_ix,:])
    #     ax.plot(times, temp[deep_ix,:], linestyle="--")
    #     ax.set_ylabel("Temp (°C)")
    #     save_fig(fig, lake_output_dir, lake_key, "wtemp_line")
        
    if is_on(postprocess_config, lake_key, "wtemp_line"):
        
        #Get depth for plotting
        surf_depth, deep_depth = get_plot_depths(postprocess_config, lake_key, "wtemp")
        surf_ix = depth_to_index(depth, surf_depth)
        deep_ix = depth_to_index(depth, deep_depth)
        
        #Plotting
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(times,  temp[surf_ix,:],color='blue',linestyle='solid',label=f"{int(surf_depth)}m Modeled Temp")
        ax.plot(times,  temp[deep_ix,:],color='blue',linestyle='dashed',label=f"{int(deep_depth)}m Modeled Temp")

        if df_obs is not None:
            df_obs_surf = df_obs[(df_obs["variable"]=="wtemp") &
                          (df_obs["depth"]==surf_depth)]
            df_obs_deep = df_obs[(df_obs["variable"]=="wtemp") &
                          (df_obs["depth"]==deep_depth)]
            if not df_obs_surf.empty:
                ax.plot(
                    df_obs_surf["datetime"],
                    df_obs_surf["observation"],
                    color='red',
                    linestyle='solid',
                    marker='o',
                    zorder=5,
                    label=f"{int(surf_depth)}m Observed Temp")

            if not df_obs_deep.empty:
                ax.plot(
                    df_obs_deep["datetime"],
                    df_obs_deep["observation"],
                    color='red',
                    linestyle='dashed',
                    marker='o',
                    zorder=5,
                    label=f"{int(deep_depth)}m Observed Temp")

        ax.set_ylabel("Temp (°C)", fontsize=15)
        ax.set_xlabel("Time", fontsize=15)
        ax.legend(loc="best")
        plt.tight_layout()
        save_fig(fig,lake_output_dir, lake_key, "wtemp_line")
        plt.close(fig)

#DOC and POC

    variables = {
        "docr": docr,
        "docl": docl,
        "doctot": doc_total,
        "pocr": pocr,
        "pocl": pocl,
        "poctot": poc_total
    }
    
    depth_group={
        'docl':'doctot',
        'docr':'doctot',
        'doctot':'doctot',
        'pocl':'poctot',
        'pocr':'poctot',
        'poctot':'poctot'}

    for varname, var in variables.items():

        if is_on(postprocess_config, lake_key, f"{varname}_heat"):
            heatmap_plot(var/volume[:,None],res["temp"], f"{varname}_heat", vmin=0, vmax=5)

        if is_on(postprocess_config, lake_key, f"{varname}_line"):
            
            #Get depths for plotting
            group_name=depth_group[varname]
            surf_depth, deep_depth = get_plot_depths(postprocess_config, lake_key, group_name)
            surf_ix = depth_to_index(depth, surf_depth)
            deep_ix = depth_to_index(depth, deep_depth)
            
            #plotting
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(
                times,
                var[surf_ix, :] / volume[surf_ix],
                color="blue",
                linestyle="solid",
                label=f"{int(surf_depth)}m Modeled")
    
            ax.plot(
                times,
                var[deep_ix, :] / volume[deep_ix],
                color="blue",
                linestyle="dashed",
                label=f"{int(deep_depth)}m Modeled")

            if df_obs is not None and varname in ["doctot","poctot"]:
                    obs_var='doc' if varname =='doctot' else "poc"
                    
                    df_obs_surf = df_obs[
                        (df_obs["variable"] == obs_var) &
                        (df_obs["depth"] == surf_depth)]
        
                    df_obs_deep = df_obs[
                        (df_obs["variable"] == obs_var) &
                        (df_obs["depth"] == deep_depth)]
        
                    if not df_obs_surf.empty:
                        ax.plot(
                            df_obs_surf["datetime"],
                            df_obs_surf["observation"],
                            color="red",
                            linestyle="solid",
                            marker="o",
                            zorder=5,
                            label=f"{int(surf_depth)}m Observed")
        
                    if not df_obs_deep.empty:
                        ax.plot(
                            df_obs_deep["datetime"],
                            df_obs_deep["observation"],
                            color="red",
                            linestyle="dashed",
                            marker="o",
                            zorder=5,
                            label=f"{int(deep_depth)}m Observed")
            ax.set_ylabel(f"{varname} (mg/L)")
            plt.ylim(0, 8)
            ax.set_xlabel("Time")
            ax.legend(loc="best")
            save_fig(fig, lake_output_dir, lake_key, f"{varname}_line")

#Secchi
    if is_on(postprocess_config, lake_key, "secchi"):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(times, secchi.T, color='blue', label='Modeled')

        if df_obs is not None:
            df_sec = df_obs[df_obs["variable"]=="secchi"]
            ax.scatter(df_sec["datetime"], df_sec["observation"], color="red", label='Observed')
        ax.invert_yaxis()
        ax.set_ylabel("Secchi (m)")
        save_fig(fig, lake_output_dir, lake_key, "secchi")

#Met data
    if is_on(postprocess_config, lake_key, "met_line"):

        fig, axis = plt.subplots(4, 1, figsize=(12,6), sharex=True)
    
        # Light
        axis[0].plot(times,meteo_all['Shortwave_Radiation_Downwelling_wattPerMeterSquared'],color='goldenrod')
        axis[0].set_ylabel("SW Radiation\n(W m$^{-2}$)")
        axis[0].set_title("Meteorological Drivers")
    
        # Wind
        axis[1].plot( times,meteo_all['Ten_Meter_Elevation_Wind_Speed_meterPerSecond'],color='steelblue')
        axis[1].set_ylabel("Wind Speed\n(m s$^{-1}$)")
    
        # Precip
        axis[2].plot(times,meteo_all['Precipitation_millimeterPerDay'],color='forestgreen')
        axis[2].set_ylabel("Precip\n(mm d$^{-1}$)")
    
        # Air Temp
        axis[3].plot(times,meteo_all['Air_Temperature_celsius'],color='firebrick')
        axis[3].set_ylabel("Air Temp\n(°C)")
        axis[3].set_xlabel("Time")
    
        plt.tight_layout()
    
        save_fig(fig, lake_output_dir, lake_key, "meteo_panel")

#Rates of GPP, R, AtmEx
    if is_on(postprocess_config, lake_key, "rates_panel"):

        r_all = (
            (docl * docl_resp) +
            (docr * docr_resp) +
            (pocl * poc_resp) +
            (pocr * poc_resp)
        ) / volume[:,None]

        gpp_all = npp/volume[:,None] + r_all

        r = r_all[surf_ix,:]
        gpp = gpp_all[surf_ix,:]
        atm = atm_flux[0,:] / volume[0]

        fig, ax = plt.subplots(3,1, figsize=(10,8), sharex=True)
        ax[0].plot(times, gpp, color='green')
        ax[0].set_ylabel("GPP (g/m3/d)")
        ax[1].plot(times, r, color='red')
        ax[1].set_ylabel("R (g/m3/d)")
        ax[2].plot(times, atm, color='purple')
        ax[2].set_ylabel("Atm Ex (g/m3/d)")

        save_fig(fig, lake_output_dir, lake_key, "rates_panel")

#Integrated GPP and R
    if is_on(postprocess_config, lake_key, "integrated_gpp") \
       or is_on(postprocess_config, lake_key, "integrated_r")\
       or is_on(postprocess_config, lake_key, "integrated_nep"):

        r_layer = (
            (docl * docl_resp) +
            (docr * docr_resp) +
            (pocl * poc_resp) +
            (pocr * poc_resp)) #g/d per layer
        r_layer_m3=r_layer/volume[:,None]
        
        gpp_layer=npp #g/d per layer
        gpp_layer_m3=gpp_layer/volume[:,None]
        nep_layer=gpp_layer-r_layer
        nep_layer_m3=(nep_layer)/volume[:,None]
        
        total_r=np.sum(r_layer,axis=0) #g/d lake
        total_gpp=np.sum(gpp_layer,axis=0) #g/d lake
        total_nep=np.sum(nep_layer,axis=0)

        lake_area=np.max(area) #m2

        integrated_gpp = total_gpp/lake_area #g/m2/d
        integrated_r = total_r/lake_area #g/m2/d
        integrated_nep = total_nep / lake_area       # g m⁻² d⁻¹ water column
        
        def plot_integrated(layer_data, integrated_series, ylabel, fname,vmin=None,vmax=None):

            fig, ax = plt.subplots(
                2, 1,
                figsize=(12, 6),
                gridspec_kw={"height_ratios": [3, 1]})
    
            # Heatmap
            heatmap_plot(layer_data, res["temp"], ax=ax[0], vmin=0,vmax=5)
            ax[0].set_title(ylabel)
    
            # Line
            ax[1].plot(times, integrated_series)
            ax[1].set_ylabel(ylabel)
            ax[1].set_xlabel("Time")
            ax[1].xaxis.set_major_formatter(mdates.DateFormatter("%m/%d/%Y"))
            fig.autofmt_xdate() 
            ax[1].set_ylim(0, 10)

    
            plt.tight_layout()
            save_fig(fig, lake_output_dir, lake_key, fname)
            plt.close(fig)

        if is_on(postprocess_config, lake_key, "integrated_gpp"):
            plot_integrated(
                gpp_layer_m3,
                integrated_gpp,
                "Integrated GPP (g m⁻3 d⁻¹)",
                "integrated_gpp"
            )

        if is_on(postprocess_config, lake_key, "integrated_r"):
            plot_integrated(
                r_layer_m3,
                integrated_r,
                "Integrated R (g m⁻3 d⁻¹)",
                "integrated_r"
            )
            
        if is_on(postprocess_config, lake_key, "integrated_nep"):
            v = np.nanmax(np.abs(nep_layer_m3))
            plot_integrated(
                nep_layer_m3,
                integrated_nep,
                "Integrated NEP (g m⁻3 d⁻¹)",
                "integrated_nep",
                vmin=-v,
                vmax=v
            )

#Driver_panel graphs

    #Get driver variables at 1m
    target_depth = 1.0
    ix_1m = depth_to_index(depth, target_depth)
    temp_1m = temp[ix_1m, :] 
    npp_1m = npp[ix_1m, :] / volume[ix_1m]
    do_1m  = o2[ix_1m, :] / volume[ix_1m]
    doc_1m = doc_total[ix_1m, :] / volume[ix_1m]
    poc_1m = poc_total[ix_1m, :] / volume[ix_1m]
    light = meteo_all['Shortwave_Radiation_Downwelling_wattPerMeterSquared']   
    tp = res.get("TP", np.zeros_like(light)).flatten()
    tp=tp*1000 # convert mg/L -> ug/L
    
    print("light:", light.shape)
    print("temp:", temp_1m.shape)
    print("tp:", tp.shape)
    print("npp:", npp_1m.shape)
    print("TPm keys check:", "TP" in res)

    if "TPm" in res:
        print("TPm stats:", np.nanmin(res["TPm"]), np.nanmax(res["TPm"]))
        print("TPm shape:", np.shape(res["TPm"]))
    
    def plot_driver_panels(light, temp, tp, response, ylabel, fname):

        fig, ax = plt.subplots(3, 1, figsize=(6, 10))   
        ax[0].scatter(light, response, alpha=0.4)
        ax[0].set_xlabel("Shortwave (W/m2)")
        ax[0].set_ylabel(ylabel)    
        ax[1].scatter(temp, response, alpha=0.4)
        ax[1].set_xlabel("Temperature (°C)")
        ax[1].set_ylabel(ylabel)    
        ax[2].scatter(tp, response, alpha=0.4)
        ax[2].set_xlabel("TP (ug/L)")
        ax[2].set_ylabel(ylabel)    
        plt.tight_layout()
        save_fig(fig, lake_output_dir, lake_key, fname)
        
        
    if is_on(postprocess_config, lake_key, 'npp_driver_panels'):
        plot_driver_panels(light, temp_1m, tp, npp_1m, "NPP", "npp_driver_panels")
    
    if is_on(postprocess_config, lake_key, 'do_driver_panels'):
        plot_driver_panels(light, temp_1m, tp, do_1m, "DO (mg/L)", "do_driver_panels")
        
    if is_on(postprocess_config, lake_key, 'doc_driver_panels'):
        plot_driver_panels(light, temp_1m, tp, doc_1m, "DOC (mg/L)", "doc_driver_panels")
    
    if is_on(postprocess_config, lake_key, 'poc_driver_panels'):
        plot_driver_panels(light, temp_1m, tp, poc_1m, "POC (mg/L)", "poc_driver_panels")


#FM_ Lake files


    if (is_on(postprocess_config, lake_key, "fm_lake_hourly") or
        is_on(postprocess_config, lake_key, "fm_lake_daily")):
    
        def melt_var(arr_2d, datetimes, depth, varname):
            arr_2d = np.asarray(arr_2d)
            if arr_2d.shape[0] == len(datetimes) and arr_2d.shape[1] == len(depth):
                arr_2d = arr_2d.T
            n_depths, n_times = arr_2d.shape
            return pd.DataFrame({
                "datetime": np.repeat(datetimes, n_depths),
                "depth": np.tile(depth, n_times),
                varname: arr_2d.flatten()})
    
        temp = res["temp"]
        o2 = res["o2"] / volume[:, None]
        doc = (res["docl"] + res["docr"]) / volume[:, None]
        poc = (res["pocl"] + res["pocr"]) / volume[:, None]
    
        dfs = [
            melt_var(temp, times, depth, "WaterTemp_C"),
            melt_var(o2, times, depth, "Water_DO_mg_per_L"),
            melt_var(doc, times, depth, "Water_DOC_mg_per_L"),
            melt_var(poc, times, depth, "Water_POC_mg_per_L"),]
    
        fm_lake = dfs[0]
        for df in dfs[1:]:
            fm_lake = fm_lake.merge(df, on=["datetime", "depth"], how="left")
        
        #hourly
        if is_on(postprocess_config, lake_key, "fm_lake_hourly"):
            fm_lake.to_csv(lake_output_dir / f"{lake_key}_fm_lake_hourly.csv",index=False)

        #daily
        if is_on(postprocess_config, lake_key, "fm_lake_daily"):
            fm_lake["Date"] = pd.to_datetime(fm_lake["datetime"]).dt.floor("D")
            fm_lake_daily = (fm_lake.groupby(["Date", "depth"], as_index=False).mean(numeric_only=True))
            fm_lake_daily.to_csv(lake_output_dir / f"{lake_key}_fm_lake_daily.csv",index=False)

#Lake FM Driver files

    if (is_on(postprocess_config, lake_key, "fm_driver_hourly") or
        is_on(postprocess_config, lake_key, "fm_driver_daily")):
    
        meteo = res["meteo_input"]
        secchi = res["secchi"]
        TPm = res.get("TPm", np.zeros_like(secchi))
    
        fm_driver = pd.DataFrame({
            "datetime": times,
            "Shortwave_Wm2": meteo[4, :],
            "sum_Longwave_Radiation_Downwelling_wattPerMeterSquared": meteo[1, :],
            "AirTemp_C": meteo[0, :],
            "median_Ten_Meter_Elevation_Wind_Speed_meterPerSecond": meteo[12, :],
            "sum_Precipitation_millimeterPerDay": meteo[15, :],
            "Water_Secchi_m": secchi.flatten(),
            "TP_load_ug_per_L": TPm.flatten(),})
    
        #hourly
        if is_on(postprocess_config, lake_key, "fm_driver_hourly"):
            fm_driver.to_csv(lake_output_dir / f"{lake_key}_fm_driver_hourly.csv",index=False)
    
        #daily
        if is_on(postprocess_config, lake_key, "fm_driver_daily"):
            fm_driver["Date"] = fm_driver["datetime"].dt.floor("D")
            sum_vars = [
                "sum_Longwave_Radiation_Downwelling_wattPerMeterSquared",
                "sum_Precipitation_millimeterPerDay",
                "TP_load_ug_per_L"]
                # "TOC_load_g_per_d",
                # "Discharge_m3_per_d"]
    
            median_vars = [
                "Shortwave_Wm2",
                "AirTemp_C",
                "median_Ten_Meter_Elevation_Wind_Speed_meterPerSecond",
                "Water_Secchi_m"]
    
            fm_driver_daily = (fm_driver.groupby("Date")
                .agg({
                        **{v: "sum" for v in sum_vars if v in fm_driver.columns},
                        **{v: "median" for v in median_vars if v in fm_driver.columns}
                    }).reset_index())
    
            fm_driver_daily.to_csv(lake_output_dir / f"{lake_key}_fm_driver_daily.csv",index=False)
