import xarray as xr               #netcdf multidim reading/writing/manipulation
import glob                       #globbing
import numpy as np                #numerics
import os                         #operating system commands
import subprocess
import time as timer
import pop_tools
import sys

indir = '/glade/collections/cdg/timeseries-cmip6/b.e21.B1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1'
outdir = '/glade/scratch/whokim/temp_dat/'
case = 'b.e21.B1850.f09_g17.CMIP6-piControl.001'
tstmp = '000101-009912'
fout = f'{outdir}/{case}.pop.h.SDEN_F.{tstmp}.nc'

ht_vars = ['SHF', 'SHF_QSW', 'LWDN_F', 'LWUP_F', 'SENH_F', 'MELTH_F', 'QFLUX']
fw_vars = ['SFWF', 'EVAP_F', 'PREC_F', 'ROFF_F', 'IOFF_F', 'MELT_F', 'SALT_F', 'SNOW_F']
fin_ht = [f'{indir}/{case}.pop.h.' + var + f'.{tstmp}.nc' for var in ht_vars]
fin_fw = [f'{indir}/{case}.pop.h.' + var + f'.{tstmp}.nc' for var in fw_vars]
fin_t = f'{indir}/{case}.pop.h.TEMP.{tstmp}.nc'
fin_s = f'{indir}/{case}.pop.h.SALT.{tstmp}.nc'
# fin_ts = [f'{indir}/{case}.pop.h.TEMP.{tstmp}.nc', f'{indir}/{case}.pop.h.SALT.{tstmp}.nc']

# Open temp & salt
# Note: this is an workaround to avoid the imcompatibility of pop_tools eos function with daskarray
ds_t = xr.open_dataset(fin_t).isel(z_t=0)
ds_s = xr.open_dataset(fin_s).isel(z_t=0)
# ds_ts = xr.open_mfdataset(fin_ts, combine='by_coords').isel(z_t=0)

# compute alpha and beta using pop_tools
sst = ds_s['SALT']
sss = ds_t['TEMP']
# print(tmpt)

# pop_tools expects the same dimesion of depth as T&S
depth=xr.DataArray(np.zeros(np.shape(sst)),dims=sst.dims,coords=sst.coords)

rho,drhods,drhodt = pop_tools.eos(salt=sss,temp=sst,return_coefs=True,depth=depth)

alpha = drhodt/rho*-1
beta = drhods/rho*1e-3

# Load flux datasets
ds_ht = xr.open_mfdataset(fin_ht, combine='by_coords')
ds_fw = xr.open_mfdataset(fin_fw, combine='by_coords')

#conversion factors
cpsw = ds_fw['cp_sw']                    # erg/g/K
cpsw = cpsw/1.e4
cpsw.attrs['units'] = 'J/Kg/K'

latvap=ds_fw['latent_heat_vapor']        # J/kg

latfus=ds_fw['latent_heat_fusion']       # erg/g
latfus = latfus/1.e4
latfus.attrs['units'] = 'J/kg'

sfluxfact = ds_fw['sflux_factor']        # (msu*cm/s)/(kg SALT /m^2/s)
salinityfact = ds_fw['salinity_factor']  # (msu*cm/s)/(kg FW /m^2/s)

# Modify heat flux dataset
ds_ht['SHF'] = ds_ht['SHF'] + ds_ht['QFLUX']

# Make net LW and overide over LWDN_F, rename it LW, and drop LWUP_F
ds_ht['LWDN_F'] = ds_ht['LWDN_F'] + ds_ht['LWUP_F']

ds_ht = ds_ht.rename({'LWDN_F':'LW'})
ds_ht = ds_ht.drop('LWUP_F')

# Add latent heat flux to the heat flux dataset
latent = ds_fw['EVAP_F']*latvap
latent.attrs = {'units':'W/m^2','long_name':'Latent Heat Flux'}
ds_ht['Latent'] = latent

# Add heat flux due to snow to the heat flux dataset
snowmelt = ds_fw['SNOW_F']*latfus*-1
snowmelt.attrs = {'units':'W/m^2','long_name':'Heat Flux due to Snow Melt'}
ds_ht['Snowmelt'] = snowmelt

# Add heat flux due to solid runoff to the heat flux dataset
ioffmelt = ds_fw['IOFF_F']*latfus*-1
ioffmelt.attrs = {'units':'W/m^2','long_name':'Heat Flux due to Ice Runoff Melt'}
ds_ht['Ioffmelt'] = ioffmelt

# Modify freshwater flux dataset
ds_fw['SFWF'] = ds_fw['SFWF'] - ds_ht['QFLUX']/latfus

# (kg salt/m^2/s to kg freshwater/m^2/s); unit does not need to be changed
ds_fw['SALT_F'] = ds_fw['SALT_F']*sfluxfact/salinityfact

qflux = ds_ht['QFLUX']/latfus*-1
qflux.attrs = {'units':'kg/m^2/s','long_name':'Freshwater Flux due to Qflux'}
ds_fw['QFLUX'] = qflux

#========================================================================================
# Compute Surface Density Flux
# ---------------------------------------------------------------------------------------
# NOTE from Griffies on units:
# 
#    mass FW = mass SW - mass SALT
#            = rho*dV*(1 - S), since S == (mass SALT)/(mass SW)
#    ==> (1-S) == (mass FW)/(mass SW)
# 
#    F == (mass FW)/(m^2s), so F*S/(1-S) == [(kg FW)/(m^2s)]*[(kg SALT)/(kg SW)]
#                                           ------------------------------------
#                                                      (kg FW)/(kg SW)
#    Beta = (1/psu) = (kg SW)/(kg SALT), so
#    Beta * F * S/(1-S) == (kg SW)/(m^2 s) is a mass flux of SW
# 
# Origin of "rho_o" in some publications is that it is needed to convert F from
# cm/year or whateever into kg/m^2/s.
#----------------------------------------------------------------------------------------

# Compute total surface density flux and convert to a new dataset
df_t = ds_ht['SHF']*alpha/cpsw*-1 + ds_fw['SFWF']*beta*(sss/(1-sss))*-1
df_t.attrs = {'units':'kg/m^2/s','long_name':'Total Surface Density Flux'}
ds_df=df_t.to_dataset(name='SDEN_F')

# Compute surface density flux due to heat flux and add to the new dataset
df_h = ds_ht['SHF']*alpha/cpsw*-1
df_h.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Heat Flux'}
ds_df['SDEN_F_Q'] = df_h

# Compute surface density flux due to freshwater flux and add to the new dataset
df_f = ds_fw['SFWF']*beta*(sss/(1-sss))*-1
df_f.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Freshwater Flux'}
ds_df['SDEN_F_F'] = df_f

# Heat flux components
# Shortwave
df_h = ds_ht['SHF_QSW']*alpha/cpsw*-1
df_h.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Shortwave radiation'}
ds_df['SDEN_F_QSW'] = df_h

# Longwave
df_h = ds_ht['LW']*alpha/cpsw*-1
df_h.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Longwave radiation'}
ds_df['SDEN_F_QLW'] = df_h

# Latent
df_h = ds_ht['Latent']*alpha/cpsw*-1
df_h.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Latent Heat Flux'}
ds_df['SDEN_F_QLH'] = df_h

# Sensible
df_h = ds_ht['SENH_F']*alpha/cpsw*-1
df_h.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Sensible Heat Flux'}
ds_df['SDEN_F_QSH'] = df_h

# Melt heat flux
df_h = ds_ht['MELTH_F']*alpha/cpsw*-1
df_h.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Sea-Ice Melt Heat Flux'}
ds_df['SDEN_F_QME'] = df_h

# Snow melt
df_h = ds_ht['Snowmelt']*alpha/cpsw*-1
df_h.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Snow Melt Heat Flux'}
ds_df['SDEN_F_QSN'] = df_h

# Ice runoff melt
df_h = ds_ht['Ioffmelt']*alpha/cpsw*-1
df_h.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Ice Runoff Melt Heat Flux'}
ds_df['SDEN_F_QIO'] = df_h

#Qflux
df_h = ds_ht['QFLUX']*alpha/cpsw*-1
df_h.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Qflux Heat Flux'}
ds_df['SDEN_F_QQF'] = df_h

# Fw flux components
# Precipitation
df_f = ds_fw['PREC_F']*beta*(sss/(1-sss))*-1
df_f.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Precipitation'}
ds_df['SDEN_F_FPR'] = df_f

# Precipitation
df_f = ds_fw['EVAP_F']*beta*(sss/(1-sss))*-1
df_f.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Evaporation'}
ds_df['SDEN_F_FEV'] = df_f

# Runoff
df_f = ds_fw['ROFF_F']*beta*(sss/(1-sss))*-1
df_f.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Runoff Fw'}
ds_df['SDEN_F_FRO'] = df_f

# Ice runoff
df_f = ds_fw['IOFF_F']*beta*(sss/(1-sss))*-1
df_f.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Ice Runoff Fw'}
ds_df['SDEN_F_FIO'] = df_f

# Sea-ice melt
df_f = ds_fw['MELT_F']*beta*(sss/(1-sss))*-1
df_f.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Sea-Ice Melt Fw'}
ds_df['SDEN_F_FME'] = df_f

# Snow melt
df_f = ds_fw['SNOW_F']*beta*(sss/(1-sss))*-1
df_f.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Snow Melt Fw'}
ds_df['SDEN_F_FSN'] = df_f

# Salt flux
df_f = ds_fw['SALT_F']*beta*(sss/(1-sss))*-1
df_f.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Salt Flux Fw'}
ds_df['SDEN_F_FSA'] = df_f

# Qflux
df_f = ds_fw['QFLUX']*beta*(sss/(1-sss))*-1
df_f.attrs = {'units':'kg/m^2/s','long_name':'Surface Density Flux due to Qflux Fw'}
ds_df['SDEN_F_FQF'] = df_f

# Write SDF dataset in netcdf
ds_df.to_netcdf(fout,unlimited_dims='time')

