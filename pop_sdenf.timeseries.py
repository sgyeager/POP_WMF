import xarray as xr               #netcdf multidim reading/writing/manipulation
import glob                       #globbing
import numpy as np                #numerics
import os                         #operating system commands
import subprocess
import time as timer
import pop_tools
import sys
#import gsw

time1=timer.time()

indir = '/glade/collections/cdg/timeseries-cmip6/b.e21.B1850.f09_g17.CMIP6-piControl.001/ocn/proc/tseries/month_1'
outdir = '/glade/scratch/yeager/cesm2/wmf'
case = 'b.e21.B1850.f09_g17.CMIP6-piControl.001'
tstmp = '000101-009912'
fout = f'{outdir}/{case}.pop.h.SDEN_F.{tstmp}.nc'

ht_vars = ['SHF', 'SHF_QSW', 'LWDN_F', 'LWUP_F', 'SENH_F', 'MELTH_F', 'QFLUX']
fw_vars = ['SFWF', 'EVAP_F', 'PREC_F', 'ROFF_F', 'IOFF_F', 'MELT_F', 'SALT_F', 'SNOW_F']

fin_ht = [f'{indir}/{case}.pop.h.' + var + f'.{tstmp}.nc' for var in ht_vars]
fin_fw = [f'{indir}/{case}.pop.h.' + var + f'.{tstmp}.nc' for var in fw_vars]
fin_t = f'{indir}/{case}.pop.h.TEMP.{tstmp}.nc'
fin_s = f'{indir}/{case}.pop.h.SALT.{tstmp}.nc'

#Load SST/SSS
ds_s = xr.open_dataset(fin_s)
sss_psu = ds_s['SALT'].isel(z_t=0)
sss_msu=sss_psu.copy()/1000.
sss_msu.attrs['units']='kg/kg'

ds_t = xr.open_dataset(fin_t)
non_dim_coords_reset=set(ds_t.coords)-set(ds_t.dims)
ds_t = ds_t.reset_coords(non_dim_coords_reset)
sst = ds_t['TEMP'].isel(z_t=0)
nt = np.shape(sst)[0]
ny = np.shape(sst)[1]
nx = np.shape(sst)[2]
time=ds_t['time']
time.encoding['_FillValue']=None        # because xarray is weird
tlon = ds_t['TLONG']
tlat = ds_t['TLAT']
cpsw = ds_t['cp_sw']                    # erg/g/K
cpsw = cpsw/1.e4
cpsw.attrs['units'] = 'J/Kg/K'
latvap=ds_t['latent_heat_vapor']       # J/kg
latfus=ds_t['latent_heat_fusion']      # erg/g
latfus = latfus/1.e4
latfus.attrs['units'] = 'J/kg'
sfluxfact = ds_t['sflux_factor']        # (msu*cm/s)/(kg SALT /m^2/s)
salinityfact = ds_t['salinity_factor']  # (msu*cm/s)/(kg FW /m^2/s)

# Compute rho, alpha, & beta using pop-tools.eos
depth=xr.DataArray(np.zeros(np.shape(sst)),dims=sst.dims,coords=sst.coords)
rho,drhods,drhodt = pop_tools.eos(salt=sss_psu,temp=sst,return_coefs=True,depth=depth)
alpha = (drhodt/rho)*-1		# 1/degC
beta = (drhods/rho)		# kg(Seawater)/kg(SALT)

#Load fluxes
ds_ht = xr.open_mfdataset(fin_ht, combine='by_coords')
ds_fw = xr.open_mfdataset(fin_fw, combine='by_coords')

#Perform needed heat flux manipulations
ds_ht['SHF'] = ds_ht['SHF'] + ds_ht['QFLUX']		# Net heat flux
ds_ht['LWDN_F'] = ds_ht['LWDN_F'] + ds_ht['LWUP_F']	# Net LW
# Compute Latent heat flux
latent=ds_fw['EVAP_F']*latvap
latent.attrs = {'units':'W/m^2','long_name':'Latent Heat Flux'}
ds_ht['Latent']=latent
# Compute Snowmelt heat flux
snowmelt=ds_fw['SNOW_F']*latfus*-1
snowmelt.attrs = {'units':'W/m^2','long_name':'Heat Flux due to snow melt'}
ds_ht['Snowmelt']=snowmelt
# Compute heat flux due to solid river runoff
ioffmelt=ds_fw['IOFF_F']*latfus*-1
ioffmelt.attrs = {'units':'W/m^2','long_name':'Heat Flux due to solid runoff melt'}
ds_ht['IOFFmelt']=ioffmelt
# rename Q components
qcomps = ['Total', 'Shortwave', 'Longwave', 'Latent', 'Sensible', 'Icemelt', 'Snowmelt', 'IOFFmelt', 'Frazil']
ds_ht=ds_ht.rename({'SHF':'Total'})
ds_ht=ds_ht.rename({'SHF_QSW':'Shortwave'})
ds_ht=ds_ht.rename({'LWDN_F':'Longwave'})
ds_ht=ds_ht.rename({'SENH_F':'Sensible'})
ds_ht=ds_ht.rename({'MELTH_F':'Icemelt'})
ds_ht=ds_ht.rename({'QFLUX':'Frazil'})

#Perform needed freshwater flux manipulations
ds_fw['SFWF'] = ds_fw['SFWF'] - ds_ht['Frazil']/latfus	
ds_fw['SALT_F'] = ds_fw['SALT_F']*sfluxfact/salinityfact	# (kg salt/m^2/s to kg freshwater/m^2/s); units remain unchanged
qflux = ds_ht['Frazil']/latfus*-1
qflux.attrs = {'units':'kg/m^2/s','long_name':'Freshwater Flux due to Qflux'}
ds_fw['Frazil'] = qflux
# rename FW components
ds_fw=ds_fw.rename({'SFWF':'Total'})
ds_fw=ds_fw.rename({'PREC_F':'Precip'})
ds_fw=ds_fw.rename({'EVAP_F':'Evap'})
ds_fw=ds_fw.rename({'ROFF_F':'ROFF'})
ds_fw=ds_fw.rename({'IOFF_F':'IOFF'})
ds_fw=ds_fw.rename({'MELT_F':'Icemelt'})
ds_fw=ds_fw.rename({'SALT_F':'BrineReject'})

#========================================================================================
# Compute Surface Density Flux
#
# References
# ----------
# Speer & Tziperman, 1992: Rates of Water Mass Formation in the North 
# Atlantic Ocean, J. Phys. Oceanogr., 22, 93-104.
#
# Large, W. G. and G. Nurser, 2001: Chapter 5.1 ocean surface water mass transformation. Ocean
# Circulation and Climate – Observing and Modelling the Global Ocean, Academic Press, International
# Geophysics, Vol. 77, 317–336, doi:10.1016/S0074-6142(01)80126-1.
#
#	SDENS_F = -(alpha*SHF)/C_p    + (beta*SFWF)*(SSS/(1-SSS))
#
# NOTE on units:
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

# set up output xarrays
qcomps = ['Total', 'Shortwave', 'Longwave', 'Latent', 'Sensible', 'Icemelt', 'Snowmelt', 'IOFFmelt', 'Frazil']
ncompq = np.size(qcomps)
fcomps = ['Total', 'Precip','Evap','ROFF','IOFF','Icemelt','BrineReject','Frazil']
ncompf = np.size(fcomps)
SDEN = xr.DataArray(np.zeros((nt,ny,nx),dtype=np.float32),dims=['time','nlat','nlon'], \
      coords={'time':time,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
      name='SDEN',attrs={'long_name':'Surface Density Flux','units':'kg/m^2/s'})
SDEN.encoding['_FillValue']=1.e30
SDEN_Q = xr.DataArray(np.zeros((nt,ncompq,ny,nx),dtype=np.single),dims=['time','qcomp','nlat','nlon'], \
      coords={'time':time,'qcomp':qcomps,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
      name='SDEN_Q',attrs={'long_name':'Surface Density Flux (Heat)','units':'kg/m^2/s'})
SDEN_Q.encoding['_FillValue']=1.e30
SDEN_F = xr.DataArray(np.zeros((nt,ncompf,ny,nx),dtype=np.single),dims=['time','fcomp','nlat','nlon'], \
      coords={'time':time,'fcomp':fcomps,'TLAT':(('nlat','nlon'),tlat),'TLONG':(('nlat','nlon'),tlon)}, \
      name='SDEN_F',attrs={'long_name':'Surface Density Flux (Freshwater)','units':'kg/m^2/s'})
SDEN_F.encoding['_FillValue']=1.e30

# Compute surface density flux and fill xarrays
work = (-alpha*ds_ht['Total']/cpsw - beta*ds_fw['Total']*(sss_msu/(1-sss_msu))).astype(np.float32)
SDEN.values=work.values

# Compute surface density flux due to heat flux and add to the new dataset
for i in qcomps:
   work = (-alpha*ds_ht[i]/cpsw).astype(np.float32)
   j = qcomps.index(i)
   SDEN_Q.values[:,j,:,:] = work.values

# Compute surface density flux due to freshwater flux and add to the new dataset
for i in fcomps:
   work = (-beta*ds_fw[i]*(sss_msu/(1-sss_msu))).astype(np.float32)
   j = fcomps.index(i)
   SDEN_F.values[:,j,:,:] = work.values

# Write SDF dataset in netcdf
out_ds=SDEN.to_dataset(name='SDEN_F')
out_ds.TLONG.attrs=tlon.attrs
out_ds.TLONG.encoding['_FillValue']=None        # because xarray is weird
out_ds.TLAT.attrs=tlat.attrs
out_ds.TLAT.encoding['_FillValue']=None        # because xarray is weird
out_ds['SDEN_F_Q']=SDEN_Q
out_ds['SDEN_F_F']=SDEN_F
#out_ds['alpha']=alpha
#out_ds['alpha_gsw']=alpha_gsw
#out_ds['beta']=beta
#out_ds['rho_computed']=rho
#out_ds['SSD']=ssd
print(out_ds.TLONG.attrs)
out_ds.to_netcdf(fout,unlimited_dims='time')

time2=timer.time()
print('DONE creating ',fout,'. Total time = ',time2-time1,'s')
