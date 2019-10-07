import xarray as xr               #netcdf multidim reading/writing/manipulation
import numpy as np                #numerics
import os                         #operating system commands
import subprocess
import time as timer
import pop_tools
import sys

time1=timer.time()

# Set Options
append_to_infile=False  # True with "-a"
sigmachoice = 'sigma0'

indir = '/glade/scratch/yeager/cesm2/wmf'
outdir = '/glade/scratch/yeager/cesm2/wmf'
case = 'b.e21.B1850.f09_g17.CMIP6-piControl.001'
tstmp = '000101-009912'
fin = f'{indir}/{case}.pop.h.SDEN_F.{tstmp}.nc'

focn = '/glade/p/cgd/oce/people/yeager/POP_grids/gx1v6_ocn.nc'

if ('-a' in sys.argv[:]):
   append_to_infile=True
if ('-sig2' in sys.argv[:]):
   sigmachoice='sigma2'

#Load SST/SSS
ds = xr.open_dataset(focn)
rmask = ds['REGION_MASK']
rmask = rmask.drop(['ULONG','ULAT'])
ds = xr.open_dataset(fin)
time=ds['time']
time.encoding['_FillValue']=None       
tlon = ds['TLONG']
tlat = ds['TLAT']
tarea = ds['TAREA'].astype(np.float32)/1.e4
tarea.attrs['units']='m^2'
sden = ds['SDEN_F']
sdenq = ds['SDEN_F_Q']
sdenf = ds['SDEN_F_F']
ssd0 = ds['sigma0']
ssd2 = ds['sigma2']
tlon = tlon.where(tlon<180.,other=tlon-360.)
nt = np.shape(ssd0)[0]

# Convert surface density fluxes to kg/s
sden = sden*tarea
sdenq = sdenq*tarea
sdenf = sdenf*tarea

# define WMF regions
wmf_regions = rmask.copy()
lab = (tlat.values>=51) & (tlat.values<=65) & (tlon.values>=-65.) & (tlon.values<=-45.) & (rmask.values>0)
wmf_regions.values = np.where(lab,1,0)
spg = (tlat.values>=51) & (tlat.values<=60) & (tlon.values>=-45.) & (tlon.values<=-5.) & (rmask.values>0)
wmf_regions.values = np.where(spg,2,wmf_regions.values)
nor = (tlat.values>=60) & (tlat.values<=80) & (tlon.values>=-30.) & (tlon.values<=15.) & (rmask.values>0)
wmf_regions.values = np.where(nor,4,wmf_regions.values)
irm = (tlat.values>=60) & (tlat.values<=70) & (tlon.values>=-45.) & (tlon.values<=-5.) & (rmask.values>0)     & (tlat.values<=(58.-(2./5)*tlon.values))
wmf_regions.values = np.where(irm,3,wmf_regions.values)
nregions=5
regions=['North Atlantic','Labrador Sea','Subpolar Gyre','Irminger Sea','Norwegian Sea']

# Define sigma coordinates
if (sigmachoice=='sigma0'):
    ssd = ssd0
    sig = np.linspace(24,30.,61)
    nsig = len(sig)
    sig_lo = sig.copy()
    tmp = 0.5*(sig[1:]-sig[0:-1])
    sig_lo[0] = sig[0]-tmp[0]
    sig_lo[1:] = sig[0:-1]+tmp
    sig_hi = sig.copy()
    sig_hi[-1] = sig[-1]+tmp[-1]
    sig_hi[0:-1] = sig[0:-1]+tmp
    dsig=sig_hi-sig_lo
    sigma=xr.DataArray(sig,dims=['sigma_wmt'],coords={'sigma_wmt':sig},attrs={'long_name':'Sigma0','units':'kg/m^3'})
    sigma_hi=xr.DataArray(sig_hi[0:-1],dims=['sigma_wmf'],coords={'sigma_wmf':sig_hi[0:-1]},attrs={'long_name':'Sigma0','units':'kg/m^3'})
if (sigmachoice=='sigma2'):
    ssd = ssd2
    sig = np.linspace(35,38,61)
    nsig = len(sig)
    sig_lo = sig.copy()
    tmp = 0.5*(sig[1:]-sig[0:-1])
    sig_lo[0] = sig[0]-tmp[0]
    sig_lo[1:] = sig[0:-1]+tmp
    sig_hi = sig.copy()
    sig_hi[-1] = sig[-1]+tmp[-1]
    sig_hi[0:-1] = sig[0:-1]+tmp
    dsig=sig_hi-sig_lo
    sigma=xr.DataArray(sig,dims=['sigma_wmt'],coords={'sigma_wmt':sig},attrs={'long_name':'Sigma2','units':'kg/m^3'})
    sigma_hi=xr.DataArray(sig_hi[0:-1],dims=['sigma_wmf'],coords={'sigma_wmf':sig_hi[0:-1]},attrs={'long_name':'Sigma2','units':'kg/m^3'})
sigma.encoding['_FillValue']=None       
sigma_hi.encoding['_FillValue']=None       
# get rid of Nan's in ssd:
ssd.values[np.isnan(ssd.values)]=0.

# set up output xarrays
qcomps = ['Total', 'Shortwave', 'Longwave', 'Latent', 'Sensible', 'Icemelt', 'Snowmelt', 'IOFFmelt', 'Frazil']
ncompq = np.size(qcomps)
fcomps = ['Total', 'Precip','Evap','ROFF','IOFF','Icemelt','BrineReject','Frazil']
ncompf = np.size(fcomps)
WMT = xr.DataArray(np.zeros((nt,nregions,nsig),dtype=np.float32),dims=['time','wmf_region','sigma_wmt'], \
    coords={'time':time,'wmf_region':regions,'sigma_wmt':sigma}, name='WMT',attrs={'long_name':'Water Mass Transformation','units':'Sv'})
WMT.encoding['_FillValue']=1.e30
WMT_Q = xr.DataArray(np.zeros((nt,ncompq,nregions,nsig),dtype=np.single),dims=['time','qcomp','wmf_region','sigma_wmt'], \
    coords={'time':time,'qcomp':qcomps,'wmf_region':regions,'sigma_wmt':sigma}, name='WMT_Q',attrs={'long_name':'Water Mass Transformation (Heat)','units':'Sv'})
WMT_Q.encoding['_FillValue']=1.e30
WMT_F = xr.DataArray(np.zeros((nt,ncompf,nregions,nsig),dtype=np.single),dims=['time','fcomp','wmf_region','sigma_wmt'], \
    coords={'time':time,'fcomp':fcomps,'wmf_region':regions,'sigma_wmt':sigma}, name='WMT_F',attrs={'long_name':'Water Mass Transformation (Freshwater)','units':'Sv'})
WMT_F.encoding['_FillValue']=1.e30

WMF = xr.DataArray(np.zeros((nt,nregions,nsig-1),dtype=np.float32),dims=['time','wmf_region','sigma_wmf'], \
    coords={'time':time,'wmf_region':regions,'sigma_wmf':sigma_hi}, name='WMF',attrs={'long_name':'Water Mass Formation','units':'Sv'})
WMF.encoding['_FillValue']=1.e30
WMF_Q = xr.DataArray(np.zeros((nt,ncompq,nregions,nsig-1),dtype=np.single),dims=['time','qcomp','wmf_region','sigma_wmf'], \
    coords={'time':time,'qcomp':qcomps,'wmf_region':regions,'sigma_wmf':sigma_hi}, name='WMT_Q',attrs={'long_name':'Water Mass Formation (Heat)','units':'Sv'})
WMF_Q.encoding['_FillValue']=1.e30
WMF_F = xr.DataArray(np.zeros((nt,ncompf,nregions,nsig-1),dtype=np.single),dims=['time','fcomp','wmf_region','sigma_wmf'], \
    coords={'time':time,'fcomp':fcomps,'wmf_region':regions,'sigma_wmf':sigma_hi}, name='WMT_F',attrs={'long_name':'Water Mass Formation (Freshwater)','units':'Sv'})
WMF_F.encoding['_FillValue']=1.e30

# Compute transformation
for region in regions:
    ir = regions.index(region)
    if (ir==0):
        region_domain = (wmf_regions.values>0)
    else:
        region_domain = (wmf_regions.values==ir)
    for isig in range(nsig):
        outcrop_domain = (ssd.values>=sig_lo[isig]) & (ssd.values<sig_hi[isig]) & region_domain
        work = sden.where(outcrop_domain).sum(dim='nlon').sum(dim='nlat')/dsig[isig]   # m^3/s
        WMT.values[:,ir,isig] = work.values/1.e6
        work = sdenq.where(outcrop_domain[:,None,:,:]).sum(dim='nlon').sum(dim='nlat')/dsig[isig]   # m^3/s
        WMT_Q.values[:,:,ir,isig] = work.values/1.e6
        work = sdenf.where(outcrop_domain[:,None,:,:]).sum(dim='nlon').sum(dim='nlat')/dsig[isig]   # m^3/s
        WMT_F.values[:,:,ir,isig] = work.values/1.e6

# Compute formation
WMF.values = -(WMT.values[:,:,1:]-WMT.values[:,:,0:-1])
WMF_Q.values = -(WMT_Q.values[:,:,:,1:]-WMT_Q.values[:,:,:,0:-1])
WMF_F.values = -(WMT_F.values[:,:,:,1:]-WMT_F.values[:,:,:,0:-1])

# Write to netcdf
out_ds=wmf_regions.to_dataset(name='WMF_REGION_MASK')
out_ds.TLONG.attrs=tlon.attrs
out_ds.TLONG.encoding['_FillValue']=None        # because xarray is weird
out_ds.TLAT.attrs=tlat.attrs
out_ds.TLAT.encoding['_FillValue']=None        # because xarray is weird
out_ds['WMT']=WMT
out_ds['WMT_Q']=WMT_Q
out_ds['WMT_F']=WMT_F
out_ds['WMF']=WMF
out_ds['WMF_Q']=WMF_Q
out_ds['WMF_F']=WMF_F
out_ds.to_netcdf(fout,unlimited_dims='time')

if append_to_infile:
   cmd = ['ncks','-A','-h','-v','WMF_REGION_MASK,WMT,WMT_Q,WMT_F,WMF,WMF_Q,WMF_F',fout,fin]
   subprocess.call(cmd)
   cmd = ['rm','-f',fout]
   subprocess.call(cmd)
   time2=timer.time()
   print('DONE appending to ',fin,'.  Total time = ',time2-time1,'s')
else:
   time2=timer.time()
   print('DONE creating ',fout,'.  Total time = ',time2-time1,'s')
