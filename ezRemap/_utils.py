import xarray as xr

ds = xr.open_dataset('https://www.unidata.ucar.edu/software/netcdf/examples/sresa1b_ncar_ccsm3-example.nc')
print(ds)