import xarray as xr
import xarray as xr
import ezRemap.esmf
import ezRemap.xr
from pathlib import Path


sample_data_dir = Path('../sample_data')

original_dataset = sample_data_dir.joinpath('sample_data-c48/GCHP.SpeciesConc_O3.20190916_0000z.nc4')
weights_file = sample_data_dir.joinpath('weights_c48_to_latlon361x576_bilinear.nc')
output_dataset_template = sample_data_dir.joinpath('latlon361x576_output_template.nc')


# Open input dataset
ds = xr.open_dataset(original_dataset)

# Load weights from file
transform = ezRemap.esmf.load_weights(
    weights_file,
    input_dims=[('nf', 'Ydim', 'Xdim'), (6, 48, 48)],
    output_dims=[('lat', 'lon'), (361, 576)],
)

# Output dataset template
output_dataset_template = xr.open_dataset(output_dataset_template)

# Remap input dataset
ds_out = ezRemap.xr.apply(transform, ds, output_dataset_template=output_dataset_template)

print('Output dataset:\n', ds_out)

# Plot
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
surface_ozone = ds_out.SpeciesConc_O3.isel(lev=0).squeeze()
p = surface_ozone.plot(subplot_kws=dict(projection=ccrs.PlateCarree(), facecolor="gray"), transform=ccrs.PlateCarree())
p.axes.set_global()
p.axes.coastlines()
plt.show()
