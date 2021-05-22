import xarray as xr
import xarray as xr
import ezRemap.esmf
import ezRemap.xr

# Open input dataset
ds = xr.open_dataset('../sample_data/sresa1b_ncar_ccsm3-example.nc')
print('Input dataset:\n', ds)

# Load weights from file
transform = ezRemap.esmf.load_weights(
    '../sample_data/sresa1b_to_c24_weights.nc',
    input_dims=[('lat', 'lon'), (128, 256)],
    output_dims=[('nf', 'Ydim', 'Xdim'), (6, 24, 24)],
)

# Output dataset template
output_dataset_template = xr.open_dataset('../sample_data/c24_template.nc')

# Remap input dataset
ds_out = ezRemap.xr.apply(transform, ds[['tas', 'pr']], output_dataset_template=output_dataset_template)

print('Output dataset:\n', ds_out)
