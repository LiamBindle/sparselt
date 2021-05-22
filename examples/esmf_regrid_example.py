import xarray as xr
import splint.esmf
import splint.xr

# Load dataset on C48 grid
ds = xr.open_dataset('sample_data/gchp13_sample_species.nc')
print('Original dataset: {}\n\n'.format(str(ds)))

# Create SparseLinearTransform object from an ESMF weights file
transform = splint.esmf.load_weights(
    'sample_data/esmf_regrid_weights_c48_to_latlon90x180.nc',
    input_dims=[('nf', 'Ydim', 'Xdim'), (6, 48, 48)],
    output_dims=[('lat', 'lon'), (90, 180)],
)

# Open an output template dataset (optional)
output_template = xr.open_dataset('sample_data/regular_lat_lon_90x180.nc')

# Apply the transform to ds
ds = splint.xr.apply(transform, ds, output_template)

print('Transformed dataset: {}\n\n'.format(str(ds)))
