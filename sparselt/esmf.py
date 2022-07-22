import xarray as xr
import sparselt.linear_transform


def load_weights(filename, input_dims, output_dims):
    if isinstance(filename, xr.Dataset):
        ds_weights = filename
    else:
        ds_weights = xr.open_dataset(filename)

    # Get sparse matrix elements
    weights = ds_weights.S
    row_ind = ds_weights.row
    col_ind = ds_weights.col

    # Create a linear transform object
    transform = sparselt.linear_transform.SparseLinearTransform(
        weights, row_ind, col_ind,
        input_transform_dims=input_dims,
        output_transform_dims=output_dims,
        one_based_indices=True,
        order="C",
    )

    return transform
