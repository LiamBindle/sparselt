import xarray as xr


def apply(transform, data_in, output_dataset_template=None):
    data_out = xr.apply_ufunc(
        transform.vfunc,
        data_in,
        input_core_dims=[transform.input_core_dims],
        output_core_dims=[transform.output_core_dims],
    )

    if output_dataset_template is not None:
        output_dataset = output_dataset_template.copy()
        if isinstance(data_out, xr.DataArray):
            output_dataset[data_out.name] = data_out
        else:
            output_dataset.update(data_out)
        return output_dataset
    else:
        return data_out
