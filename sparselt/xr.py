import numpy as np
import xarray as xr


def apply(transform, data_in, output_dataset_template=None):
    drop_list = [name for name in data_in.data_vars if any([dim not in data_in[name].dims for dim in transform.demangled_input_core_dims])] 
    data_in = data_in.drop_vars(drop_list)
    data_in = data_in.rename({k: v for k, v in zip(transform.demangled_input_core_dims, transform.mangled_input_core_dims)})

    keep_attrs = output_dataset_template is None
    data_out = xr.apply_ufunc(
        transform.vfunc,
        data_in,
        input_core_dims=[transform.mangled_input_core_dims],
        output_core_dims=[transform.mangled_output_core_dims],
        keep_attrs=keep_attrs
    )
    data_out = data_out.rename({k: v for k, v in zip(transform.mangled_output_core_dims, transform.demangled_output_core_dims)})

    if output_dataset_template is not None:
        output_dataset = output_dataset_template.copy()
        
        for name in output_dataset.data_vars:
            output_dataset[name][...] = np.nan
        if isinstance(data_out, xr.DataArray):
            output_dataset[data_out.name] = data_out
        else:
            data_vars_intersection = set(output_dataset_template.data_vars).intersection(set(data_out.data_vars))
        
        for name in data_vars_intersection:
            output_dataset[name].values = data_out[name].values
        
        data_out = data_out.drop(data_vars_intersection)
        output_dataset = xr.merge([output_dataset, data_out], compat='override', join='override', combine_attrs='override')
        return output_dataset
    else:
        return data_out
