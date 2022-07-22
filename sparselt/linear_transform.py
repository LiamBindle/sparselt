import numpy as np
import scipy.sparse
import random
import string

class SparseLinearTransform:
    _input_core_shape = None
    _mangled_input_core_dims = None
    _output_core_shape = None
    _mangled_output_core_dims = None
    _matrix = None
    _order = None
    _vfunc = None

    def __init__(self, weights, row_ind, col_ind,
                 input_transform_dims, output_transform_dims,
                 one_based_indices=False, order="C"):
        weights = np.asarray(weights)
        row_ind = np.asarray(row_ind)
        col_ind = np.asarray(col_ind)
        if one_based_indices:
            row_ind -= 1
            col_ind -= 1

        self._input_mangle_suffix = ''.join(random.choice(string.ascii_letters) for _ in range(10))
        self._demangled_input_core_dims = tuple(input_transform_dims[0])
        self._mangled_input_core_dims = tuple(f'{name}_{self._input_mangle_suffix}' for name in self.demangled_input_core_dims)
        self._input_core_shape = tuple(input_transform_dims[1])
        input_size = np.product(self._input_core_shape)

        self._output_mangle_suffix = ''.join(random.choice(string.ascii_letters) for _ in range(10))
        self._demangled_output_core_dims = tuple(output_transform_dims[0])
        self._mangled_output_core_dims = tuple(f'{name}_{self._output_mangle_suffix}' for name in self.demangled_output_core_dims)
        self._output_core_shape = tuple(output_transform_dims[1])
        output_size = np.product(self._output_core_shape)

        self._matrix = scipy.sparse.csr_matrix((weights, (row_ind, col_ind)), shape=(output_size, input_size))
        self._order = order

        self._vfunc = self._create_vfunc()

    def mangle_dim_names(self, dim_names, are_input_dims):
        return [f'{name}_{are_input_dims if self._input_mangle_suffix else self._output_mangle_suffix}' for name in dim_names]

    def demangle_dim_names(self, dim_names):
        mangle_suffix_len = 11
        return [name[:-mangle_suffix_len] for name in dim_names]

    def _func(self, a: np.ndarray):
        a = a.flatten(order=self._order)
        return self._matrix.dot(a).reshape(self._output_core_shape, order=self._order)

    def _create_vfunc(self) -> callable:
        input_signature = ','.join(self.mangled_input_core_dims)
        output_signature = ','.join(self.mangled_output_core_dims)
        return np.vectorize(self._func, signature='({})->({})'.format(input_signature, output_signature))

    @property
    def vfunc(self) -> callable:
        return self._vfunc

    @property
    def mangled_input_core_dims(self) -> tuple:
        return self._mangled_input_core_dims
    
    @property
    def demangled_input_core_dims(self) -> tuple:
        return self._demangled_input_core_dims

    @property
    def mangled_output_core_dims(self) -> tuple:
        return self._mangled_output_core_dims
    
    @property
    def demangled_output_core_dims(self) -> tuple:
        return self._demangled_output_core_dims
