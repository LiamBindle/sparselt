import numpy as np
import scipy.sparse


class SparseLinearTransform:
    _input_core_shape = None
    _input_core_dims = None
    _output_core_shape = None
    _output_core_dims = None
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

        self._input_core_dims = tuple(input_transform_dims[0])
        self._input_core_shape = tuple(input_transform_dims[1])

        self._output_core_dims = tuple(output_transform_dims[0])
        self._output_core_shape = tuple(output_transform_dims[1])

        self._matrix = scipy.sparse.csr_matrix((weights, (row_ind, col_ind)))
        self._order = order

        self._vfunc = self._create_vfunc()

    def _func(self, a: np.ndarray):
        a = a.flatten(order=self._order)
        return self._matrix.dot(a).reshape(self._output_core_shape, order=self._order)

    def _create_vfunc(self) -> callable:
        input_signature = ','.join(self.input_core_dims)
        output_signature = ','.join(self.output_core_dims)
        return np.vectorize(self._func, signature='({})->({})'.format(input_signature, output_signature))

    @property
    def vfunc(self) -> callable:
        return self._vfunc

    @property
    def input_core_dims(self) -> tuple:
        return self._input_core_dims

    @property
    def output_core_dims(self) -> tuple:
        return self._output_core_dims
