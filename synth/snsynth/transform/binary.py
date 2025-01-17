
from snsynth.transform.definitions import ColumnType
from snsynth.transform.base import ColumnTransformer
import numpy as np


class BinaryEncoder(ColumnTransformer):
    """Transforms integer-labeled data into binary representation of its values.  Inputs are assumed to be 0-based.
    To convert from unstructured categorical data, chain with LabelTransformer first.
    """
    cache_fit = False
    def __init__(self, digits=16):
        super().__init__()
        self.digits = digits
    @property
    def output_type(self):
        return ColumnType.CATEGORICAL
    @property
    def cardinality(self):
        return [2] * (self.max + 1)
    def _fit(self, val):
        if val > self.max:
            self.max = val
    def _fit_finish(self):
        s#elf.digits = len(bin(self.max)) - 2 # Determine digits based on max level. Subtract 2 to remove '0b'
        self.output_width = self.digits
        super()._fit_finish()
    def _clear_fit(self):
        self._fit_complete = False
        self.max = -1
    def _transform(self, val):
        if self.max < 0 or not self._fit_complete:
            raise ValueError("OneHotEncoder has not been fit yet.")
        elif val < 0 or val > self.max:
            raise ValueError(
                f"Provided integer-label {val} is invalid."
                " Please ensure that all inputs are 0-based and provided during data fit."
            )
        elif self.max == 0:
            return 1

        binary_string = format(val, f'0{self.digits}b')
        bits = [int(bit) for bit in binary_string]
        #print(len(bits), bits, self.output_width)
        return tuple(bits)
    def _inverse_transform(self, val):
        # will always choose first if multiple are set
        binary_list = [int(round(v)) for v in val]
        print(val, len(binary_list), binary_list)
        return int(''.join(map(str, binary_list)), 2)
        
        
'''


class BinaryEncoder(ColumnTransformer):
    """Transforms integer-labeled data into one-hot encoding.  Inputs are assumed to be 0-based.
    To convert from unstructured categorical data, chain with LabelTransformer first.
    """
    cache_fit = False
    def __init__(self):
        super().__init__()
    @property
    def output_type(self):
        return ColumnType.CATEGORICAL
    @property
    def cardinality(self):
        return [2] * (self.max + 1)
    def _fit(self, val):
        if val > self.max:
            self.max = val
    def _fit_finish(self):
        self.output_width = self.max + 1
        super()._fit_finish()
    def _clear_fit(self):
        self._fit_complete = False
        self.max = -1
    def _transform(self, val):
        if self.max < 0 or not self._fit_complete:
            raise ValueError("OneHotEncoder has not been fit yet.")
        elif val < 0 or val > self.max:
            raise ValueError(
                f"Provided integer-label {val} is invalid."
                " Please ensure that all inputs are 0-based and provided during data fit."
            )
        elif self.max == 0:
            return 1

        bits = [0] * (self.max + 1)
        bits[val] = 1
        return tuple(bits)
    def _inverse_transform(self, val):
        # will always choose first if multiple are set
        return np.argmax(val)
'''
