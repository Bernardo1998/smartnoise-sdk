from snsynth.transform.definitions import ColumnType
from snsynth.transform.base import CachingColumnTransformer
from snsql.sql._mechanisms.approx_bounds import approx_bounds
from snsql.sql.privacy import Privacy
import numpy as np
import pandas as pd

from rdt.transformers import ClusterBasedNormalizer, OneHotEncoder

class LogBoundedTransformer(CachingColumnTransformer):
    """Logarithmic transformation of values.  Useful for transforming skewed data.
    """
    def __init__(self):
        super().__init__()
        self.lower_bound = -1
    @property
    def output_type(self):
        return ColumnType.CONTINUOUS
    @property
    def cardinality(self):
        return [None]
    def _fit(self, val, idx=None):
        self.lower_bound = min(self.lower_bound, val)
    def _fit_finish(self):
        pass
    def _clear_fit(self):
        self._fit_complete = True
        self.lower_bound = -1
        self.output_width = 1
    def _transform(self, val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None
        else:
            return float(np.log(val- self.lower_bound)) 
    def _inverse_transform(self, val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return np.nan
        else:
            return float(np.exp(val))+self.lower_bound