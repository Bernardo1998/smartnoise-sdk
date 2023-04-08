'''
03052023: THIS FILES HAS BEEN COMPLETED.
'''

from snsynth.transform.definitions import ColumnType
from snsynth.transform.base import CachingColumnTransformer
from snsql.sql._mechanisms.approx_bounds import approx_bounds
from snsql.sql.privacy import Privacy
import numpy as np
import pandas as pd

from .numerical import FloatFormatter, ClusterBasedNormalizer

class ZeroInflatedNULLTransformer(CachingColumnTransformer):
    """Transforms a column of zero-inflated continuous values.
       Just add inflated columns, keep the original values.
    :param epsilon: The privacy budget to use to infer bounds, if none provided.
    :param nullable: If null values are expected, a second output will be generated indicating null.
    :param odometer: The optional odometer to use to track privacy budget.
    """
    def __init__(self, *, negative=True, epsilon=0.0, nullable=False, odometer=None,model_missing_values=True,inflat_thres=0.1):
        #self.lower = lower
        #self.upper = upper
        self.epsilon = epsilon
        self.budget_spent = []
        self.nullable = nullable
        self.odometer = odometer
        self.model_missing_values = model_missing_values
        self.gm = None
        self.output_width = None
        self.inflat_thres = inflat_thres # The threshold that one value is determine inflated
        super().__init__()
    @property
    def output_type(self):
        return ColumnType.CONTINUOUS
    @property
    def needs_epsilon(self):
        # TODO: confirm how to apply privacy budget for this.
        return False
    @property
    def cardinality(self):
        # TODO: confirm this
        if self.nullable:
            return [None, 2]
        else:
            return [None]
    def allocate_privacy_budget(self, epsilon, odometer):
        self.epsilon = epsilon
        self.odometer = odometer
    def _fit_finish(self):
        if self.epsilon is not None and self.epsilon > 0.0 and (self.lower is None or self.upper is None):
            if self.odometer is not None:
                self.odometer.spend(Privacy(epsilon=self.epsilon, delta=0.0))
            self.budget_spent.append(self.epsilon)

        self._fit_vals = [v for v in self._fit_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            
        self._fit_complete = True
        self.output_width = 1 

        
        self.output_width += 2
          
    def _clear_fit(self):
        self._reset_fit()
        self.output_width = None
        
    def _transform(self, val):
        if not self.fit_complete:
            raise ValueError("GMMClusterTransformer has not been fit yet.")

        output = [0] * self.output_width
        
        # If val is missing, change it to 0 for easy modeling of GAN.
        # Note that if any missing val occur here, self.model_missing_values = T
        if val != 0:
          output[0] = val
          # Set one-hot columns for non-zero
          output[self.output_width-1] = 1.0
        else:
          # Set one-hot columns for zero
          output[self.output_width-2] = 1.0
          
        return tuple(output)      
        
    def _inverse_transform(self, val, sigmas=None):
        '''
        Args: 
        val: a tuple: (distance_to_mode, one_hot_columsn_of_modes)
        sigmas: deprecated
    	
        Return: 
        Value reconstructred from mode form.
        '''
        if not self.fit_complete:
            raise ValueError("GMMClusterTransformer has not been fit yet.")
        # Use argmax on the last few elements to determine if this is a zero or NA.
        # If so just return 0 or None
        # Else remove the last two elements and feed the rest to GMM
        if val[-2]>val[-1]:
          return 0
        else:
          val = val[:(len(val)-2)]

        return val[0]

class ZeroInflatedGMMClusterTransformer(CachingColumnTransformer):
    """Transforms a column of zero-inflated continuous values
    :param epsilon: The privacy budget to use to infer bounds, if none provided.
    :param nullable: If null values are expected, a second output will be generated indicating null.
    :param odometer: The optional odometer to use to track privacy budget.
    """
    def __init__(self, *, negative=True, epsilon=0.0, nullable=False, odometer=None,model_missing_values=True):
        #self.lower = lower
        #self.upper = upper
        self.epsilon = epsilon
        self.budget_spent = []
        self.nullable = nullable
        self.odometer = odometer
        self.model_missing_values = model_missing_values
        self.gm = None
        self.output_width = None
        super().__init__()
    @property
    def output_type(self):
        return ColumnType.CONTINUOUS
    @property
    def needs_epsilon(self):
        # TODO: confirm how to apply privacy budget for this.
        return False
    @property
    def cardinality(self):
        # TODO: confirm this
        if self.nullable:
            return [None, 2]
        else:
            return [None]
    def allocate_privacy_budget(self, epsilon, odometer):
        self.epsilon = epsilon
        self.odometer = odometer
    def _fit_finish(self):
        if self.epsilon is not None and self.epsilon > 0.0 and (self.lower is None or self.upper is None):
            if self.odometer is not None:
                self.odometer.spend(Privacy(epsilon=self.epsilon, delta=0.0))
            self.budget_spent.append(self.epsilon)

        # Make a one-column pd.DataFrame following the gm format
        self._fit_vals = [v for v in self._fit_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        data = pd.DataFrame({'column':self._fit_vals})
        self.gm = ClusterBasedNormalizer(model_missing_values=self.model_missing_values, max_clusters=min(len(data), 10))
        self.gm.fit(data, 'column')

        num_components = sum(self.gm.valid_component_indicator)
        self._fit_complete = True
        self.output_width = 1 + num_components
        
        # Add two more columns at the end to indicate zero vs non-zero
        # [0,1] means non-zero and [1,0] means zero
        self.output_width += 2 
    def _clear_fit(self):
        self._reset_fit()
        self.gm = None
        self.output_width = None
    def _transform(self, val):
        if not self.fit_complete:
            raise ValueError("GMMClusterTransformer has not been fit yet.")
        data = np.array([val])
        transformed = self.gm.transform(data)

        #  Converts the transformed data to the appropriate output format.
        output = [0] * self.output_width
        # If val is non-zero, use GMM transform. 
        if val != 0:
          output[0] = transformed[0][0]
          output[transformed[0][1].astype(int) + 1] = 1.0
          # Set one-hot columns for non-zero
          output[self.output_width-1] = 1.0
        else:
          # Set one-hot columns for zero
          output[self.output_width-2] = 1.0
          
        return tuple(output)      

    def _inverse_transform(self, val, sigmas=None):
        if not self.fit_complete:
            raise ValueError("GMMClusterTransformer has not been fit yet.")

        # Use argmax on the last two elements to determine if this is a zero.
        # If so just return 0
        # Else remove the last two elements and feed the rest to GMM
        if val[-2]>val[-1]:
          return 0
        else:
          val = val[:(len(val)-2)]

        # Reshape val to fit the number of columns
        data = np.array(val[:2])
        data[1] = np.argmax(val[1:])
        recovered_data = self.gm.reverse_transform(data.reshape((1,-1))) # A 1x1 pandas.DataFrame
        
        return recovered_data.values[0]
