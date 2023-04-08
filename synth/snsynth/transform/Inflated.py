
from snsynth.transform.definitions import ColumnType
from snsynth.transform.base import CachingColumnTransformer
from snsql.sql._mechanisms.approx_bounds import approx_bounds
from snsql.sql.privacy import Privacy
import numpy as np
import pandas as pd
from collections import Counter

from .numerical import FloatFormatter, ClusterBasedNormalizer

class InflatedNULLTransformer(CachingColumnTransformer):
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

        self.inflated_values = {0:1}
        self.idx_to_inflated_values = {1:0}
        
        self.output_width += 1 + len(self.inflated_values)
        if self.model_missing_values:
          self.output_width = self.output_width + 1 
          
    def _clear_fit(self):
        self._reset_fit()
        self.output_width = None
        
    def _transform(self, val):
        if not self.fit_complete:
            raise ValueError("GMMClusterTransformer has not been fit yet.")

        output = [0] * self.output_width
        
        # If val is missing, change it to 0 for easy modeling of GAN.
        # Note that if any missing val occur here, self.model_missing_values = T
        if val is None or (isinstance(val, float) and np.isnan(val)):
          output[:, 0] = 0
          # Set the missing column
          output[self.output_width-1] = 1.0
        elif val not in self.inflated_values:
          # If val is non-zero, use GMM transform. 
          output[0] = val
          # Set one-hot columns for non-zero
          output[1] = 1.0
        else:
          # Set one-hot columns for zero (or other inflated values)
          output[1 + self.inflated_values[val]] = 1.0
          
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
        inflat_indices = val[1:]
        inflat_type = np.argmax(inflat_indices)
        if inflat_type == len(inflat_indices)-1:
            print("Missing value!")
            return None
        elif 0 < inflat_type:
            return self.idx_to_inflated_values[inflat_type]
        else:
            val = val[:1]

        return val[0]

class InflatedGMMClusterTransformer(InflatedNULLTransformer):
    """Transforms a column of zero-inflated continuous values
    :param epsilon: The privacy budget to use to infer bounds, if none provided.
    :param nullable: If null values are expected, a second output will be generated indicating null.
    :param odometer: The optional odometer to use to track privacy budget.
    """
    def __init__(self, *, negative=True, epsilon=0.0, nullable=False, odometer=None,model_missing_values=True,inflat_thres=0.1):
        super().__init__(epsilon = epsilon, nullable = nullable, odometer = odometer, model_missing_values = model_missing_values, inflat_thres=inflat_thres)
    def _fit_finish(self):
        #print("Fit finish!")
        if self.epsilon is not None and self.epsilon > 0.0 and (self.lower is None or self.upper is None):
            if self.odometer is not None:
                self.odometer.spend(Privacy(epsilon=self.epsilon, delta=0.0))
            self.budget_spent.append(self.epsilon)

        # Make a one-column pd.DataFrame following the gm format
        # Always exclude missing values. Handles later.
        #self._fit_vals = [v for v in self._fit_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        self._fit_vals = [v for v in self._fit_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            
        data = pd.DataFrame({'column':self._fit_vals})
        self.gm = ClusterBasedNormalizer(model_missing_values=self.model_missing_values, max_clusters=min(len(data), 10))
        self.gm.fit(data, 'column')
        num_components = sum(self.gm.valid_component_indicator)
        self._fit_complete = True
        # First acount for GMM components:
        self.num_components = num_components
        self.output_width = 1 + num_components

        # count all values
        # Those with more frequency than self.thres are considered inflated
        # 02192023: for now only consider 0 and NA. We have problems with dominating values in the center.
        #occur_count, self.inflated_values = Counter(self._fit_vals), {}
        #for v in occur_count:
        #  if occur_count[v] / len(self._fit_vals) >= self.inflat_thres and v not in self.inflated_values:
        #    self.inflated_values[v] = len(self.inflated_values) + 1 # +1 since the first column means normal
        self.inflated_values = {0:1}
        self.idx_to_inflated_values = {1:0}
        # Add more one-hot columns at the end to indicate inflated vs not inflated
        # for example,  [1,0,0] means non-inflated and [0,1,0] means zero inflated, [0,0,1] means NA...
        self.output_width += 1 + len(self.inflated_values)
        if self.model_missing_values:
          self.output_width = self.output_width + 1 
    def _clear_fit(self):
        self._reset_fit()
        self.gm = None
        self.output_width = None
    def _transform(self, val):
        if not self.fit_complete:
            raise ValueError("GMMClusterTransformer has not been fit yet.")
        data = np.array([val])
        transformed = self.gm.transform(data)
        output = [0] * self.output_width
        
        # If val is missing, change it to 0 for easy modeling of GAN.
        # Note that if any missing val occur here, self.model_missing_values = T
        index = transformed[0][1].astype(int) + 1
        if val is None or (isinstance(val, float) and np.isnan(val)):
          output[0] = 0
          # Set the missing column
          output[self.output_width-1] = 1.0
        elif val not in self.inflated_values:
          # If val is non-zero, use GMM transform. 
          output[0] = transformed[0][0]
          # Set one-hot columns for non-zero
          output[transformed[0][1].astype(int) + 1] = 1.0
        else:
          # Set one-hot columns for zero (or other inflated values)
          output[self.num_components + 1 + self.inflated_values[val]] = 1.0
          
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
        inflat_indices = val[(self.num_components+1):]
        inflat_type = np.argmax(inflat_indices)
        if inflat_type == len(inflat_indices)-1:
            return None
        elif 0 < inflat_type:
            return self.idx_to_inflated_values[inflat_type]
        else:
            val = val[:self.num_components]

        data = np.array(val[:2])
        data[1] = np.argmax(val[1:])
        recovered_data = self.gm.reverse_transform(data.reshape((1,-1))) # A 1x1 pandas.DataFrame
        return recovered_data.values[0]
        
        
