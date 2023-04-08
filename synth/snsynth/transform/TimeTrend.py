from snsynth.transform.definitions import ColumnType
from snsynth.transform.base import CachingColumnTransformer
from snsql.sql._mechanisms.approx_bounds import approx_bounds
from snsql.sql.privacy import Privacy
import numpy as np
import pandas as pd
import warnings

import datetime
import math

from .numerical import FloatFormatter, ClusterBasedNormalizer

class TimeTrendTransformer(CachingColumnTransformer):
    """Transforms a column of values to scale between -1.0 and +1.0.
    :param negative: If True, scale between -1.0 and 1.0.  Otherwise, scale between 0.0 and 1.0.
    :param epsilon: The privacy budget to use to infer bounds, if none provided.
    :param nullable: If null values are expected, a second output will be generated indicating null.
    :param odometer: The optional odometer to use to track privacy budget.
    """
    def __init__(self, *, negative=True, epsilon=0.0, nullable=False, odometer=None,model_missing_values=True,input_len=5):
        #self.lower = lower
        #self.upper = upper
        self.epsilon = epsilon
        self.negative = negative
        self.budget_spent = []
        self.nullable = nullable
        self.odometer = odometer
        self.model_missing_values = model_missing_values
        self.gm = []
        self.output_width = None
        self.input_len = input_len # ymd_hm -> 5 elements
        self.output_width_each_level = [None] * self.input_len
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

    def convert_timestamp(self, timestamp):
        # Assume timestamp is the output of DatetimeTransformer:
        timestamp = timestamp * (24 * 60 * 60)
        dt = datetime.datetime.fromtimestamp(timestamp)
        return [dt.year, dt.month, dt.day, dt.hour, dt.minute]
        
    def is_leap_year(self, year):
        return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    def days_in_month(self, year, month):
        if month == 2:
            return 29 if is_leap_year(year) else 28
        elif month in [4, 6, 9, 11]:
            return 30
        else:
            return 31

    def recover_timestamp(self, year, month, day, hour, minute, second=0):
        days_since_1970 = 0
        for y in range(1970, year):
            days_since_1970 += 366 if self.is_leap_year(y) else 365
        for m in range(1, month):
            days_since_1970 += self.days_in_month(year, m)
        days_since_1970 += day - 1
        seconds_since_1970 = days_since_1970 * 86400 + hour * 3600 + minute * 60 + second
        return seconds_since_1970
        
    def _fit_finish(self):
        """Finish fitting the ClusterBasedNormalizer and set the number of output components.

        This transformer follows the Datetime transformer, which transforms datetime strings into timestampes in seconds.
        
        First, we extract year/month/day/hour/minute/weekday. Then we apply GMM for each of them.

        The function creates a one-column pandas DataFrame following the GMMClusterTransformer
        format, fits the ClusterBasedNormalizer, and sets the number of output components based
        on the number of valid components in the model.

        Returns:
            None
        """
        if self.epsilon is not None and self.epsilon > 0.0 and (self.lower is None or self.upper is None):
            if self.odometer is not None:
                self.odometer.spend(Privacy(epsilon=self.epsilon, delta=0.0))
            self.budget_spent.append(self.epsilon)

        # Make a one-column pd.DataFrame following the gm format
        self.output_width = 0
        self._fit_vals = [self.convert_timestamp(v) for v in self._fit_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        for i in range(self.input_len):        
            data = pd.DataFrame({'column':[elements[i] for elements in self._fit_vals]})
            self.gm.append(ClusterBasedNormalizer(model_missing_values=self.model_missing_values, max_clusters=min(len(data), 10)))
            self.gm[i].fit(data, 'column')

            num_components = sum(self.gm[i].valid_component_indicator)
            self.output_width += 1 + num_components
            self.output_width_each_level[i] = 1 + num_components
            
        self._fit_complete = True
        #print(self.output_width_each_level)

    def _clear_fit(self):
        self._reset_fit()
        self.gm = []
        self.output_width = None
        self.output_width_each_level = [None] * self.input_len
    def _transform(self, val):
        """Transform the given value to its normalized form and one-hot encode the component.

        Args:
            val (list or numpy.ndarray):
                The value to transform. It should be a list or numpy array with a length equal
                to the number of input columns.

        Returns:
            tuple:
                The transformed value in its normalized form and one-hot encoded component.

        Raises:
            ValueError:
                If the GMMClusterTransformer has not been fit yet.
        """
        if not self.fit_complete:
            raise ValueError("GMMClusterTransformer has not been fit yet.")
        data = np.array([val])
        combined_output = []
        
        for i in range(self.input_len):
            transformed = self.gm[i].transform(data)
        
            # Converts the transformed data to the appropriate output format.
            # The first column (ending in '.normalized') stays the same,
            # but the label encoded column (ending in '.component') is one hot encoded.
            output = [0] * self.output_width_each_level[i]
            output[0] = transformed[0][0]
            output[transformed[0][1].astype(int) + 1] = 1.0
            
            combined_output += output

        return tuple(combined_output)

    def _inverse_transform(self, val, sigmas=None):
        """Inverse transform the given value to its original scale.

        Args:
            val (a tuple):
                The value to inverse transform. It should be a list or numpy array
                with a length equal to the number of output columns.

            sigmas (None or float):
                The standard deviation used to scale the data during fitting. If None,
                the standard deviation used during fitting will be used.

        Returns:
            numpy.ndarray:
                The inverse transformed value in its original scale.

        Raises:
            ValueError:
                If the MinMaxTransformer has not been fit yet.
        """
        if not self.fit_complete:
            raise ValueError("TimeTrend transformer has not been fit yet.")
            
        data_start = 0
        recovered_timestamp = []
        for i in range(self.input_len):
            val_this_level = val[data_start:(data_start + self.output_width_each_level[i])] 
            #print(i,self.output_width_each_level[i], val_this_level)
            data = np.array(val_this_level[:2])
            data[1] = np.argmax(val_this_level[1:])
            recovered_data = self.gm[i].reverse_transform(data.reshape((1,-1))) # A 1x1 pandas.DataFrame
            recovered_timestamp.append(recovered_data.values[0])
            data_start += self.output_width_each_level[i]
        return int(self.recover_timestamp(*recovered_timestamp)) / (60 * 60 * 24)  

