from snsynth.transform.definitions import ColumnType
from snsynth.transform.base import CachingColumnTransformer
from snsql.sql._mechanisms.approx_bounds import approx_bounds
from snsql.sql.privacy import Privacy
import numpy as np
import pandas as pd
import warnings

from snsynth.transform.numerical import FloatFormatter, ClusterBasedNormalizer # Modified transformers

class GMMClusterTransformer(CachingColumnTransformer):
    """Transforms a column of values to scale between -1.0 and +1.0.
    :param negative: If True, scale between -1.0 and 1.0.  Otherwise, scale between 0.0 and 1.0.
    :param epsilon: The privacy budget to use to infer bounds, if none provided.
    :param nullable: If null values are expected, a second output will be generated indicating null.
    :param odometer: The optional odometer to use to track privacy budget.
    """
    def __init__(self, *, negative=True, epsilon=0.0, nullable=False, odometer=None,model_missing_values=True):
        #self.lower = lower
        #self.upper = upper
        self.epsilon = epsilon
        self.negative = negative
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
        """Finish fitting the ClusterBasedNormalizer and set the number of output components.

        If the `epsilon` parameter is not None and is greater than 0, and either the `lower`
        or `upper` parameter is None, the budget is spent on the odometer and appended to the
        budget spent list.

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
        self._fit_vals = [v for v in self._fit_vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
        data = pd.DataFrame({'column':self._fit_vals})
        self.gm = ClusterBasedNormalizer(model_missing_values=self.model_missing_values, max_clusters=min(len(data), 10))
        self.gm.fit(data, 'column')

        num_components = sum(self.gm.valid_component_indicator)
        self._fit_complete = True
        self.output_width = 1 + num_components

    def _clear_fit(self):
        self._reset_fit()
        self.gm = None
        self.output_width = None
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
        transformed = self.gm.transform(data)
        
        # Converts the transformed data to the appropriate output format.
        # The first column (ending in '.normalized') stays the same,
        # but the label encoded column (ending in '.component') is one hot encoded.
        output = [0] * self.output_width
        output[0] = transformed[0][0]
        output[transformed[0][1].astype(int) + 1] = 1.0

        return tuple(output)

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
            raise ValueError("MinMaxTransformer has not been fit yet.")
        data = np.array(val[:2])
        data[1] = np.argmax(val[1:])
        recovered_data = self.gm.reverse_transform(data.reshape((1,-1))) # A 1x1 pandas.DataFrame
        
        #data = pd.DataFrame(np.array(val[:2]).reshape((1,-1)), columns=list(self.gm.get_output_sdtypes()))
        #data[data.columns[1]] = np.argmax(val[1:])
        #recovered_data = self.gm.reverse_transform(data) # A 1x1 pandas.DataFrame
        
        return recovered_data.values[0]

