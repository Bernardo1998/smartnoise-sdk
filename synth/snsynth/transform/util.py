"""CLI."""
import os
import pandas as pd
import json
import pandas as pd

import matplotlib as plt
from pandas.core.arrays import categorical

from snsynth.transform.type_map import TypeMap
from snsynth.transform.identity import IdentityTransformer
from snsynth.transform.label import LabelTransformer
from snsynth.transform.onehot import OneHotEncoder
from snsynth.transform.chain import ChainTransformer
from snsynth.transform.minmax import MinMaxTransformer
from snsynth.transform.log import LogTransformer
from snsynth.transform.datetime import DateTimeTransformer
from snsynth.transform.anonymization import AnonymizationTransformer

from transformers import *

# Map string to transformer objects
_REGISTERED_TRANSFORMERS = {'LabelTransformer':LabelTransformer,
                           'OneHotEncoder':OneHotEncoder,
                           'IdentityTransformer':IdentityTransformer,
                           'ChainedLabelBinary': (lambda:ChainTransformer([LabelTransformer(), BinaryEncoder()])),
                           'ChainedLabelOneHot':(lambda:ChainTransformer([LabelTransformer(), OneHotEncoder()])),
                           'ChainedTimeTrend':(lambda:ChainTransformer([DateTimeTransformer(), TimeTrendTransformer()])),
                           'GMMClusterTransformer':GMMClusterTransformer,
                           'MinMaxTransformer': MinMaxTransformer,
                           'ZeroInflatedGMMClusterTransformer':ZeroInflatedGMMClusterTransformer,
                           'ZeroInflatedNULLTransformer':ZeroInflatedNULLTransformer,
                           'LogTransformer':LogTransformer,
                           'LogBoundedTransformer':LogBoundedTransformer,
                           'InflatedGMMClusterTransformer':InflatedGMMClusterTransformer,
                           'InflatedNULLTransformer':InflatedNULLTransformer,
                           'DateTimeTransformer':(lambda:ChainTransformer([DateTimeTransformer(),MinMaxTransformer()])),
                           'AnonymizationTransformer':AnonymizationTransformer}

def fillMeta(data,meta, ordinal_as_continuous=True):
  '''
  ordinal_as_continuous: 
    meta will be used later in sdv metric. So by default we make it continuous following sdvmetric.
  '''
  inferred_column_info = TypeMap.infer_column_types(data)
  columns = inferred_column_info['columns']
  print(inferred_column_info)
  # TODO: handles other string
  for i,c in enumerate(columns):
    if c in meta:
      #print(c, " specified as ", meta[c], '!')
      continue
    if c in inferred_column_info['continuous_columns']:
      meta[c] = 'continuous'
    elif c in inferred_column_info['ordinal_columns']:
      meta[c] = 'continuous' if ordinal_as_continuous else 'ordinal'
    elif inferred_column_info['pii'][i] == 'datetime':
      meta[c] = 'datetime'
    else:
      meta[c] = 'categorical'
    print(c, " is ", meta[c])

  return inferred_column_info


def get_Transformers(args, data, meta):
    '''
    Get transformer based on model selection.
    '''

    # First determine column types. Update meta
    inferred_column_info = fillMeta(data,meta)

    # Use different mapping for categorical/continuous columns
    transformer = None
    
    if args.transformers is not None:
      # Transformer needs to be dictionary mapping each column type or colum name to a certain transformer
      # example: {'categorical':'OneHotEncoder'} or {'age':'GMMClusterTransformer'}
      # Then this function will extract corresponding attribute by name.
      # A name must be included in _REGISTERED_TRANSFORMERS.
      transformer = {}
      with open(args.transformers, 'r') as transformer_file:
        transformer_choices = json.load(transformer_file)

      key_type=None
      for c in meta:
        # All keys in transformer_choices must be the same type:
        # Either a column name or a column type (continuous/categorical/...)
        # Mixed types raises an error.
        if meta[c] in transformer_choices and key_type in ['column_types', None]:
          key_type = 'column_types'
        elif c in transformer_choices and key_type in ['column_names', None]:
          key_type = 'column_names'
        elif meta[c] not in transformer_choices and c not in transformer_choices:
          raise Exception("Column {} of {} type has no specified transformer!".format(c,meta[c]))
        else:
          raise Exception("Mixed key types in transformer configuration! All keys must be either a column name or a column type (continuous/categorical/...)")
        
        if key_type == 'column_types':
          choice = transformer_choices[meta[c]]
        elif key_type == 'column_names':
          choice = transformer_choices[c]

        transformer[c] = _REGISTERED_TRANSFORMERS[choice]()
        #print(c,' uses ', type(transformer[c]))
    return transformer, inferred_column_info
