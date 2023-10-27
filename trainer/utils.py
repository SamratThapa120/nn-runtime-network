import logging
from typing import Any

import pandas as pd
import numpy as np

class MetricsStore:
    def __init__(self):
        self.metrics = {}
        
    def get_store_key(self, epoch, key):
        return f"{epoch}-{key}"
    
    def __call__(self, epoch, key, value):
        self.metrics[self.get_store_key(epoch, key)] = value

    def to_dataframe(self):
        # Extract unique epochs and keys
        epochs = sorted(set(int(k.split('-')[0]) for k in self.metrics.keys()))
        keys = sorted(set(k.split('-')[1] for k in self.metrics.keys()))
        
        # Create an empty DataFrame with NaN values
        df = pd.DataFrame(index=epochs, columns=keys)
        
        # Populate the DataFrame with metrics
        for epoch in epochs:
            for key in keys:
                store_key = self.get_store_key(epoch, key)
                df.at[epoch, key] = self.metrics.get(store_key, np.nan)
        
        return df

    def get_metrics_by_epoch(self, epoch):
        if len(self.metrics)==0:
            return {}
        epoch_metrics = {}
        for k, v in self.metrics.items():
            e, key = k.split('-')
            e = int(e)
            if e == epoch:
                epoch_metrics[key] = v
        return epoch_metrics
    
    def get_metric_all(self, metric_name):
        epochs_for_metric = [int(k.split('-')[0]) for k, v in self.metrics.items() if k.split('-')[1] == metric_name]
        sorted_epochs = sorted(epochs_for_metric)
        values = [self.metrics[self.get_store_key(e, metric_name)] for e in sorted_epochs]
        return values

def setup_logger(logfile):
    # Initialize logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('Trainer')
    logger.setLevel(logging.INFO)

    # Create a file handler and set level to info
    handler = logging.FileHandler(logfile)
    handler.setLevel(logging.INFO)

    # Create a logging format

    # Add the handlers to the logger
    logger.addHandler(handler)

    # Add standard output (stdout) handler
    stdout_handler = logging.StreamHandler()
    logger.addHandler(stdout_handler)
    return logger