import numpy as np
import pandas as pd

df = pd.read_pickle('../dataset/spam.p')
df = df[['log_char','has_bbm_pin','label']]