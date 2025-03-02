import math
import json
import folium
import seaborn as sns

import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import skew
from utils import initialize
from scipy.stats import normaltest
from media_utils import display_image
from missingno import matrix as missing
from summarytools import dfSummary as summary

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)

%matplotlib inline
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['figure.dpi'] = 144

sns.set()
initialize()