import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    load_data,
    apply_filters,
    compute_cohorts,
    compute_rfm,
    compute_clv_empirical,
    compute_clv_formula,
    simulate_scenarios,
    data_quality_report
)
