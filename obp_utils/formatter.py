import numpy as np
import pandas as pd

def comma_separator(x):  
    if isinstance(x, int):  
        return f"{x:,}"  
    return x  
  
def format_thousands(x):  
    return "{:,}".format(x)  