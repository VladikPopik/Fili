import numpy as np
import tensorflow as tf
import pandas as pd

def create_samples(a, b, N):
	X = np.linspace(a, b, N)
	Y = map(lambda x: x*np.sin(x), X)
	return 
