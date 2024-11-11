import pandas as pd
import numpy as np

def perimiter(a, b, c):
    return a + b + c

def heron_formula(a, b, c):
    p = perimiter(a, b, c)/2
    S = np.sqrt(p*(p-a)*(p-c)*(p-b))

    h = 2*S/a

    return h

def create_samples():
    l = list(range(200))
    
    result = []

    for a in range(1, len(l)):
        for b in range(a+1, len(l)):
            for c in range(b+1, len(l)):
                if (a < b + c) and (b < a + c) and (c < a + b):
                    h = heron_formula(a, b, c)
                    result.append([a, b, c, h])
    
    return pd.DataFrame(result)

df = create_samples()

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

print(df)
X_train, X_test, y_train, y_test = train_test_split(df, df[:, 3], test_size=0.3, random_state=15)

# print(X_train.shape)
# print(y_train.shape)

# print(X_test.shape)
# print(y_test.shape)

mlp_model = MLPRegressor(hidden_layer_sizes=(128,64,32,16,8), activation="relu", solver='adam', alpha=1, batch_size=600, max_iter=1, random_state=25)
mlp_model.fit(X_train, y_train)

mlp_model.score(X_train, y_train)