# LIBRERÍAS
import bnlearn as bn
import pandas as pd

df = pd.read_csv('./src/dataset18.csv')

print(df.shape)

model = bn.structure_learning.fit(df)
model = bn.independence_test(model, df)
G = bn.plot(model)