from matplotlib import pyplot as plt
import numpy as np
randn = np.random.randn
from pandas import *

idx = Index(np.arange(1,11))
print idx
df = DataFrame(randn(10, 5)>0, index=idx, columns=['A', 'B', 'C', 'D', 'E'])
print df
vals = randn(10, 5)>0#np.around(df.values,2)
normal = plt.Normalize(0, 1)

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111, frameon=True, xticks=[], yticks=[])

the_table=plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns, colWidths = [0.03]*vals.shape[1], loc='center', cellColours=plt.cm.hot(normal(vals)))

plt.show()
