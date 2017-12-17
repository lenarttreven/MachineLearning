import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import scipy
from sklearn import svm



x1 = np.random.rand(50, 2)
x2 = np.random.rand(50, 2)
x2 = (0,1.5) + x2

x = np.concatenate((x1, x2), axis = 0)

y = np.array([1] * 50 + [-1] * 50)
df1 = pd.DataFrame(x)

df1.columns = ['1', '2']
df1['razred'] = y

fig = plt.figure()

width = 12
height = 9
plt.figure(figsize=(width, height))

first_class = df1.loc[df1['razred'] == 1]
second_class = df1.loc[df1['razred'] == -1]
fst_att, snd_att, _ = first_class.columns
plt.scatter(second_class[fst_att], second_class[snd_att], color = 'red', s=50, label='1')
plt.scatter(first_class[fst_att], first_class[snd_att], color = 'blue', s=50, label='-1')



t = np.arange(-0.2, 1.2, 0.01)
s1 = 0 * t + 1.5
s2 = 0 * t + 1
s3 = 0 * t + 1.25
plt.plot(t, s1, 'r--', linewidth=3.0, label= "<w,x> + b = 1")
plt.plot(t, s3, 'black', linewidth=3.0, label="<w,x> + b = 0")
plt.plot(t, s2, 'b--', linewidth=3.0, label="<w,x> + b = -1")
plt.legend()



plt.show()

