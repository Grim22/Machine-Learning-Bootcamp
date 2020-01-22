#import panda as pd 
import numpy as np
import sys
sys.path.insert(1, '../ex03')
from mylinearregression import MyLinearRegression
from csvreader import CsvReader
import matplotlib.pyplot as plt

# 1: import data

with CsvReader("are_blue_pills_magics.csv", header = True, skip_top = 0, skip_bottom = 0) as csv_file:
    data = np.array(csv_file.getdata(), float)
    Xpill = data[:, 1:2]
    Yscore = data[:, 2:3]

# 2: perform fit

Xpill_ = (Xpill - 3) / 6
tr = MyLinearRegression([0, 0])
print(tr.fit_(Xpill_, Yscore, 2, 1000))
#print(tr.predict_(Xpill))
#print(tr.cost_(Xpill, Yscore))

# 3: check the MSE

#linear_model1 = MyLinearRegression(np.array([[89.0], [-8]]))
#linear_model2 = MyLinearRegression(np.array([[89.0], [-6]]))
#Y_model1 = linear_model1.predict_(Xpill)
#Y_model2 = linear_model2.predict_(Xpill)
#print(linear_model1.mse_(Xpill, Yscore))
# 57.60304285714282
#print(linear_model2.mse_(Xpill, Yscore))
# 232.16344285714285

# 4: print plot

plt.plot(Xpill, Yscore, 'bo', label = "Strue")
plt.plot(Xpill, tr.predict_(Xpill_), 'g')
plt.plot(Xpill, tr.predict_(Xpill_), 'go', label ="Spredict")
plt.ylabel('space driving score')
plt.xlabel('Q of blue pill')
plt.grid()
plt.legend()
plt.show()