import numpy as np 
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../ex03')
sys.path.insert(1, '../ex05')
from mylinearregression import MyLinearRegression
from csvreader import CsvReader

# 1: import data

with CsvReader("spacecraft_data.csv", header = True, skip_top = 0, skip_bottom = 0) as csv_file:
    data = np.array(csv_file.getdata(), float)
    #Xage = data[:, 0:1]
    Xage = data[:, 0:3]
    Yprice = data[:, 3:4]

# 2: perform fit
print(Xage.shape)

#tr = MyLinearRegression([516, -1])
tr = MyLinearRegression([0., 0., 0., 0.])
tr2 = MyLinearRegression([8., -10., 7., -2.])
print(tr2.fit_(Xage, Yprice, 1e-4, 1000))
print(tr.normal_equation_(Xage, Yprice))
#print(tr.cost_(Xage, Yprice))
print(tr.mse_(Xage, Yprice))
print(tr2.mse_(Xage, Yprice))

# 3: print plot

plt.plot(Xage[:, 0:1], Yprice, 'bo', markersize = 4, label = "Strue")
#plt.plot(Xage, tr.predict_(Xage), 'g')
plt.plot(Xage[:, 0:1], tr.predict_(Xage), 'go', markersize = 3, label ="Spredict_NE")
plt.plot(Xage[:, 0:1], tr2.predict_(Xage), 'ro', markersize = 3, label ="Spredict_LGD")
plt.ylabel('sell price')
plt.xlabel('age')
plt.grid()
plt.legend()
plt.show()

  
