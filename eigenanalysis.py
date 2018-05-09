
import matplotlib.pyplot as plt
import numpy as np
import datetime
from collections import namedtuple
from collections import Counter
import sys
from collections import defaultdict
import time
    
    
path = '/Users/Cassidy/Documents/Assignments/S2018/M462/Data/'
fileName = 'clusterData.txt'

stocks, Z, columnDict, days, dates  = getData(path, fileName)

n, nDays = np.shape(Z)
print('N days = ', nDays, '\nN stocks = ', n,'\nZ shape = ',n,'X', nDays,'\n')
nStocks = len(stocks)

nDays
nStocks
'''
C = np.corrcoef(X)
print(np.array_str(C,precision = 2))

C = np.cov(X)
print(np.array_str(C,precision = 2))
'''

C = (1/nDays)*np.matrix(Z)*np.matrix(Z.T)
print('Shape  = ', dim(C))
print(C[:2,:2])
# print(np.array_str(C.A, precision = 2))

values, vectors = np.linalg.eig(C)
values[:6]/len(values)

sum(values)
len(values)

''' Eigenvalues and cumulative eigenvalues '''

plt.figure(figsize=(10, 5))
plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
y = values
x = [i for i in range(np.shape(C)[0])]
yc = np.cumsum(y)

plt.scatter(x,y,s=12) 
plt.scatter(x,yc, color = 'red',s = 12)
plt.xlabel('Eigenvalue', fontsize=15)
plt.ylabel('Value', fontsize=15)
#plt.savefig('/home/brian/M462/Figures/eigenvalues.eps', format='eps',dpi = 1200)
plt.show()

''' First and second coefficients '''
''' Coefficients correspond to stocks '''


order = np.argsort([vectors[i,0] for i in range(np.shape(vectors)[0])])
for index in order:
    g.write(str(index)+','+str(vectors[index,0])+','+str(vectors[index,1])+','+stocks[index]+'\n')
g.close()    

plt.figure(figsize=(10, 5))
x = np.matrix.tolist(vectors[:,0])
y = np.matrix.tolist(vectors[:,1])
x = [float(coef) for coef in vectors[:,0] ]
y = [float(coef) for coef in vectors[:,1] ]
plt.scatter(x,y,s=12, color = 'black') 
for i in range(n):
    plt.annotate( stocks[i], (x[i] , y[i]) ,fontsize = 9)
    
plt.xlabel('Axis 1')
plt.ylabel('Axis 2')
plt.show()

''' Dominant trend over time : '''
prinComp = Z.T*vectors[:,:2]
dim(prinComp)

plt.figure(figsize=(10, 5))
plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
legend = ['First axis','Second axis']
plt.legend(loc='upper right')

x = [datetime.datetime.strptime(x, "%Y-%m-%d") for x in dates]
y = np.matrix.tolist(values[0]*prinComp[:,0])
plt.plot(x,y, color = 'black') 
y = np.matrix.tolist(values[1]*prinComp[:,1])
plt.plot(x,y, color = 'red') 
plt.legend(legend)
plt.xlabel('Day', fontsize=15)
plt.ylabel('Projection onto eigen-axis', fontsize=15)
#plt.savefig('/home/brian/M462/Figures/pc1.eps', format='eps',dpi = 1200)
plt.show()

'''  first axis - extreme observations  '''
order = np.argsort([vectors[i,0] for i in range(np.shape(vectors)[0])])
for index in order:
    print(index, vectors[index,0],stocks[index])

[stocks[i] for i in order]

plt.figure(figsize=(10, 5))
plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
x = [datetime.datetime.strptime(x, "%Y-%m-%d") for x in dates]
for i in range(3):
    y = Z[order[i],:]
    plt.plot(x,y, color = 'black') 
    y = Z[order[n-i-1],:]
    plt.plot(x,y, color = 'red') 

plt.xlabel('Day', fontsize=15)
plt.ylabel('Price', fontsize=15)
#plt.savefig('/home/brian/M462/Figures/6stocks.eps', format='eps',dpi = 1200)
plt.show()

''' second axis - extreme observations'''
order = np.argsort([vectors[i,1] for i in range(np.shape(vectors)[1])])
plt.figure(figsize=(10, 5))
x = [datetime.datetime.strptime(x, "%Y-%m-%d") for x in dates]
for i in range(3):
    y = Z[order[i],:]
    plt.plot(x,y, color = 'black') 
    y = Z[order[n-i-1],:]
    plt.plot(x,y, color = 'red') 

plt.xlabel('Day')
plt.ylabel('Price')
#plt.savefig('/home/brian/M467/Figures/centroids.eps', format='eps',dpi = 1200)
plt.show()

sys.exit()