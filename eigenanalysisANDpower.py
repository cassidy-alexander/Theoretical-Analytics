import matplotlib.pyplot as plt
import numpy as np
import datetime
from collections import namedtuple
from collections import Counter
import sys
from collections import defaultdict
import time

def dim(Matrix):
    return np.shape(Matrix)


sys.path.insert(0, '/Users/Cassidy/Document/Assignments/S2018/M462/Data')

def calculateDay(string):
    ''' Day 0 = Dec 31, 2006 '''
    year = int(string[:4]) - 2007                   
    return time.strptime(string, "%Y-%m-%d").tm_yday + year*365

def getData(path):
    fileName = 'clusterData.txt'
    #fileName = 'clusterDataLarge.txt'
    path += fileName
    
    g = open(path, 'r')
    variables = g.readline().strip('\n').split(',')
    stocks = variables[1:]
    
    dataDict = {}
    dateDict = {}
    data = g.read().split('\n')
    for record in data:
        lst = record.split(',')
        x = [float(s) for s in lst[1:]]
        try:
            1/len(x)
            day = calculateDay(lst[0])
            dateDict[day] = lst[0]
            dataDict[day] = x
        except(ZeroDivisionError):
            pass
    
    days = list(dataDict.keys())
    nDays = len(dataDict)
    nStocks = len(dataDict[days[0]]) 
    X = np.zeros(shape = (nStocks, nDays))
    
    
    ''' Fill the matrix with values. Warning: days are NOT necessarily ordered sequentially'''
    ''' Therefore, we put them in order, and then build the matrices in sequential order:'''
    orderedDays = sorted(days)
    dates = [0]*nDays
    for i, day in enumerate(orderedDays):
        X[:, i] = dataDict[day]
        dates[i] = dateDict[day]
    
   
    ''' Scale each stock series to have mean 0 and standard deviation 1 '''
    ''' Using numpy, the operation: matrix / y divides each column of M by the corresponding element in y '''

    print('X shape = ', np.shape(X))    
    
    xTranspose = X.T
    Z = ( (xTranspose - np.mean(xTranspose, axis=0)) / np.std(xTranspose, axis=0) ).T    
    columnDict = {stock:i for i, stock in enumerate(stocks)}    
    
    return stocks, Z, columnDict, orderedDays, dates

###################

def plot(dataToPlot, dates):
    ''' Code for plotting  a time series '''
    ''' Set up time variable for x-axis '''
    x = [datetime.datetime.strptime(x, "%Y-%m-%d") for x in dates]
    plt.figure(figsize=(10, 5))
    legend = []
    for i in range(len(dataToPlot)):
        y = dataToPlot[i]    
        plt.plot(x,y) 
        legend.append('Cluster '+str(i))
    plt.xlabel('Date')
    plt.ylabel('Closing price ($)')
    plt.legend(loc='upper right')
    plt.legend(legend)
    #plt.savefig('/home/brian/M467/Figures/centroids.eps', format='eps',dpi = 1200)
    plt.show()
    
    #######################3
    
def createDataSet(path, targetList, dataSet, firstDay, future):
    
    ''' Arguments passed to the function: '''
    ''' path is the directory from which to read data files'''
    ''' targetList is a list of stocks to be used in the subsequent analyses'''
    ''' dataSet is a namedtuple definition.  It's unlikely that this variable '''
    ''' needs to be changed'''
    ''' firstday is the first day in the created data set. Setting firstday =0 '''
    ''' starts the data set with Day 0 = Dec 31, 2006 '''
    ''' future = number of days forward to set the targets '''
    
    nPast = 3  
    string = 'Center Target'
    for i in range(nPast):
        string = 'p'+str((nPast -i))+' '+string
    print('\nVariable names:',string)    
    outputVars = namedtuple('variables', string)
     
    ''' Determine series lengths: '''
    ''' Determine days with data on all stocks '''
    
    targetDict = {}
    dateDict_0 = Counter() 
    
    ''' Process the data files '''
    ''' First record contains variable names'''
    ''' Last record is an empty string '''
    ''' dateDict_0 will contain the number of occurrences of each date across the files'''
    ''' We will use only those days for which there is a complete set of records '''
    ''' across all target stocks  '''
    
    fileList = os.listdir(path)
    print('\nProcessing data files', len(fileList), ' in ',path,'... first time ...')
    for i, fileName in enumerate(fileList):
        symbol = fileName.split('.')[0]
        if symbol in targetList :
            
            g = open(path+fileName, 'r')
            print('\r '+str(i)+' '+fileName, end = "")
            records = g.read().split('\n')
            try:
                firstDataRecord = records[1]
                firstDate = firstDataRecord[:10]  #  extract the date as a string 
                firstObsDay = calculateDay(firstDate)
                
                ''' check that the first day (firstObsDay) precedes firstDay '''
                if  firstObsDay >  firstDay :
                    createException = 1/0
                
                n = len(records) - 2
                targetDict[symbol] = (n,  firstDate)
                for record in records[1:n+1]:
                    date = record.split(',')[0]
                    if calculateDay(date) >= firstDay:
                        dateDict_0[date] += 1
            except(IndexError, ValueError, ZeroDivisionError):
                pass
    
    targetList = sorted(targetDict.keys())
    for symbol, value in targetDict.items():
        if value is None:
            targetList.remove(symbol)
    
    print('\nSymbol   First date  len')        
    for symbol in targetList:
        value = targetDict[symbol]
        print('  {:5} {:10} {:5}'.format(symbol.upper(),  value[1] , value[0]))
    print('Symbol   First date  len')        
    
    ''' Determine which dates have observations on at least 95% of days'''
    ''' These dates are used to create the data file '''
    ''' Drop those dates with too few observations '''
    ''' First step: make a copy of dateDict_0 '''
    
    print('\nChecking dates for missing observations ...')
    nTargets = len(targetList)
    dateDict= dict(dateDict_0)
    len(dateDict)
    dayList = []
    for k, v in dateDict_0.items():
        calcDay = calculateDay(k)
        if v < .95*nTargets or calcDay < firstDay:
            del dateDict[k]
            print('\rDropping ', calcDay,end = "")
        else:
            dayList.append(calcDay)
            
    ''' Process data files again and build data dictionary'''
    ''' Insure that the rows of the data matrices are chronologically ordered '''
    
    dayList = sorted(dayList)
    n = len(dayList)
    print('\nNumber of observation days (series length) = ', n)
    
    ''' Extract the observations and store in observationDict  '''
    ''' You can change the target variable from closing price below: '''
    observationDict = defaultdict(dict)
    variables = namedtuple('variables','Date Open High Low Close Volume OpenInt')
    dayDateDict = {}
    print('\nProcessing data files ... second time ...')
    for i, fileName in enumerate(fileList):
        symbol = fileName.split('.')[0]
        if symbol in targetList:
        
            g = open(path+fileName, 'r')
            print('\r' +str(i)+':\t'+path+fileName+'  ',end='')
            records = g.read().split('\n')
            n = len(records) - 2
            for record in records[1:n+1]:
                dayString = record.split(',')[0]
                day = calculateDay(dayString )
                if day in dayList:
                    dayDateDict[day] = dayString
                    v = variables(*record.split(','))
                    ''' name the target variable : '''
                    observationDict[symbol][day] = float(v.Close)
                    
                    
    ''' If there are missing days in the series for symbol, delete the symbol '''   
    ''' from targetList '''
    n = len(dateDict)
    for symbol, value in observationDict.items():    
        if len(value) != n:
            targetList.remove(symbol)
            nTargets = len(targetList)    
     
    print('\nNumber of targets = '+str(nTargets),'\n') 
    ''' Create weight vectors from which to compute the lag variables '''
    wts = np.zeros(shape = (20, nPast))
    a = .75
    for i in range(3):
        wts[:,i] = [a*(1-a)**k for k in range(19,-1,-1)]
        a /=2
        
    dataDict = defaultdict(dict)
    print('\n\nCreating data dictionaries ...')
    for i, symbol in enumerate(sorted(targetList)):
        ''' extract observations in chrono order: '''
        
        values = [observationDict[symbol][k] for k in dayList]    
        print('\rBuilding dict for '+str(i)+' '+symbol+'   ',end='')
        for i in range(20, n - future, 1):
            ''' compute differences from the current value: values[i] '''
            center = values[i]
            target = values[i+future] - center
            differences = [values[j]- center for j in range(i-20,i)]
            vals = [sum([diff*w for (diff, w) in zip(differences, wts[:,j])]) for j in range(3)]
            vals.extend([center, target] )
            dataDict[symbol][dayList[i]] = outputVars(*vals)
            
    days = dataDict[symbol].keys()
    
    ''' Completed: dict with days as keys \n'''    
    ''' Number of records '''
    N = len(days)       
    print('\nBuilding data matrices:\nN = ',N,
         ' n lags = ', nPast, 'First day = ', firstDay)
    
    ''' Build data matrices: Centers , Y, and X '''
    Centers = np.matrix(np.zeros(shape = (N, nTargets)))
    Y = np.matrix(np.zeros(shape = (N, nTargets)))
    X = np.matrix(np.zeros(shape = (N, nTargets * nPast)))
   
    for row, day in enumerate(days):
        
        for j, symbol in enumerate(targetList):
            variables = dataDict[symbol][day]
            if j == 0:
                predictors = list(dataDict[symbol][day][:nPast])
                targets = [dataDict[symbol][day].Target]
                centers = [dataDict[symbol][day].Center]
                
            else:
                predictors.extend(dataDict[symbol][day][:nPast])
                targets.append(dataDict[symbol][day].Target)
                centers.append(dataDict[symbol][day].Center)            
                
        X[row,:] = predictors
        Y[row,:] = targets
        Centers[row,:] = centers
        
    print('\n\nSummary ... targets are ',future,' days ahead of predictors :')  
    print('             Centers      |     Targets      |     Predictors    |     Summary')      
    print('          First    Last   |  (Differences)   |    (Differences)  |    Statistics')
    print(' Stock    value    value  |   First     Last |   First     Last  |   Mean     Std')
    for i in range(nTargets):
        print('  {:5} {:7.2f}  {:7.2f}  | {:7.2f}  {:7.2f} | {:7.2f}  {:7.2f}  |{:7.4f} {:7.4f} '.\
         format( targetList[i].upper(), Centers[0,i], Centers[N-1,i],
                Y[0, i], Y[N-1, i],
                X[0, i*nPast], X[N-1, i*nPast+(nPast-1)],
         round(np.mean(Y[:, i]),2),round(np.std(Y[:, i]),2) ))
    
    D = dataSet(X, Y, Centers, days, [dayDateDict[day] for day in days], [target.upper() for target in targetList])

    return D

##################################3




    
path = '/Users/Cassidy/Documents/Assignments/S2018/M462/Data/'
fileName = 'clusterData.txt'
print(fileName)
print(path)
stocks, Z, columnDict, days, dates  = getData(path)

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


fileName = '/Users/Cassidy/Documents/Assignments/S2018/M462Data/eigenCoefficients.txt'

g = open(fileName, 'w')

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



#################

nIter = 128
nVectors = 1
S = C

def powerMethod(S, nIter, nVectors): 
    p = dim(S)[0]
    vectors = np.matrix(np.zeros(shape = (p, nVectors)))
    values = np.zeros(shape = (nVectors, 1))
    M = S.copy()
    for i in range(nVectors):
        x = np.matrix(np.ones(shape = (p, 1) ))*np.sqrt(1/p) 
        for j in range(nIter):
            norm = np.linalg.norm(x)
            x = M*x/norm
        vectors[:,i] = x/np.linalg.norm(x)
        values[i] = vectors[:,i].T*S*vectors[:,i]
        M -= values[i][0] * vectors[:,i] * vectors[:,i].T

    pcAcct = np.cumsum(values)/sum(np.diag(S))
    printString = '' ''.join([str(round(100*pct,1)) for pct in pcAcct]) 
    print(printString)
    return values, vectors

val, vec = powerMethod(S, nIter, nVectors)
print(vec)
print(val)


values, vectors = np.linalg.eig(C)
values[:6]/len(values)

sum(values)
len(values)

cos = sum([a*b for a,b in zip(vectors[:,0], vec[:,0])])