import re
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# --------------数据目录---------------
rootdir = os.getcwd()
rootfiles = os.listdir(rootdir)
datadir = os.getcwd() + '\\data'
datafiles = os.listdir(datadir)

# --------------全局变量---------------
class Data_style:
    def __init__(self, resolution, header_size, setpointv, num2regression, noise,startv):
        self.resolution = resolution # bias resolution
        self.header_size = header_size
        self.noise = noise
        self.setpointv = setpointv
        self.num2regression = num2regression
        self.startv = startv

nanonis_style = Data_style(resolution=0.001, header_size=17, setpointv = 0.0, num2regression =10, noise=2, startv=0.040)
# resolution是dIdV谱的电压步长；headersize=17是nanonis文件格式；setpointv是用来判断在setpoint前是否hop，如果hop就把该数据删除；num2regression是用于判断hop时线性拟合用的数据点数；
# noise目前没用到；startv是判断HS-LS的偏压值，取startv后5个值平均，用于判断hs-ls

# --------------载入数据---------------
def loaddata(filepath, style):

    cur_datapd = pd.read_csv(filepath,sep='\t',header=style.header_size)
    cur_datapd.iloc[:,1:]=cur_datapd.iloc[:,1:]*1e12
    # print(datapd.shape)
    cur_datapd = cur_datapd.loc[:,cur_datapd.iloc[-1,:]!=0]
    # cur_datapd.to_csv(rootdir+'\\data.csv')

    return cur_datapd

def analysis(cur_datapd, style):
    start_row = abs(int((style.startv - cur_datapd.iloc[0,0])/style.resolution))
    num_column = cur_datapd.shape[1]

    print('Current data curves:', (num_column-4)/3, 'Continue? 1/0')
    choice = input()
    if choice == '0':
        num_column = int(input('Use first ____ curves:'))*3+4
        cur_datapd = cur_datapd.iloc[:,:num_column]

    point_to_avg = 5
    i = 4
    avg = []
    while i < num_column - 1:
        avg.append(np.average(cur_datapd.iloc[start_row:start_row+point_to_avg - 1, i]))
        i += 3

    binsize = 50
    plt.figure(1)
    plt.hist(avg, bins=binsize)
    HL_setpoint = plt.ginput(1)[0][0]
    plt.close(1)

    HS = LS = DataFrame(columns=['V', 'I', 'dIdV-X', 'dIdV-Y'])
    HS['V'] = cur_datapd.iloc[:, 0]
    LS['V'] = cur_datapd.iloc[:, 0]
    # print(HS, LS)

    HSpd = cur_datapd.copy()
    LSpd = cur_datapd.copy()
    # print(HSpd)
    i = 4
    countHS = 0
    countLS = 0
    while i < num_column - 1:
        tmp = cur_datapd.iloc[:, i]
        # print(tmp)
        tmpv = cur_datapd.iloc[:,0]
        if abs(cur_datapd.iloc[start_row, i]) - abs(HL_setpoint) > 0:
            tag = ishopped(tmp, tmpv, nanonis_style, 1)
            # print('LS',cur_datapd.iloc[0, i], HL_setpoint)
            LSpd.drop(columns=cur_datapd.columns[[i, i + 1, i + 2]], inplace=True, axis=1)
            if tag:
                countHS += 1
                HSpd.drop(columns=cur_datapd.columns[[i, i + 1, i + 2]], inplace=True, axis=1)
        else:
            tag = ishopped(tmp, tmpv, nanonis_style, 0)
            # print('HS',cur_datapd.iloc[0, i],HL_setpoint)
            HSpd.drop(columns=cur_datapd.columns[[i, i + 1, i + 2]], inplace=True, axis=1)
            if tag:
                countLS += 1
                LSpd.drop(columns=cur_datapd.columns[[i, i + 1, i + 2]], inplace=True, axis=1)
            # print(HSpd)
        i += 3

    # print(HSpd.shape[1])
    # print(HSpd.loc[:, HSpd.columns[np.arange(4, HSpd.shape[1], 3)]])

    tmpI = []
    tmpX = []
    tmpY = []
    for i in range(HSpd.shape[0]):
        tmpI.append(HSpd.loc[i, HSpd.columns[np.arange(4, HSpd.shape[1], 3)]].mean())
        tmpX.append(HSpd.loc[i, HSpd.columns[np.arange(5, HSpd.shape[1], 3)]].mean())
        tmpY.append(HSpd.loc[i, HSpd.columns[np.arange(6, HSpd.shape[1], 3)]].mean())

    HSpd.iloc[:, 1] = tmpI
    HSpd.iloc[:, 2] = tmpX
    HSpd.iloc[:, 3] = tmpY

    tmpI = []
    tmpX = []
    tmpY = []
    for i in range(LSpd.shape[0]):
        tmpI.append(LSpd.loc[i, LSpd.columns[np.arange(4, LSpd.shape[1], 3)]].mean())
        tmpX.append(LSpd.loc[i, LSpd.columns[np.arange(5, LSpd.shape[1], 3)]].mean())
        tmpY.append(LSpd.loc[i, LSpd.columns[np.arange(6, LSpd.shape[1], 3)]].mean())

    LSpd.iloc[:, 1] = tmpI
    LSpd.iloc[:, 2] = tmpX
    LSpd.iloc[:, 3] = tmpY

    LS_count = (LSpd.shape[1] - 4) / 3
    HS_count = (HSpd.shape[1] - 4) / 3

    print(str(countLS)+' from '+str(LS_count+countLS)+' LS starting curves has been deleted.')
    print(str(countHS) + ' from ' + str(HS_count+countHS) + ' HS starting curves has been deleted.')

    return HSpd,LSpd,HS_count,LS_count

def savedata(HSpd, LSpd, HS_count,LS_count,i):

    oldname = HSpd.columns[[0, 1, 2, 3]].tolist()
    newname = [i[-9:-4] + '-HS-Bias(V)', i[-9:-4] + '-HS-I(pA)', i[-9:-4] + '-HS-dI/dV-X(pA)',
               i[-9:-4] + '-HS-dI/dV-Y(pA)']
    namedict = dict(zip(oldname, newname))
    # print(namedict)
    HSpd.rename(columns=namedict, inplace=True)
    # print(HSpd)
    oldname = LSpd.columns[[0, 1, 2, 3]].tolist()
    newname = [i[-9:-4] + '-LS-Bias(V)', i[-9:-4] + '-LS-I(pA)', i[-9:-4] + '-LS-dI/dV-X(pA)',
               i[-9:-4] + '-LS-dI/dV-Y(pA)']
    namedict = dict(zip(oldname, newname))
    # print(namedict)
    LSpd.rename(columns=namedict, inplace=True)
    result = DataFrame()
    result = pd.concat([result, HSpd.iloc[:, :4]], sort=False, axis=1)
    result = pd.concat([result, LSpd.iloc[:, :4]], sort=False, axis=1)
    stat = pd.DataFrame([{'Name': i[-9:-4], 'HS_count': HS_count, 'LS_count': LS_count}])

    return result, stat

def ishopped(dataser, bias, fstyle, hstag):
    global fignum
    resolution = fstyle.resolution
    setv = fstyle.setpointv
    num_rows = abs(int((setv - bias[0]) / resolution))
    xfit = np.array(bias[0:fstyle.num2regression]).reshape(-1,1)
    yfit = dataser[0:fstyle.num2regression]
    model = LinearRegression()
    model = model.fit(xfit, yfit)
    x = np.array(bias[0:num_rows]).reshape(-1,1)
    y = dataser[0:num_rows]
    # plt.show()
    if hstag == 0:
        for i in range(num_rows):
            normcur = model.predict([[bias[i]]])[0] * 2.5
            # print(normcur, dataser[i])
            if abs(dataser[i]) > abs(normcur):
                # print('LS hopped')
                plt.plot(x, y)
                plt.plot(x, model.predict(x))
                plt.savefig(rootdir+'\\fig\\''LS-hop-'+str(fignum) +'.png')
                plt.clf()
                fignum += 1
                return True
        # print('LS',count,num_rows)
    elif hstag == 1:
        for i in range(num_rows):
            normcur = model.predict([[bias[i]]])[0] * 0.4
            # print(normcur,dataser[i])
            if abs(dataser[i]) < abs(normcur):
                # print('HS hopped')
                plt.plot(x, y)
                plt.plot(x, model.predict(x))
                plt.savefig(rootdir + '\\fig\\' + 'HS-hop-'+str(fignum) +'.png')
                plt.clf()
                fignum += 1
                return True
    plt.plot(x, y)
    plt.plot(x, model.predict(x))
    plt.savefig(rootdir + '\\fig\\' + 'Not-hop-'+str(fignum) +'.png')
    plt.clf()
    fignum += 1
        # print('HS',count,num_rows)
    # print(unhopped/num_rows)

    # percent = count / num_rows
    # print(percent)
    return False


# --------------数据处理---------------


# --------------main---------------
result = DataFrame()
stat = DataFrame()

num_files = len(datafiles)
# sp = input('setpoint(mV):')
# nanonis_style.setpointv = float(sp)/1000
fignum = 0

if num_files == 1:
    print('You are using one combined file.')
    filepath = datadir+'\\'+datafiles[0]
    HSpd, LSpd, HS_count, LS_count = analysis(loaddata(filepath,nanonis_style),nanonis_style)
    result, stat = savedata(HSpd,LSpd,HS_count,LS_count,datafiles[0])
    result.to_csv(rootdir + '\\result_average_' + datafiles[0][-9:-4] +'_combined' + '.csv')
    stat.to_csv(rootdir + '\\result_count_' + datafiles[0][-9:-4] + '_combined' + '.csv')
else:
    print('You are using many files to combine into one.')
    combined_pd = DataFrame()
    count = 1
    for i in datafiles:
        filepath = datadir + '\\' + i
        tmp = loaddata(filepath,nanonis_style)
        num_col = tmp.shape[1]
        # print(num_col)
        if num_col < 5:
            oldname = tmp.columns[[1, 2, 3]].tolist()
            newname = []
            for name in oldname:
                newname.append(name + ' ['+i[-9:-4] +']')
            # print(newname)
            namedict = dict(zip(oldname, newname))
            tmp.rename(columns=namedict, inplace=True)
            if count == 1:
                combined_pd = tmp
            else:
                combined_pd = pd.concat([combined_pd, tmp.iloc[:, 1:4]], sort=False, axis=1)
            count += 1
        else:
            if count == 1:
                combined_pd = pd.concat([combined_pd, tmp.iloc[:,0]], sort=False, axis=1)
            j = 4
            while j < num_col:
                oldname = tmp.columns[[j, j + 1, j + 2]].tolist()
                newname = []
                for name in oldname:
                    newname.append(name + ' [' + i[-9:-4] + ']' + '-' + str(j))
                namedict = dict(zip(oldname, newname))
                # print(newname)
                tmp.rename(columns=namedict, inplace=True)

                combined_pd = pd.concat([combined_pd, tmp.iloc[:, j:j + 3]], sort=False, axis=1)
                j += 3

            count +=1

    # print(combined_pd)
    # combined_pd.to_csv(rootdir + '\\data.csv')
    tmpI = []
    tmpX = []
    tmpY = []
    for i in range(combined_pd.shape[0]):
        tmpI.append(combined_pd.loc[i, combined_pd.columns[np.arange(1, combined_pd.shape[1], 3)]].mean())
        tmpX.append(combined_pd.loc[i, combined_pd.columns[np.arange(2, combined_pd.shape[1], 3)]].mean())
        tmpY.append(combined_pd.loc[i, combined_pd.columns[np.arange(3, combined_pd.shape[1], 3)]].mean())

    combined_pd.insert(loc=1, column='Current (A) AVG', value = tmpI)
    combined_pd.insert(loc=2, column='LI Demod 1 X (A) AVG', value=tmpX)
    combined_pd.insert(loc=3, column='LI Demod 1 Y (A) AVG', value=tmpY)
    # combined_pd.to_csv(rootdir + '\\data2.csv')

    # print(analysis(combined_pd))
    HSpd,LSpd,HS_count,LS_count = analysis(combined_pd,nanonis_style)
    result, stat = savedata(HSpd,LSpd,HS_count,LS_count, datafiles[0])
    result.to_csv(rootdir + '\\result_average_' + datafiles[0][-9:-4] + '_combined' + '.csv')
    stat.to_csv(rootdir + '\\result_count_' + datafiles[0][-9:-4] + '_combined' + '.csv')



    # print()




