"""
Keji's helping functions for Quan&Meth 2010
need numpy and scipy

readFile(name,tag=0)
initData(lineList)
group(tempData)

oneWay(dict_in)
propVar(oneWayDict)
pwrOneWay(vMean, var, vSize)
contrastOneWay(dict_in, coef)
leveneTest(dict_in)
welchADF(dict_in)
studentizedRange(rValue,dfError,errorRateFW)
"""
from __future__ import division
import numpy as np
import scipy.stats as st 

#to keep the relative path usable
from twisted.python.modules import getModule
moduleDirectory = getModule(__name__).filePath.parent()

def readFile(name, tag=0):
    "read file as table, tag=number of columns for group names"
    hFile=open(name)
    tempList=hFile.readlines()
    bigList=[map(int, l.split()[tag:]) for l in tempList]
    return bigList

def initData(lineList):
    "columns=groups, first_row=names, -1=empty_cell, no blank lines"
    nameList=lineList[0].split()
    dataList=[[int(a) for a in b.split()] for b in lineList[1:]]
    dataList=list(zip(*dataList))
    dataList=[np.array(filter(lambda b:b!=-1, a)) for a in dataList]
    dict_in=dict(zip(nameList, dataList))
    return dict_in

def group(tempData):
    "column 1 as group name and column 2 as data, return dict of ndnp.array" 
    data=list(zip(*tempData))
    nameList=list(set(data[0]))
    bigMat=[[c[1] for c in (filter(lambda a:a[0]==b, tempData))] for b in nameList]
    np.arrayList=[np.array(a) for a in bigMat]
    bigDict=dict(zip(nameList, np.arrayList))
    return bigDict

def oneWay(dict_in):
    "normal one way fixed effect ANOVA, argument rows as groups, dict output" 
    vMean=np.array([a.mean() for a in dict_in.values()])
    vSize=np.array([a.size for a in dict_in.values()])
    grandMean=((vMean*vSize).sum())/(vSize.sum())
    dfB=vSize.size-1
    dfW=vSize.sum()-dfB-1
    SSB=((vMean**2)*vSize).sum()-(grandMean**2)*vSize.sum()
    SSW=np.array([a.var()*a.size for a in dict_in.values()]).sum()
    SST=SSB+SSW
    MSB=SSB/dfB
    MSW=SSW/dfW
    fValue=MSB/MSW
    pValue=1-st.f.cdf(fValue,dfB,dfW)
    bigDict=dict({'SSW':SSW,'SSB':SSB,'SST':SST,'dfW':dfW,'dfB':dfB,'MSW':MSW,'MSB':MSB,'F':fValue,'p':pValue})
    return bigDict

def propVar(oneWayDict):
    "from oneWay output calculate R^2, shrunken R^2, omega^2"
    d=oneWayDict
    rSquare=d['SSB']/d['SST']
    rShrunken=1.0-((d['dfB']+d['dfW'])/d['dfW'])*(1.0-rSquare)
    omegaSquare=(d['SSB']-d['dfB']*d['MSW'])/(d['SST']+d['MSW'])
    proportion=dict({'R^2':rSquare,'sR^2':rShrunken,'omega^2':omegaSquare})
    return proportion

def pwrOneWay(vMean, var, vSize):
    "from vMean, var within, vSize calculate the power, assume alpha=0.05"
    grandMean=(vMean*vSize).sum()/vSize.sum()
    ncp=(vSize*(vMean-grandMean)**2).sum()/var
    critValue=st.f.ppf(0.95, vSize.size-1, vSize.sum()-vSize.size)
    power=1.0-st.ncf.cdf(critValue, vSize.size-1, vSize.sum()-vSize.size, ncp) 
    return power

def contrastOneWay(dict_in, coef):
    "from the data in dict_in, and coefficient dict in coef"
    zeroDict=dict_in.fromkeys(dict_in.keys(),0)
    zeroDict.update(coef)
    vCoef=np.array(zeroDict.values()) #give coef the right order, pad with zero
    vMean=np.array([a.mean() for a in dict_in.values()])
    vSize=np.array([a.size for a in dict_in.values()])
    grandMean=((vMean*vSize).sum())/(vSize.sum())
    dfB=vSize.size-1
    dfW=vSize.sum()-dfB-1
    SSW=np.array([a.var()*a.size for a in dict_in.values()]).sum()
    MSW=SSW/dfW
    numerator=(1.0*vMean*vCoef).sum()
    denominator=np.sqrt(MSW*(vCoef**2/vSize).sum())
    tValue=1.0*numerator/denominator
    contrast=1-st.t.cdf(tValue, dfW)
    return contrast

def leveneTest(dict_in):
    "levene's test using qms.oneWay"
    value_out=[np.abs(a-a.mean()) for a in dict_in.values()]
    dict_out=dict(zip(dict_in.keys(), value_out))
    return oneWay(dict_out)

def welchADF(dict_in):
    "welch adjusted degrees of freedom test"
    vMean=np.array([a.mean() for a in dict_in.values()])
    vSize=np.array([a.size for a in dict_in.values()])
    #a.size-1 because a.var() is not estimation
    vWeight=np.array([(a.size-1)/a.var() for a in dict_in.values()])
    adfMean=(vWeight*vMean).sum()/vWeight.sum()
    dfB=vSize.size-1.0
    lam=3.0*((1-vWeight/vWeight.sum())**2/(vSize-1)).sum()/(vSize.size**2-1)

    numerator=(vWeight*(vMean-adfMean)**2).sum()/dfB
    denominator=1.0+2*(dfB-1.0)*lam/3
    v_Value=numerator/denominator
    pValue=1-st.f.cdf(v_Value,dfB,1.0/lam)
    bigDict=dict({'V':v_Value,'p':pValue,'dfW':1/lam})
    return bigDict

def studentizedRange(rValue,dfError,errorRateFW):
    "studentized range table, r=number of means, r and df are int, -1 for    infinity, ER .01 or .05"
    lineList=moduleDirectory.child('tTRange').open().readlines()
    table=[[float(a) for a in b.split()] for b in lineList]
    dfList=range(5,21)
    dfList.extend([24,30,40,60,120,-1])
    indice=dfList.index(dfError)
    if errorRateFW==0.05:
        return table[indice*2][rValue-2]
    if errorRateFW==0.01:
        return table[indice*2+1][rValue-2]
    return
