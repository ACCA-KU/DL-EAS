import numpy as np
import pandas as pd
import math
import os
import sys
from rdkit import Chem
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def seconds_to_ddhhmmss(seconds):
    days = seconds // (60*60*24)
    seconds %= (60*60*24)
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02id:%02ih:%02im:%02is" % (days, hours, minutes, seconds)


def ddhhmmss_to_seconds(ddhhmmss):
    multiply_elements = [(60*60*24),(60*60),60,1]
    seconds = 0
    for i,j in zip(ddhhmmss.split(":"), multiply_elements):
        seconds += int(i[:2])*j
    return seconds


def mdir(path, override=False):

    if not (os.path.isdir(path)):
        os.makedirs(os.path.join(path))
    else:
        if override==False:
            path = path[:-1]+'_re/'

            path = mdir(path)
        else:
            pass
    return path


def make_fold(prop):
    N = ['train','val','test']
    M = ['adj','features','sol_adj','sol_features','result']
    if not(os.path.isdir('database')):
        os.makedirs(os.path.join('database'))
    if not(os.path.isdir('database/'+prop)):
        os.makedirs(os.path.join('database/'+prop))
    if not(os.path.isdir('database/'+prop+'/result')):
        os.makedirs(os.path.join('database/'+prop+'/result'))
        
    for n in N:
        if not(os.path.isdir('database/'+prop+'/'+n)):
            os.makedirs(os.path.join('database/'+prop+'/'+n))
        for m in M:
            if not(os.path.isdir('database/'+prop+'/'+n+'/'+m)):
                os.makedirs(os.path.join('database/'+prop+'/'+n+'/'+m))       
                            

def deNormalization(data,prop):
    Norm_data = open("database/"+prop+"/Normdata.txt","r").read()
    Norm_data = eval(Norm_data)
    df = pd.DataFrame(data)
    df = df.rename(columns={0:'abs',1:'emi',2:'life',3:'PLQY',4:'extin'})
    #Denormalize
    df['abs']=df['abs']*Norm_data['abs_std']+Norm_data['abs_mean']
    df['emi']=df['emi']*Norm_data['emi_std']+Norm_data['emi_mean']
    df['life']=df['life']*Norm_data['life_std']+Norm_data['life_mean']
    df['PLQY']=df['PLQY']*Norm_data['PLQY_std']+Norm_data['PLQY_mean']
    df['extin']=df['extin']*Norm_data['extin_std']+Norm_data['extin_mean']
    df = np.array(df)
    return df      



def cal_loss_mse(prop,pred):
    loss = np.mean((prop - pred) ** 2)
    return loss


def cal_loss_mse_val(prop,pred,i):
    prop = pd.DataFrame(prop[:,i])
    prop = prop.rename(columns={0:'prop'})
    pred = pd.DataFrame(pred[:,i])
    pred = pred.rename(columns={0:'pred'})
    total = pd.concat([prop,pred],axis=1)
    total = total.dropna()
    prop = pd.DataFrame(total['prop'])
    pred = pd.DataFrame(total['pred'])
    prop = np.array(prop)
    pred = np.array(pred)
    loss = np.mean((prop - pred) ** 2)
    prop = None
    pred = None
    return loss


def nan_delete(prop,pred,i):
    prop = pd.DataFrame(prop[:,i])
    prop = prop.rename(columns={0:'prop'})
    pred = pd.DataFrame(pred[:,i])
    pred = pred.rename(columns={0:'pred'})
    total = pd.concat([prop,pred],axis=1)
    total = total.dropna()
    prop = pd.DataFrame(total['prop'])
    pred = pd.DataFrame(total['pred'])
    prop = np.array(prop)
    pred = np.array(pred)
    return prop, pred


def nolist(x):
    return eval(x)
