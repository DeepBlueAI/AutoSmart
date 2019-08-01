# -*- coding: utf-8 -*-
import pandas as pd

def get_downsampling_num(npos,nneg,sample_num,unbalanced_ratio,min_neg_pos_ratio=2):
  
    reverse = False
    ntol = npos + nneg
    if npos>nneg:
        reverse = True
        tmp = npos
        npos = nneg
        nneg = tmp

    max_sample_num = min(npos, nneg)*(unbalanced_ratio+1)
    if max_sample_num>sample_num:
        max_sample_num = sample_num

    if npos+nneg > max_sample_num:

        if nneg/npos <= min_neg_pos_ratio:
            pos_num = npos/ntol * max_sample_num
            neg_num = nneg/ntol * max_sample_num
            
        elif nneg/npos <= unbalanced_ratio:
            if npos > max_sample_num/(min_neg_pos_ratio+1):
                pos_num = max_sample_num/(min_neg_pos_ratio+1)
                neg_num = max_sample_num - pos_num            
            else:
                pos_num = npos
                neg_num = max_sample_num - pos_num

        elif nneg/npos > unbalanced_ratio:
            if npos > max_sample_num/(unbalanced_ratio+1):
                pos_num = max_sample_num/(unbalanced_ratio+1)
                neg_num = max_sample_num - pos_num  

            else:
                pos_num = npos
                neg_num = max_sample_num - npos

    else:
        neg_num = nneg
        pos_num = npos
    
    if neg_num/pos_num > unbalanced_ratio:
        neg_num = pos_num*unbalanced_ratio

    neg_num = int(neg_num)
    pos_num = int(pos_num)
    if reverse:
        return neg_num,pos_num

    return pos_num,neg_num

def sample(X,frac,seed,y=None): 
    if frac == 1:
        X = X.sample(frac=1,random_state=seed)
    elif frac > 1:
        mul = int(frac)
        frac = frac - int(frac)
        X_res = X.sample(frac=frac,random_state=seed)
        X = pd.concat([X] * mul + [X_res])
    else:
        X = X.sample(frac=frac,random_state=seed)
    
    if y is not None:
        y = y.loc[X.index]
        return X,y
    return X


def downsampling_num(y,max_sample_num):
    npos = (y==1).sum()
    nneg = (y==0).sum()
    
    
    min_num = min(npos,nneg)
    min_num = max(min_num,1000)
    
    if min_num < 8000:
        unbalanced_ratio = 10 - (min_num//1000) 
    else:
        unbalanced_ratio = 3
        
    pos_num,neg_num = get_downsampling_num(npos,nneg,max_sample_num,unbalanced_ratio)
    return pos_num,neg_num


def class_sample(X,y,pos_num,neg_num,seed=2019):
    
    npos = float((y == 1).sum())
    nneg = len(y) - npos
    
    pos_frac = pos_num / npos
    neg_frac = neg_num / nneg
    
    X_pos = X[y == 1]
    X_pos = sample(X_pos,pos_frac,seed)
    
    X_neg = X[y == 0]
    X_neg = sample(X_neg,neg_frac,seed)
    
    X = pd.concat([X_pos,X_neg])
    
    X,y = sample(X,1,seed,y)
    
    return X,y

def downsampling(X,y,max_sample_num,seed=2019):
    pos_num,neg_num = downsampling_num(y,max_sample_num)
    return class_sample(X,y,pos_num,neg_num,seed)

def class_sample_y(y,pos_num,neg_num,seed=2019):
    
    npos = float((y == 1).sum())
    nneg = len(y) - npos
    
    pos_frac = pos_num / npos
    neg_frac = neg_num / nneg
    
    y_pos = y[y == 1]
    y_pos = sample(y_pos,pos_frac,seed)
    
    y_neg = y[y == 0]
    y_neg = sample(y_neg,neg_frac,seed)
    
    y = pd.concat([y_pos,y_neg])
    
    y = sample(y,1,seed)
    
    return y

def downsampling_y(y,max_sample_num,seed=2019):
    pos_num,neg_num = downsampling_num(y,max_sample_num)
    y = class_sample_y(y,pos_num,neg_num,seed)
    return y
    

