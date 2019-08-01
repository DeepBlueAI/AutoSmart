# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
  

def downcast(series,accuracy_loss = True, min_float_type='float16'):
    if series.dtype == np.int64:
        ii8 = np.iinfo(np.int8)
        ii16 = np.iinfo(np.int16)
        ii32 = np.iinfo(np.int32)
        max_value = series.max()
        min_value = series.min()
        
        if max_value <= ii8.max and min_value >= ii8.min:
            return series.astype(np.int8)
        elif  max_value <= ii16.max and min_value >= ii16.min:
            return series.astype(np.int16)
        elif max_value <= ii32.max and min_value >= ii32.min:
            return series.astype(np.int32)
        else:
            return series
        
    elif series.dtype == np.float64:
        fi16 = np.finfo(np.float16)
        fi32 = np.finfo(np.float32)
        
        if accuracy_loss:
            max_value = series.max()
            min_value = series.min()
            if np.isnan(max_value):
                max_value = 0
            
            if np.isnan(min_value):
                min_value = 0
                
            if min_float_type=='float16' and max_value <= fi16.max and min_value >= fi16.min:
                return series.astype(np.float16)
            elif max_value <= fi32.max and min_value >= fi32.min:
                return series.astype(np.float32)
            else:
                return series
        else:
            tmp = series[~pd.isna(series)]
            if(len(tmp)==0):
                return series.astype(np.float16)
            
            if (tmp == tmp.astype(np.float16)).sum() == len(tmp):
                return series.astype(np.float16)
            elif (tmp == tmp.astype(np.float32)).sum() == len(tmp):
                return series.astype(np.float32)
           
            else:
                return series
            
    else:
        return series
    
def gen_segs_array(shape0,nseg):
    segs = np.zeros(shape0)
    block_size = int(shape0/nseg)+1
    for i in range(nseg):
        segs[i*block_size:(i+1)*block_size] = i
    return segs


def gen_segs_tuple(shape0,nseg):
    segs = []
    block_size = int(shape0/nseg)
    i = -1
    for i in range(nseg-1):
        segs.append( (i*block_size,(i+1)*block_size) )
    segs.append(((i+1)*block_size,shape0))
    return segs
    

def gen_segs_tuple_by_time_nseg(shape0,nseg,time_series):
    block_size = None
    if time_series is None:
        block_size = int(shape0/nseg)+1
    else:
        max_time = time_series.max().value
        min_time = time_series.min().value
        block_size = int( (max_time-min_time)/nseg )
    return block_size
    
def gen_combine_cats(df, cols):

    category = df[cols[0]].astype('float64')
    for col in cols[1:]:
        mx = df[col].max()
        category *= mx
        category += df[col]
    return category

def gen_segs_tuple_by_time_size(shape0,block_size,time_series):
    segs = []
    if time_series is None:
        nseg = int(shape0/block_size)
        block_size = int( shape0/nseg ) + 1
        for i in range(nseg):
            segs.append( (i*block_size,(i+1)*block_size) )
    else:
        max_time = time_series.max().value
        min_time = time_series.min().value
        nseg = int( (max_time-min_time)/block_size )
        if nseg == 0:
            nseg = 1
        block_size = int( (max_time-min_time)/nseg ) + 1
        t = time_series.reset_index(drop=True)
        t = t.astype('int64')
        
        
        for i in range(nseg):
            
            l_time = min_time + i*block_size
            r_time = min_time + (i+1)*block_size
            if i == nseg-1:
                r_time = max_time+1
            indexs = t[ (l_time<=t) & (t < r_time) ].index
            l_index = indexs[0]
            r_index = indexs[-1]+1
            segs.append( (l_index,r_index) )
            
    return segs
    

