# -*- coding: utf-8 -*-
import numpy as np
from cython cimport boundscheck,wraparound




@boundscheck(False) 
@wraparound(False)
def pre_tuple_encode_func(int[:] muldata, int[:] muldatalens,K):
    cdef:
        int index = 0
        int i,j,N = muldatalens.shape[0]
        int les
#        list tmp = []
        int c_K = K
        dict map_dict = {}
        int ids = 0
    
    
    ans = np.zeros( N ,dtype=np.float )
    
    for i in range(N):
        les = muldatalens[i]
        if les == 0:
            ans[i] = np.nan 
        else:
            tmp = []
            if c_K > les:
                for j in range(index,index+les):
                    tmp.append(muldata[j])
            else:
                for j in range(index,index+c_K):
                    tmp.append(muldata[j])
            
            thash = tuple( tmp )
            if thash in map_dict:
                ans[i] =  map_dict[ thash ] 
            else:
                map_dict[ thash ] = ids
                ans[i] = ids
                ids += 1
                
        index += les
    
    return ans
        

@boundscheck(False) 
@wraparound(False)
def post_tuple_encode_func(int[:] muldata, int[:] muldatalens,K):
    cdef:
        int index = 0
        int i,j,N = muldatalens.shape[0]
        int les
#        list tmp = []
        int c_K = K
        dict map_dict = {}
        int ids = 0
    
    
    ans = np.zeros( N ,dtype=np.float )
    
    for i in range(N):
        les = muldatalens[i]
        if les == 0:
            ans[i] = np.nan 
        else:
            tmp = []
            if c_K > les:
                for j in range(index,index+les):
                    tmp.append(muldata[j])
            else:
                for j in range(index+les-c_K,index+les):
                    tmp.append(muldata[j])
            
            thash = tuple( tmp )
            if thash in map_dict:
                ans[i] =  map_dict[ thash ] 
            else:
                map_dict[ thash ] = ids
                ans[i] = ids
                ids += 1
                
        index += les
    
    return ans


@boundscheck(False) 
@wraparound(False)
def tuple_encode_func_1(int[:] muldata, int[:] muldatalens):
    cdef:
        int index = 0
        int i,j,N = muldatalens.shape[0]
        int les
        dict map_dict = {}
        int ids = 1
    
    
    ans = np.zeros( N ,dtype=np.float )
    
    for i in range(N):
        les = muldatalens[i]
        if les == 0:
            ans[i] = np.nan 
        else:
            tmp = []
            for j in range(index,index+les):
                tmp.append(muldata[j])
            
            thash = tuple( tmp )
            if thash in map_dict:
                ans[i] =  map_dict[ thash ] 
            else:
                map_dict[ thash ] = ids
                ans[i] = ids
                ids += 1
                
        index += les
    
    return ans

#@boundscheck(False) 
#@wraparound(False)
#def tuple_encode_func_2(vals):
#    cdef:
#        int idx,N = vals.shape[0]
#        dict map_dict = {}
#        int ids = 0
#    
#    ans = np.zeros( N ,dtype=np.float )
#    
#    for idx in range(N):
#        i = vals[idx]
#        if type(i)==float or i==():
#            ans[idx] = np.nan
#        else:
#            if i in map_dict:
#                ans[idx] = map_dict[ i ]
#            else:
#                map_dict[ i ] = ids
#                ans[idx] = ids
#                ids += 1
#    
#    return ans


      
@boundscheck(False) 
@wraparound(False)
def cat_in_multi(  int[:] muldata, int[:] muldatalens, int[:] catdata ):
    cdef:
        int index = 0
        int i,j,N = muldatalens.shape[0]
        int les
        int cat
        int flag
#        list ans = []
        
    ans = np.zeros( N ,dtype=np.int8 )
        
    for i in range(N):
        les = muldatalens[i]
        flag = 0
        cat = catdata[ i ]
        for j in range(index,index+les):
            if muldata[j] == cat:
                flag = 1
                break
            
        if flag :
            ans[i] = 1
        else:
            ans[i] = 0
            
        index += les
    return ans

@boundscheck(False) 
@wraparound(False)
def cat_rank_multi(  int[:] muldata, int[:] muldatalens, int[:] catdata ):
    cdef:
        int index = 0
        int i,j,N = muldatalens.shape[0]
        int les
        int cat
        int flag
#        list ans = []
        
    ans = np.zeros( N ,dtype=np.int16 )
        
    for i in range(N):
        les = muldatalens[i]
        flag = 0
        cat = catdata[ i ]
        for j in range(index,index+les):
            if muldata[j] == cat:
                flag = j-index+1
                break
        ans[i] = flag     
        index += les
    return ans


@boundscheck(False) 
@wraparound(False)
def cat_frank_multi(  int[:] muldata, int[:] muldatalens, int[:] catdata ):
    cdef:
        int index = 0
        int i,j,N = muldatalens.shape[0]
        int les
        int cat
        int flag
#        list ans = []
        
    ans = np.zeros( N ,dtype=np.int16 )
        
    for i in range(N):
        les = muldatalens[i]
        flag = 0
        cat = catdata[ i ]
        for j in range(index,index+les):
            if muldata[j] == cat:
                flag = index+les - j
                break
        ans[i] = flag     
        index += les
    return ans


@boundscheck(False) 
@wraparound(False)
def get_need_data(  vals ):
    cdef:
        int idx,N = vals.shape[0]
        list datas = []
        list datalen = []
        
    for idx in range(N):
        i = vals[idx]
        if type(i) == float:
            datalen.append( 0 )
        else:
            datas.extend( i )
            datalen.append( len(i) )
        
    return datas,datalen




@boundscheck(False) 
@wraparound(False)
def mscat_fit(vals ):
    cdef:
        set ans = set()
        int idx,N = vals.shape[0]
        
    for idx in range(N):
        val = vals[idx]
        if type(val) == float:
            continue
        ans.update( val.split(',') )
        
    return ans

@boundscheck(False) 
@wraparound(False)
def mscat_trans(vals,cats):
    cdef:
        dict cat2index = {index: i + 1 for i,index in enumerate(cats)}
        list ans = []
        int idx,N = vals.shape[0]
        list tmp = []
    
    
    for idx in range(N):
        val = vals[idx]
        if type(val) == float:
            ans.append( tuple() )
        else:
            tmp = []
            x = val.split(',')
            for i in x:
                tmp.append( cat2index[i] )
                
            ans.append( tuple( tmp ) )
         
    return ans


