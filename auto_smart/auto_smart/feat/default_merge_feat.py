# -*- coding: utf-8 -*-

from .merge_feat import O2O,M2O,O2M,M2M,TimeM2M,PreO2O,PreM2O,PreO2M,PreM2M,PreTimeM2M
from util import timeclass
import CONSTANT
import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from feat_context import FeatContext
import util
from data_tools import downcast
import gc
namespace = 'default'

class M2OJoin(M2O):
    def fit(self,U,V):
        pass

    @timeclass(cls='M2OJoin')
    def transform(self,U,V):
        v = V.data
        key = self.key
        v = v.set_index(key)
        new_cols = []
        col2type = {}
        col2block = {}
        for col in v.columns:
            feat_type = V.col2type[col]
            new_col = FeatContext.gen_merge_feat_name(namespace,self.__class__.__name__,col,feat_type,V.name)
            new_cols.append(new_col)
            col2type[new_col] = feat_type

            if col in V.col2block:
                block_id = V.col2block[col]
                col2block[new_col] = block_id

        v.columns = new_cols
        return v,col2type,col2block

    @timeclass(cls='M2OJoin')
    def fit_transform(self,U,V):
        return self.transform(U,V)

class M2MKeyCount(M2M):
    @timeclass(cls='M2MKeyCount')
    def fit(self,U,V):
        pass

    @timeclass(cls='M2MKeyCount')
    def transform(self,U,V):
        v = V.data
        key = self.key
        col2type = {}
        ss = v.groupby(key)[key].count()
        ss = downcast(ss)
        feat_type = CONSTANT.NUMERICAL_TYPE
        new_col = key+'_M2MKeyCount'
        new_col = FeatContext.gen_merge_feat_name(namespace,self.__class__.__name__,new_col,feat_type,V.name)
        ss.name = new_col
        col2type[new_col] = feat_type
        return pd.DataFrame(ss),col2type,{}

    @timeclass(cls='M2MKeyCount')
    def fit_transform(self,U,V):
        return self.transform(U,V)

class M2MNumMean(M2M):
    @timeclass(cls='M2MNumMean')
    def fit(self,U,V):
        pass

    @timeclass(cls='M2MNumMean')
    def transform(self,U,V):
        v = V.data
        key = self.key
        col2type = {}
        
        def func(df):
            key = df.columns[0]
            col = df.columns[1]
            df[col] = df[col].astype('float32')
            
            ss = df.groupby(key)[col].mean()
            ss = downcast(ss)
            return ss
        
        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(v[[key,col]]) for col in V.num_cols)
        if res:
            new_cols = []
            for col in V.num_cols:
                feat_type = CONSTANT.NUMERICAL_TYPE
                col = col+'_M2MNumMean'
                new_col = FeatContext.gen_merge_feat_name(namespace,self.__class__.__name__,col,feat_type,V.name)
                new_cols.append(new_col)
                col2type[new_col] = feat_type
            
            tmp = pd.concat(res,axis=1)
            tmp.columns = new_cols
            return tmp,col2type,{} 
        return pd.DataFrame(),col2type,{}

    @timeclass(cls='M2MNumMean')
    def fit_transform(self,U,V):
        return self.transform(U,V)

class TimeM2MnewLastData(M2M):
    @timeclass(cls='TimeM2MnewLastData')
    def fit(self,U,V):
        pass

    @timeclass(cls='TimeM2MnewLastData')
    def transform(self,U,V):
        key = self.key
        
        if U.key_time_col != V.key_time_col:
            return 
        
        key_time_col = V.key_time_col
        
        todo_cols = V.multi_cat_cols
        if not todo_cols:
            return 
        
        v = V.data[[V.key_time_col,key] + todo_cols]
        u = U.data[[U.key_time_col,key]]
        
        u_index = u.index
        u.reset_index(drop=True,inplace=True)
        col2type = {}
        col2block = {}
        
        u.index = -u.index-1
        v_large = pd.concat([v,u])
        v_large.sort_values(by=[key,key_time_col],inplace=True)
        
        symbol = 1
        key_diff = v_large[key].diff()
        for col in todo_cols:
            v_large[col].loc[key_diff!=0].replace(np.nan,symbol)
            
        new_cols = []
        for col in todo_cols:
            feat_type = CONSTANT.MULTI_CAT_TYPE
            new_col = FeatContext.gen_merge_feat_name(namespace,self.__class__.__name__,col,feat_type,V.name)
            new_cols.append(new_col)
            col2type[new_col] = feat_type
            if col in V.col2block:
                col2block[new_col] = V.col2block[col]
            
        def func(series):
            ss = series.fillna(method='ffill')
            ss = ss.replace(symbol,np.nan)
            return ss
        
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(v_large[col]) for col in todo_cols)
        if res:
            tmp = pd.concat(res,axis=1)
            del res
            gc.collect()
            
            tmp.columns = new_cols
            tmp = tmp.loc[tmp.index<0]
            tmp.index = -(tmp.index+1)
            
            tmp.sort_index(inplace=True)
            tmp.index = u_index
            del u_index
            gc.collect()
            U.data[new_cols] = tmp
            del tmp
            gc.collect()
            U.update_data(U.data,col2type,None,None,col2block,None)
            
    @timeclass(cls='TimeM2MnewLastData')
    def fit_transform(self,U,V):
        self.transform(U,V)
        
class M2MDataLast(TimeM2M):
    @timeclass(cls='M2MDataLast')
    def fit(self,U,V):
        pass

    @timeclass(cls='M2MDataLast')
    def transform(self,U,V):
        data = V.data
        key = self.key
        col2type = {}
        col2block = {}

        col_sets = []
        cols = list(data.columns)
        
        if key in cols:
            cols.remove(key)
        
        del_cols = []
        for col in cols:
            if col in V.col2type:
                if V.col2type[col] == CONSTANT.NUMERICAL_TYPE:
                    del_cols.append(col)
                    
        for col in del_cols:
            if col in cols:
                cols.remove(col)
        
        if len(cols)==0:
            return pd.DataFrame(),{},{}
        cols_len = 20
        cols_num = len(cols)
        if cols_num % cols_len == 0:
            blocks = int(cols_num / cols_len)
        else:
            blocks = int(cols_num / cols_len) + 1

        for i in range(blocks):
            col_t = []
            for j in range(i*cols_len,(i+1)*cols_len):
                if j < len(cols):
                    col_t.append(cols[j])
            col_sets.append(col_t)

        feats = []
        for col_set in col_sets:
            
            feats.append( data.groupby( key )[ col_set ].last() )
        if feats:
            df = pd.concat(feats,axis=1)
            
            new_cols = []
            for col in df.columns:
                feat_type = V.col2type[col]
                new_col = FeatContext.gen_merge_feat_name(namespace,self.__class__.__name__,col,feat_type,V.name)
                new_cols.append(new_col)
                col2type[new_col] = feat_type
    
    
                if col in V.col2block:
                    block_id = V.col2block[col]
                    col2block[new_col] = block_id
    
            df.columns = new_cols
            return df,col2type,col2block
        else:
            return pd.DataFrame(),{},{}

    @timeclass(cls='M2MDataLast')
    def fit_transform(self,U,V):
        self.fit(U,V)
        return self.transform(U,V)