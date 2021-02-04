# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import ac
import CONSTANT
from data_tools import downcast
from joblib import Parallel, delayed
from util import timeclass
from feat_context import FeatContext

namespace = 'preprocess'

class Preprocessor:
    def __init__(self):
        pass
    
    def fit(self,ss):
        pass
    
    def transform(self,ss):
        pass

    def fit_transform(self,ss):
        pass

class GeneralPreprocessor(Preprocessor):
    def __init__(self):
        self.K = 5
        
    @timeclass(cls='GeneralPreprocessor')
    def transform(self,X):
    
        todo_list = X.multi_cat_cols
        if todo_list  != []:
            
            col2muldatas = {}
            col2muldatalens = {}
            
            data = X.data[todo_list]
            for col in todo_list:
                vals = data[col].values
                datas,datalen = ac.get_need_data(vals)
            
                if len(datalen) != data.shape[0]:
                    raise Exception('An error with data length happens!!')
                    
                col2muldatas[col] = np.array(datas,dtype='int64').astype(np.int32)
                col2muldatalens[col] = np.array(datalen,dtype='int32')

            data = X.data[todo_list]
            col2type = {}
            col2groupby = {}
            for col in data.columns:
                data[col] = ac.tuple_encode_func_1(col2muldatas[col],col2muldatalens[col])
            
            new_cols = []
            for col in todo_list:
                feat_type = CONSTANT.CATEGORY_TYPE
                new_col = col+'_MCEncode'
                new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
                new_cols.append(new_col)
                col2type[new_col] = feat_type
                col2groupby[new_col] = col
            
            data.columns = new_cols
            df = X.data
            for col in data.columns:
                df[col] = downcast(data[col],accuracy_loss=False)
            
            X.update_data(df,col2type,col2groupby)
            
            df = X.data
            index = df.index
            col2type = {}
            col2groupby = {}
            for col in todo_list:
                new_col = col+'_MCLenAsCat'
                feat_type = CONSTANT.CATEGORY_TYPE
                new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
                df[new_col] = downcast( pd.Series( col2muldatalens[col],index ),accuracy_loss=False)

                col2type[new_col] = feat_type
                col2groupby[new_col] = col
                
            X.update_data(df,col2type,col2groupby)

            todo_list = X.time_cols
            
            if todo_list != []:
                df = X.data
                col2type = {}
                for col in X.time_cols:
                    new_col = col+'_TimeNum'
                    feat_type = CONSTANT.NUMERICAL_TYPE
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
                    
                    ss = (df[col] - pd.to_datetime('1970-01-01')).dt.total_seconds()
                    ss[ss<0] = np.nan
                    min_time = ss.min()
                    ss = ss-min_time
        
                    df[new_col] = downcast(ss)
        
                    col2type[new_col] = feat_type
        
                if len(col2type) > 0:
                    X.update_data(df,col2type,None)
    
    @timeclass(cls='GeneralPreprocessor')
    def fit_transform(self,X):
        return self.transform(X)

class BinaryPreprocessor(Preprocessor):
    def __init__(self):
        self.col2cats = {}
    
    @timeclass(cls='BinaryPreprocessor')
    def fit(self,X):
        def func(ss):
            cats = pd.Categorical(ss).categories 
            return cats
        
        df = X.data
        todo_cols = X.binary_cols
        
        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[col]) for col in todo_cols)
        for col,cats in zip(todo_cols,res):
            self.col2cats[col] = cats
        
    @timeclass(cls='BinaryPreprocessor')
    def transform(self,X):
        
        def func(ss,cats):
            codes = pd.Categorical(ss,categories=cats).codes
            codes = codes.astype('float16')
            codes[codes==-1] = np.nan
            
            return codes
        
        df = X.data
        todo_cols = X.binary_cols
        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[col],self.col2cats[col]) for col in todo_cols)
        for col,codes in zip(todo_cols,res):
            df[col] = codes
            
    @timeclass(cls='BinaryPreprocessor') 
    def fit_transform(self,X):
        self.fit(X)
        self.transform(X)

class MSCatPreprocessor(Preprocessor):
    def __init__(self):
        self.cats = []
        
    def fit(self,ss):
        vals = ss.values
        
        ss = pd.Series( list(ac.mscat_fit(vals)) )
        
        if ss.name is None:
            ss.name = 'ss'
        
        cats = ss.dropna().drop_duplicates().values

        if len(self.cats) == 0:
            self.cats = sorted(list(cats))
        else:
            added_cats = sorted(set(cats) - set(self.cats))
            self.cats.extend(added_cats)

    def transform(self,ss,kind):

        if kind == CONSTANT.CATEGORY_TYPE:

            codes = pd.Categorical(ss,categories=self.cats).codes + CONSTANT.CAT_SHIFT
            codes = codes.astype('float')
            codes[codes==(CONSTANT.CAT_SHIFT-1)] = np.nan

            codes = downcast(codes,accuracy_loss=False)
            return codes
        else:
            codes = pd.Series( ac.mscat_trans(ss.values,self.cats) , index = ss.index )
            return codes
            
    def fit_transform(self,ss):
        return self.transform(ss)    
    
class NumPreprocessor(Preprocessor):
    def fit(self,X):
        pass
    
    def transform(self,X):
        df = X.data
        todo_cols = X.num_cols
        for col in todo_cols:
            df[col] = downcast(df[col])
    
    def fit_transform(self,X):
        return self.transform(X)
    
class UniquePreprocessor(Preprocessor):
    @timeclass(cls='UniquePreprocessor')
    def fit(self,X):
        def func(ss):
            length = len(ss.unique())
            if length <= 1:
                return True
            else:
                return False
            
        df = X.data
        todo_cols = X.cat_cols + X.multi_cat_cols + X.num_cols + X.time_cols + X.binary_cols
        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[col]) for col in todo_cols)
        
        drop_cols = []
        for col,unique in zip(todo_cols,res):
            if unique:
                drop_cols.append(col)
        
        self.drop_cols = drop_cols

    @timeclass(cls='UniquePreprocessor')
    def transform(self,X):
        X.drop_data(self.drop_cols)
    
    @timeclass(cls='UniquePreprocessor')
    def fit_transform(self,X):
        self.fit(X)
        self.transform(X)

class AllDiffPreprocessor(Preprocessor):
    @timeclass(cls='AllDiffPreprocessor')
    def fit(self,X):
        def func(ss):
            length = len(ss.unique())
            if length >= len(ss)-10:
                return True
            else:  
                return False
        
        df = X.data
        todo_cols = X.cat_cols
        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[col]) for col in todo_cols)
        
        drop_cols = []
        for col,all_diff in zip(todo_cols,res):
            if all_diff:
                drop_cols.append(col)
        
        self.drop_cols = drop_cols
        
    @timeclass(cls='AllDiffPreprocessor')
    def transform(self,X):
        X.drop_data(self.drop_cols)
    
    @timeclass(cls='AllDiffPreprocessor')
    def fit_transform(self,X):
        self.fit(X)
        self.transform(X)
