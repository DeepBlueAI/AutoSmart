# -*- coding: utf-8 -*-
from .feat import Feat
import CONSTANT
import pandas as pd
from util import timeclass,log
import numpy as np
from joblib import Parallel, delayed
from feat_context import FeatContext
import util
from data_tools import downcast
import gc
from data_tools import gen_segs_tuple,gen_combine_cats
import ac
namespace = 'default'

class ApartCatRecognize(Feat):
    @timeclass(cls='ApartCatRecognize')
    def fit(self,X,y):
        df = X.data
        apart_cols = []

        def func(ss):
            length = len(ss)//100
            part1 = ss.iloc[:length*49].dropna().drop_duplicates()
            part2 = ss.iloc[length*49:].dropna().drop_duplicates()
            union_len = len(pd.concat([part1,part2]).drop_duplicates())
            inter_len = len(part1) + len(part2) - union_len
            if union_len == 0:
                return True

            if inter_len/union_len <= 0.001:
                return True
            else:
                return False

        todo_cols = X.session_cols+X.user_cols+X.key_cols+X.cat_cols
        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[col]) for col in todo_cols)

        for col,apart in zip(todo_cols,res):
            if apart:
                apart_cols.append(col)

        self.apart_cat_cols = apart_cols


    @timeclass(cls='ApartCatRecognize')
    def transform(self,X):
        X.add_apart_cat_cols(self.apart_cat_cols)
        X.add_post_drop_cols(self.apart_cat_cols)
        log(f'apart_cat_cols:{self.apart_cat_cols}')

    @timeclass(cls='ApartCatRecognize')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)

class TimeNum(Feat):
    @timeclass(cls='TimeNum')
    def fit(self,X,y):
        pass

    @timeclass(cls='TimeNum')
    def transform(self,X):
        df = X.data
        col2type = {}
        for col in X.time_cols:
            new_col = col+'_TimeNum'
            feat_type = CONSTANT.NUMERICAL_TYPE
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
            
            ss = (df[col]-pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            ss[ss<0] = np.nan
            min_time = ss.min()
            ss = ss-min_time

            df[new_col] = downcast(ss)

            col2type[new_col] = feat_type

        if len(col2type) > 0:
            X.update_data(df,col2type,None)

    @timeclass(cls='TimeNum')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)

class CatSegCtrOrigin(Feat):
    @timeclass(cls='CatSegCtrOrigin')
    def fit(self,X,y):
        raise Exception("CatSegCtrOrigin doesn't support fit.")

    @timeclass(cls='CatSegCtrOrigin')
    def transform(self,X):
        if len(X.cat_cols) == 0:
            return

        df = X.data
        col2type = {}
        col2groupby = {}

        for col in self.todo_cols:

            new_col = col+'_CatSegCtrOrigin'
            feat_type = CONSTANT.NUMERICAL_TYPE
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)

            ss = df[col].map(self.catsegctr[col])
            df[new_col] = downcast(ss)
            col2type[new_col] = feat_type
            col2groupby[new_col] = col

        if len(col2type) > 0:
            X.update_data(df,col2type,col2groupby)

    @timeclass(cls='CatSegCtrOrigin')
    def fit_transform(self,X,y):
        if len(X.cat_cols) == 0:
            return

        self.catsegctr = {}
        df = X.data
        col2type = {}
        col2groupby = {}

        shape0 = df.shape[0]
        nsegs = 2
        groups_tuple = gen_segs_tuple(shape0,nsegs)

        label = CONSTANT.LABEL
        df[label] = y

        def func(df):
            col = df.columns[0]
            label = df.columns[1]
            info_pre = pd.Series()

            datas = []
            for i,block in enumerate(groups_tuple):
                left = block[0]
                right = block[1]
                data = df.iloc[left:right]
                ss1 = data.groupby( col ,sort=False)[label].sum()
                ss2 = data.groupby( col ,sort=False)[label].count()
                info_now = ss1/(ss2+100)
                datas.append(data[col].map(info_pre))

                info_pre_index = set(info_pre.index)
                info_now_index = set(info_now.index)
                info_all_index = list( info_pre_index | info_now_index )
                info_now_index_out = list( info_now_index - info_pre_index )
                info_pre_index_out = list( info_pre_index - info_now_index )
                info_index_in = list( info_now_index & info_pre_index )

                res = pd.Series( data = [0]*len(info_all_index),index = info_all_index )

                info_now_out = info_now.loc[ info_now_index_out ]
                info_now_in = info_now.loc[ info_index_in ]
                info_pre_out = info_pre.loc[ info_pre_index_out  ]
                info_pre_in = info_pre.loc[ info_index_in ]

                if i==0:
                    res += 1*info_now

                else:
                    res = res.add((0*info_pre_in) ,fill_value=0)
                    res = res.add((1*info_now_in) ,fill_value=0)
                    res = res.add(info_pre_out,fill_value=0)
                    res = res.add(info_now_out,fill_value=0)
                info_pre = res

            self.catsegctr[col] = info_pre
            ss = pd.concat( datas )
            ss = downcast(ss)
            return ss

            
        cat_cols = []
        for col in X.combine_cat_cols[:50]:
            if col in df.columns and col not in (X.user_cols+X.key_cols) :
                cat_cols.append(col)
        todo_cols = X.user_cols+X.key_cols+cat_cols
        log(f'cat cols {len(X.cat_cols)} combine_cat_cols {len(cat_cols)}')    
        
        todo_cols = X.get_not_apart_cat_cols(todo_cols)
        self.todo_cols = todo_cols

        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[[col,label]]) for col in todo_cols)

        df.drop(label,axis=1,inplace=True)
        if res:
            new_cols = []
            for col in self.todo_cols:
                new_col = col+'_CatSegCtrOrigin'
                feat_type = CONSTANT.NUMERICAL_TYPE
                new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
                new_cols.append(new_col)
                col2type[new_col] = feat_type
                col2groupby[new_col] = col

            for i,col in enumerate(new_cols):
                df[col] = res[i]

            X.update_data(df,col2type,col2groupby)

class KeysTimeDiffAndFuture(Feat):

    def fit(self,X,y):
        pass

    @timeclass(cls='KeysTimeDiffAndFuture')
    def transform(self,X):
        if X.key_time_col is None or len(X.key_cols)==0:
            return

        df = X.data
        col2type = {}
        key_time_col = X.key_time_col
        time_int_ss = df[key_time_col].astype('int64')/1000000

        todo_cols = X.key_cols + X.user_cols + X.session_cols

        new_cols = []
        for col in todo_cols:
            feat_type = CONSTANT.NUMERICAL_TYPE

            new_col = col + '_KeysTimeDiff'
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
            col2type[new_col] = feat_type
            new_cols.append(new_col)

            new_col = col + '_KeysTimeDiffFuture'
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
            col2type[new_col] = feat_type
            new_cols.append(new_col)



        def func(cat_ss,time_ss):
            df = pd.concat([cat_ss,time_ss],axis=1)
            cat_col = df.columns[0]
            time_col = df.columns[1]
            index = df.index
            df.reset_index(drop=True,inplace=True)
            df.sort_values([cat_col,time_col],inplace=True)

            time_ss = df[time_col].diff()
            cat_ss = df[cat_col].diff()
            time_ss[cat_ss!=0] = np.nan
            time_ss.sort_index(inplace=True)
            time_ss = downcast(time_ss)
            time_ss.index = index

            time_ss_future = df[time_col].diff(-1)
            cat_ss_future = df[cat_col].diff(-1)
            time_ss_future[cat_ss_future!=0] = np.nan
            time_ss_future.sort_index(inplace=True)
            time_ss_future = downcast(time_ss_future)
            time_ss_future.index = index

            sss = pd.concat([time_ss,time_ss_future],axis=1)
            return sss

        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[cat_col],time_int_ss) for cat_col in todo_cols)
        if res:
            tmp = pd.concat(res,axis=1)
            del res
            gc.collect()
            tmp.columns = new_cols
            df[new_cols] = tmp
            del tmp
            gc.collect()

            X.update_data(df,col2type,None)

    @timeclass(cls='KeysTimeDiffAndFuture')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)

class UserKeyTimeDiffAndFuture(Feat):
    def fit(self,X,y):
        pass

    @timeclass(cls='UserKeyTimeDiffAndFuture')
    def transform(self,X):
        if X.key_time_col is None or len(X.key_cols)==0:
            return

        df = X.data
        col2type = {}
        key_time_col = X.key_time_col
        time_int_ss = df[key_time_col].astype('int64')/1000000

        user_key_cols = []
        for user_col in X.user_cols:
            for key_col in X.key_cols:
                user_key_cols.append([user_col,key_col])

        user_key_series = []
        for i in range(len(user_key_cols)):
            series = gen_combine_cats(df, user_key_cols[i])
            user_key_series.append(series)


        new_cols = []
        for user_key_col in user_key_cols:
            feat_type = CONSTANT.NUMERICAL_TYPE
            new_col = '{}_{}_UserKeyTimeDiff'.format(user_key_col[0],user_key_col[1])
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
            new_cols.append(new_col)
            col2type[new_col] = feat_type

            new_col = '{}_{}_UserKeyTimeDiffFuture'.format(user_key_col[0],user_key_col[1])
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
            new_cols.append(new_col)
            col2type[new_col] = feat_type

        def func(cat_ss,time_ss):

            df = pd.concat([cat_ss,time_ss],axis=1)
            cat_col = df.columns[0]
            time_col = df.columns[1]
            index = df.index
            df.reset_index(drop=True,inplace=True)
            df.sort_values([cat_col,time_col],inplace=True)

            time_ss = df[time_col].diff()
            cat_ss = df[cat_col].diff()
            time_ss[cat_ss!=0] = np.nan
            time_ss.sort_index(inplace=True)
            time_ss = downcast(time_ss)
            time_ss.index = index

            time_ss_future = df[time_col].diff()
            cat_ss_future = df[cat_col].diff()
            time_ss_future[cat_ss_future!=0] = np.nan
            time_ss_future.sort_index(inplace=True)
            time_ss_future = downcast(time_ss_future)
            time_ss_future.index = index

            sss = pd.concat([time_ss,time_ss_future],axis=1)
            return sss
        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(series,time_int_ss) for series in user_key_series)
        if res:
            tmp = pd.concat(res,axis=1)
            tmp.columns = new_cols
            df = pd.concat([df,tmp],axis=1)
            X.update_data(df,col2type,None)

    @timeclass(cls='UserKeyTimeDiffAndFuture')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)


class KeyTimeDate(Feat):
    def fit(self,X,y):
        if X.key_time_col is None:
            return

        df = X.data
        col = X.key_time_col
        self.attrs = []
        for atr,nums in zip(['year','month','day','hour','weekday'],[2,12,28,24,7]):
            atr_ss = getattr(df[col].dt,atr)
            if atr_ss.nunique() > 1:
                self.attrs.append(atr)

    @timeclass(cls='KeyTimeDate')
    def transform(self,X):
        if X.key_time_col is None:
            return

        df = X.data
        col2type = {}
        col = X.key_time_col

        new_cols = []

        for atr in self.attrs:
            new_col = col+'_'+atr
            new_cols.append(new_col)
            df[new_col] = downcast(getattr(df[col].dt,atr),accuracy_loss=False)

        X.bin_cols.extend(new_cols)
        X.update_data(df,col2type,None)
        X.add_post_drop_cols(new_cols)

    @timeclass(cls='KeyTimeDate')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)

class TimeDate(Feat):
    def fit(self,X,y):
        if len(X.time_cols) == 0:
            return

        df = X.data
        time_cols = X.time_cols
        self.col2attrs = {col:[] for col in time_cols}

        for col in time_cols:
            attrs = self.col2attrs[col]
            for atr,nums in zip(['year','month','day','hour','weekday'],[2,12,28,24,7]):
                atr_ss = getattr(df[col].dt,atr)
                if atr_ss.nunique() >= nums:
                    attrs.append(atr)

    @timeclass(cls='TimeDate')
    def transform(self,X):
        if len(X.time_cols) == 0:
            return

        df = X.data
        col2type = {}
        time_cols = X.time_cols

        for col in time_cols:
            attrs = self.col2attrs[col]
            for atr in attrs:
                feat_type = CONSTANT.CATEGORY_TYPE
                new_col = col+'_'+atr
                new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
                col2type[new_col] = feat_type
                df[new_col] = downcast(getattr(df[col].dt,atr),accuracy_loss=False)

        X.update_data(df,col2type,None)

    @timeclass(cls='TimeDate')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)

class OriginSession(Feat):
    '''
    TO BE OPTIMIZED...
    '''
    @timeclass(cls='OriginSession')
    def fit(self,X,y):
        pass

    @timeclass(cls='OriginSession')
    def transform(self,X):
        if len(X.session_cols)!=0 or not X.user_cols or not X.key_time_col:
            return

        df = X.data.sort_index()

        user_col = df[X.user_cols[0]]
        ss = user_col.diff()

        time_col = df[X.key_time_col]
        time_diff = time_col.diff().dt.total_seconds()

        judge = ((ss!=0) | (time_diff>3600) | (time_diff<-3600))

        unique = ss[judge].shape[0]
        ss.loc[judge] = [i+1 for i in range(unique)]
        ss.loc[~judge] = np.nan
        ss = ss.fillna(method='ffill')
        new_col = 'OriginSession'
        new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.CATEGORY_TYPE)
        ss.name = new_col

        df = X.data
        df[new_col] = ss
        X.update_data(df,{},None)
        X.add_session_col(new_col)
        X.add_apart_cat_cols([new_col])
        X.add_post_drop_cols([new_col])

    @timeclass(cls='OriginSession')
    def fit_transform(self,X,y):
        self.transform(X)



class KeyTimeBin(Feat):
    @timeclass(cls='KeyTimeBin')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeyTimeBin')
    def transform(self,X):
        if not X.key_time_col:
            return

        new_col = 'KeyTimeBin'

        col2type = {}
        ss = X.data[X.key_time_col]
        timerange = (ss.max() - ss.min()).total_seconds()
        minbins = CONSTANT.TIME_MIN_BINS
        if timerange // 86400 >= minbins:
            sss = (ss-ss.iloc[0]).astype('int64')//1e9//86400
            new_col += '_Day'
        elif timerange // 3600 >= minbins:
            sss = (ss-ss.iloc[0]).astype('int64')//1e9//3600
            new_col += '_Hour'
        elif timerange // 60 >= minbins:
            sss = (ss-ss.iloc[0]).astype('int64')//1e9//60
            new_col += '_Minute'
        elif timerange >= minbins:
            sss = (ss-ss.iloc[0]).astype('int64')//1e9
            new_col += '_Second'
        else:
            return

        sss.name = new_col
        df = X.data
        df[new_col] = downcast(sss,accuracy_loss=False)

        X.bin_cols.extend([new_col])
        X.update_data(df,col2type,None)
        X.add_post_drop_cols([new_col])

    @timeclass(cls='KeyTimeBin')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)

class CatMeanEncoding(Feat):
    @timeclass(cls='CatMeanEncoding')
    def fit(self,X,y):
        raise Exception("CatMeanEncoding doesn't support fit.")

    @timeclass(cls='CatMeanEncoding')
    def transform(self,X):
        if len(X.cat_cols) == 0:
            return
        
        if not self.do:
            log(f'nsegs > shape, CatMeanEncoding is useless.')
            return

        df = X.data
        col2type = {}
        col2groupby = {}

        for col in self.todo_cols:
            new_col = col+'_CatMeanEncoding'
            feat_type = CONSTANT.NUMERICAL_TYPE
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)

            ss = df[col].map(self.catsegctr[col])
            df[new_col] = downcast(ss)
            col2type[new_col] = feat_type
            col2groupby[new_col] = col

        if len(col2type) > 0:
            X.update_data(df,col2type,col2groupby)


    @timeclass(cls='CatMeanEncoding')
    def fit_transform(self,X,y):
        if len(X.cat_cols) == 0:
            return

        self.catsegctr = {}
        df = X.data
        col2type = {}
        col2groupby = {}

        shape0 = df.shape[0]
        nsegs = 3
        log(f'Mean Encoding Segment {nsegs}')
        
        self.do = True
        if nsegs >= shape0:
            self.do = False
            log(f'nsegs > shape, CatMeanEncoding is useless.')
            return
        
        groups_tuple = gen_segs_tuple(shape0,nsegs)

        label = CONSTANT.LABEL
        df[label] = y

        def func(df):
            col = df.columns[0]
            label = df.columns[1]
            p = pd.Series()

            group = df.groupby( col ,sort=False)[label].sum()
            ss = pd.DataFrame(index=group.index)

            priori_pre = 0
            posterior_pre = 0

            ss1_pre = 'ss1_pre'
            ss2_pre = 'ss2_pre'
            ss[ss1_pre] = 0
            ss[ss2_pre] = 0

            datas = []
            for i,block in enumerate(groups_tuple):
                left = block[0]
                right = block[1]
                data = df.iloc[left:right]

                ss1 = data.groupby( col ,sort=False)[label].sum()
                ss2 = data.groupby( col ,sort=False)[label].count()

                ss['new_ss1'] = ss1
                ss['new_ss2'] = ss2

                priori = data[label].sum() / len(data) 

                if i != 0:
                    posterior = ss[ss1_pre]/ss[ss2_pre]
                    k = 50
                    lamda = 1/(1+np.exp(ss[ss2_pre]-k))
                    p = lamda * priori_pre + (1-lamda) * posterior

                datas.append(data[col].map(p))

                ss[ss1_pre] += ss['new_ss1'].fillna(0)
                ss[ss2_pre] += ss['new_ss2'].fillna(0)

                priori_pre = priori

            self.catsegctr[col] = p
            ss = pd.concat( datas )
            ss = downcast(ss)
            return ss

        cat_cols = []
        for col in X.combine_cat_cols[:50]:
            if col in df.columns and col not in (X.user_cols+X.key_cols) :
                cat_cols.append(col)
        todo_cols = X.user_cols+X.key_cols+cat_cols
        
        todo_cols = X.get_not_apart_cat_cols(todo_cols)
        self.todo_cols = todo_cols

        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[[col,label]]) for col in todo_cols)

        df.drop(label,axis=1,inplace=True)
        if res:
            new_cols = []
            for col in self.todo_cols:
                new_col = col+'_CatMeanEncoding'
                feat_type = CONSTANT.NUMERICAL_TYPE
                new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
                new_cols.append(new_col)
                col2type[new_col] = feat_type
                col2groupby[new_col] = col

            for i,col in enumerate(new_cols):
                df[col] = res[i]
            X.update_data(df,col2type,col2groupby)

class KeysCumCntRateAndReverse(Feat):
     @timeclass(cls='KeysCumCntRateAndReverse')
     def fit(self,X,y):
         pass

     @timeclass(cls='KeysCumCntRateAndReverse')
     def transform(self,X):
         df = X.data
         col2type = {}
         index = df.index
         todo_cols = X.key_cols + X.session_cols + X.user_cols

         new_cols = []

         for col in todo_cols:
             feat_type = CONSTANT.NUMERICAL_TYPE
             new_col = col+'_KeysCumCntRate'
             new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
             col2type[new_col] = feat_type
             new_cols.append(new_col)

             new_col = col+'_KeysCumCntRateReverse'
             new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
             col2type[new_col] = feat_type
             new_cols.append(new_col)

         def func(series):
             ss = series.groupby(series).cumcount()
             ss_reverse = series.groupby(series).cumcount(ascending=False)
             tmp = pd.concat([ss,ss_reverse],axis=1)
             tmp.index = series
             tmp['count'] = tmp[0]+tmp[1]+1
             tmp[0] = downcast(tmp[0]/tmp['count'])
             tmp[1] = downcast(tmp[1]/tmp['count'])
             tmp.index = index
             return tmp[[0,1]]

         res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[col]) for col in todo_cols)
         if res:
             tmp = pd.concat(res,axis=1)
             tmp.columns = new_cols

             df[new_cols] = tmp
             X.update_data(df,col2type,None)

     @timeclass(cls='KeysCumCntRateAndReverse')
     def fit_transform(self,X,y):
         self.fit(X,y)
         self.transform(X)

class UserKeyCumCntRateAndReverse(Feat):
     @timeclass(cls='UserKeyCumCntRateAndReverse')
     def fit(self,X,y):
         pass

     @timeclass(cls='UserKeyCumCntRateAndReverse')
     def transform(self,X):
         df = X.data
         col2type = {}
         index = df.index
         
         user_cols = X.user_cols
         
         key_cols = X.key_cols
         
         exec_cols = []
         new_cols = []

         for col1 in user_cols:
             for col2 in key_cols:
                 exec_cols.append((col1,col2))
                 
                 feat_type = CONSTANT.NUMERICAL_TYPE
                 new_col = f'{col1}_{col2}_CumCntRate'
                 new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
                 col2type[new_col] = feat_type
                 new_cols.append(new_col)
    
                 new_col = '{col1}_{col2}_CumCntRateReverse'
                 new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,feat_type)
                 col2type[new_col] = feat_type
                 new_cols.append(new_col)

         def func(df):
             series = gen_combine_cats(df,df.columns)
             ss = series.groupby(series).cumcount()
             ss_reverse = series.groupby(series).cumcount(ascending=False)
             tmp = pd.concat([ss,ss_reverse],axis=1)
             tmp.index = series
             tmp['count'] = tmp[0]+tmp[1]+1
             tmp[0] = downcast(tmp[0]/tmp['count'])
             tmp[1] = downcast(tmp[1]/tmp['count'])
             tmp.index = index
             return tmp[[0,1]]

         res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[[col1,col2]]) for col1,col2 in exec_cols)
         if res:
             tmp = pd.concat(res,axis=1)
             del res
             gc.collect()
             tmp.columns = new_cols
             df[new_cols] = tmp
             del tmp
             gc.collect()

             X.update_data(df,col2type,None)

     @timeclass(cls='UserKeyCumCntRateAndReverse')
     def fit_transform(self,X,y):
         self.fit(X,y)
         self.transform(X)
         
class KeysCountDIY(Feat):
    @timeclass(cls='KeysCountDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysCountDIY')
    def transform(self,X):
        df = X.data
        col2type = {}
        col2groupby = {}
        todo_cols = X.session_cols + X.key_cols + X.user_cols

        for col in todo_cols:
            col_count = df[col].value_counts()
            new_col = col+'_KeysCount'
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)

            df[new_col] =  downcast(df[col].map(col_count))
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            col2groupby[new_col] = col

        if len(col2type)>0:
            X.update_data(df,col2type,col2groupby,col2source_cat=col2groupby)

    @timeclass(cls='KeysCountDIY')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)
        
class UserKeyCntDIY(Feat):
    @timeclass(cls='UserKeyCntDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='UserKeyCntDIY')
    def transform(self,X):
        if not X.user_cols:
            return
        
        user_cols = sorted(X.user_cols)
        key_cols = sorted(X.key_cols)

        col2type = {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for user_col in user_cols:
            for key_col in key_cols:
                new_col = '{}_{}_cnt'.format(user_col, key_col)
                new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                exec_cols.append((user_col, key_col))
                new_cols.append(new_col)
                col2groupby[new_col] = (user_col,key_col)
                
        df = pd.DataFrame(index=X.data.index)
        for col in user_cols+key_cols:
            df = pd.concat([df,X.data[col]],axis=1)
            
        
        def func(df):
            cats = gen_combine_cats(df, df.columns)
            cnt = cats.value_counts()
            ss = cats.map(cnt)
            return downcast(ss)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
            X.update_data(df, col2type,col2groupby)
            
    @timeclass(cls='UserKeyCntDIY')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)

class SessionKeyCntDIY(Feat):
    @timeclass(cls='SessionKeyCntDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='SessionKeyCntDIY')
    def transform(self,X):

        if not X.session_cols:
            return

        session_cols = sorted(X.session_cols)
        key_cols = sorted(X.key_cols)

        col2type = {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for session_col in session_cols:
            for key_col in key_cols:
                new_col = '{}_{}_cnt'.format(session_col, key_col)
                new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                exec_cols.append((session_col, key_col))
                new_cols.append(new_col)
                col2groupby[new_col] = (session_col,key_col)
            
        df = pd.DataFrame(index=X.data.index)
        for col in session_cols+key_cols:
            df = pd.concat([df,X.data[col]],axis=1)
            

        def func(df):
            cats = gen_combine_cats(df, df.columns)
            cnt = cats.value_counts()
            ss = cats.map(cnt)
            return downcast(ss)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
            X.update_data(df, col2type,col2groupby)

    @timeclass(cls='SessionKeyCntDIY')
    def fit_transform(self, X,y):
        self.transform(X)

class UserSessionNuniqueDIY(Feat):
    @timeclass(cls='UserSessionNuniqueDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='UserSessionNuniqueDIY')
    def transform(self,X):

        user_cols = X.user_cols
        session_cols = X.session_cols

        col2type =  {}
        col2groupby = {}

        exec_cols = []
        new_cols = []


        for col1 in user_cols:
            for col2 in session_cols:
                   exec_cols.append((col1, col2))
                   new_col = '{}_{}_Nunique'.format(col1, col2)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col1
                   new_cols.append(new_col)

        df = pd.DataFrame(index=X.data.index)
        for col in user_cols+session_cols:
            df = pd.concat([df,X.data[col]],axis=1)
        
        def func(df):
            col1 = df.columns[0]
            col2 = df.columns[1]
            group = df.groupby(col1)[col2]
            ss = group.nunique()
            new_col = '{}_{}_Nunique'.format(col1, col2)
            ss = downcast(ss)
            ss = df[col1].map(ss)
            ss.name = new_col
            return ss
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            for new_col in new_cols:
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
                
            X.update_data(df,col2type,col2groupby)

    @timeclass(cls='UserSessionNuniqueDIY')
    def fit_transform(self,X,y):
        self.transform(X)
        
class UserSessionCntDivNuniqueDIY(Feat):
    @timeclass(cls='UserSessionCntDivNuniqueDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='UserSessionCntDivNuniqueDIY')
    def transform(self,X):

        user_cols = X.user_cols
        session_cols = X.session_cols

        col2type =  {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for col1 in user_cols:
            for col2 in session_cols:
                   exec_cols.append((col1, col2))
                   new_col = '{}_{}_CntDivNunique'.format(col1, col2)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col1
                   new_cols.append(new_col)

        df = pd.DataFrame(index=X.data.index)
        for col in user_cols+session_cols:
            df = pd.concat([df,X.data[col]],axis=1)

        def func(df):
            col1 = df.columns[0]
            col2 = df.columns[1]
            group = df.groupby(col1)[col2]
            ss = group.count() / group.nunique()
            new_col = '{}_{}_CntDivNunique'.format(col1, col2)
            ss = downcast(ss)
            ss = df[col1].map(ss)
            ss.name = new_col
            return ss
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            for new_col in new_cols:
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
            X.update_data(df,col2type,col2groupby)

    @timeclass(cls='UserSessionCntDivNuniqueDIY')
    def fit_transform(self,X,y):
        self.transform(X)
        
class UserKeyNuniqueDIY(Feat):
    @timeclass(cls='UserKeyNuniqueDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='UserKeyNuniqueDIY')
    def transform(self,X):

        keys_cols = X.user_cols
        cat_cols = X.key_cols

        col2type =  {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for col1 in keys_cols:
            todo_cols = X.get_groupby_cols(by=col1,cols=cat_cols)
            for col2 in todo_cols:
                if col1 != col2:
                   exec_cols.append((col2, col1))
                   new_col = '{}_{}_Nunique'.format(col2, col1)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col2
                   new_cols.append(new_col)

                   exec_cols.append((col1, col2))
                   new_col = '{}_{}_Nunique'.format(col1, col2)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col1
                   new_cols.append(new_col)
                   
        df = pd.DataFrame(index=X.data.index)
        for col in keys_cols+cat_cols:
            df = pd.concat([df,X.data[col]],axis=1)

        def func(df):
            col1 = df.columns[0]
            col2 = df.columns[1]
            group = df.groupby(col1)[col2]
            ss = group.nunique()
            new_col = '{}_{}_Nunique'.format(col1, col2)

            ss = downcast(ss)
            ss = df[col1].map(ss)
            ss.name = new_col
            return ss
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            for new_col in new_cols:
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
            X.update_data(df,col2type,col2groupby)

    @timeclass(cls='UserKeyNuniqueDIY')
    def fit_transform(self,X,y):
        self.transform(X)

class SessionKeyNuniqueDIY(Feat):
    @timeclass(cls='SessionKeyNuniqueDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='SessionKeyNuniqueDIY')
    def transform(self,X):

        keys_cols = X.session_cols
        cat_cols = X.key_cols

        col2type =  {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for col1 in keys_cols:
            todo_cols = X.get_groupby_cols(by=col1,cols=cat_cols)
            for col2 in todo_cols:
                if col1 != col2:
                   exec_cols.append((col2, col1))
                   new_col = '{}_{}_Nunique'.format(col2, col1)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col2
                   new_cols.append(new_col)

                   exec_cols.append((col1, col2))
                   new_col = '{}_{}_Nunique'.format(col1, col2)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col1
                   new_cols.append(new_col)
                   
        df = pd.DataFrame(index=X.data.index)
        for col in keys_cols+cat_cols:
            df = pd.concat([df,X.data[col]],axis=1)

        def func(df):
            col1 = df.columns[0]
            col2 = df.columns[1]
            group = df.groupby(col1)[col2]
            ss = group.nunique()
            new_col = '{}_{}_Nunique'.format(col1, col2)
            ss = downcast(ss)
            ss = df[col1].map(ss)
            ss.name = new_col
            return ss
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            for new_col in new_cols:
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
            X.update_data(df,col2type,col2groupby)

    @timeclass(cls='SessionKeyNuniqueDIY')
    def fit_transform(self,X,y):
        self.transform(X)
        
class UserKeyCntDivNuniqueDIY(Feat):
    @timeclass(cls='UserKeyCntDivNuniqueDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='UserKeyCntDivNuniqueDIY')
    def transform(self,X):

        keys_cols = X.user_cols
        cat_cols = X.key_cols

        col2type =  {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for col1 in keys_cols:
            todo_cols = X.get_groupby_cols(by=col1,cols=cat_cols)
            for col2 in todo_cols:
                if col1 != col2:
                   exec_cols.append((col2, col1))
                   new_col = '{}_{}_CntDivNunique'.format(col2, col1)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col2
                   new_cols.append(new_col)

                   exec_cols.append((col1, col2))
                   new_col = '{}_{}_CntDivNunique'.format(col1, col2)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col1
                   new_cols.append(new_col)
                   
        df = pd.DataFrame(index=X.data.index)
        for col in cat_cols+keys_cols:
            df = pd.concat([df,X.data[col]],axis=1)
            
        def func(df):
            col1 = df.columns[0]
            col2 = df.columns[1]
            group = df.groupby(col1)[col2]
            ss = group.count() / group.nunique()
            new_col = '{}_{}_CntDivNunique'.format(col1, col2)

            ss = downcast(ss)
            ss = df[col1].map(ss)
            ss.name = new_col
            return ss
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            for new_col in new_cols:
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
            X.update_data(df,col2type,col2groupby)

    @timeclass(cls='UserKeyCntDivNuniqueDIY')
    def fit_transform(self,X,y):
        self.transform(X)
        
class SessionKeyCntDivNuniqueDIY(Feat):
    @timeclass(cls='SessionKeyCntDivNuniqueDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='SessionKeyCntDivNuniqueDIY')
    def transform(self,X):

        keys_cols = X.session_cols
        cat_cols = X.key_cols

        col2type =  {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for col1 in keys_cols:
            todo_cols = X.get_groupby_cols(by=col1,cols=cat_cols)
            for col2 in todo_cols:
                if col1 != col2:
                   exec_cols.append((col2, col1))
                   new_col = '{}_{}_CntDivNunique'.format(col2, col1)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col2
                   new_cols.append(new_col)

                   exec_cols.append((col1, col2))
                   new_col = '{}_{}_CntDivNunique'.format(col1, col2)
                   new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                   col2groupby[new_col] = col1
                   new_cols.append(new_col)
                   
        df = pd.DataFrame(index=X.data.index)
        for col in cat_cols+keys_cols:
            df = pd.concat([df,X.data[col]],axis=1)
            
        def func(df):
            col1 = df.columns[0]
            col2 = df.columns[1]
            group = df.groupby(col1)[col2]
            ss = group.count() / group.nunique()
            new_col = '{}_{}_CntDivNunique'.format(col1, col2)

            ss = downcast(ss)
            ss = df[col1].map(ss)
            ss.name = new_col
            return ss
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            for new_col in new_cols:
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]
            X.update_data(df,col2type,col2groupby)

    @timeclass(cls='SessionKeyCntDivNuniqueDIY')
    def fit_transform(self,X,y):
        self.transform(X)
        
class KeysBinCntDIY(Feat):
    @timeclass(cls='KeysBinCntDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysBinCntDIY')
    def transform(self,X):

        if not X.key_cols or not X.bin_cols:
            return

        key_cols = sorted(X.key_cols+X.session_cols+X.user_cols)
        bin_cols = sorted(X.bin_cols)

        col2type = {}
        col2groupby = {}

        exec_cols = []
        new_cols = []

        for key_col in key_cols:
            for bin_col in bin_cols:
                new_col = '{}_{}_cnt'.format(key_col, bin_col)
                new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                exec_cols.append((key_col, bin_col))
                new_cols.append(new_col)
                col2groupby[new_col] = (key_col,bin_col)
        
        tmp = []
        
        for col in key_cols+bin_cols:
            tmp.append(X.data[col])
        
        df = pd.concat( tmp ,axis=1)
        
        def func(df):
            cats = gen_combine_cats(df, df.columns)
            cnt = cats.value_counts()
            ss = cats.map(cnt)
            return downcast(ss)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            df = X.data
            for col in tmp.columns:
                df[col] = tmp[col]

            X.update_data(df, col2type,col2groupby)

    @timeclass(cls='KeysBinCntDIY')
    def fit_transform(self, X,y):
        self.transform(X)
        
class CatCountDIY(Feat):
    @timeclass(cls='CatCountDIY')
    def fit(self,X,y):
        pass

    @timeclass(cls='CatCountDIY')
    def transform(self,X):
        df = X.data
        col2type = {}
        col2groupby = {}
        todo_cols = X.cat_cols
        todo_cols = todo_cols[:300]
        
        if not todo_cols:
            return

        new_cols = []
        for col in todo_cols:
            new_col =  col+'_CatCount'
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
            new_cols.append(new_col)
            col2groupby[new_col] = col

        def func(series):
            col = series.name
            col_count = series.value_counts()
            new_col = col+'_CatCount'
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
            ss = downcast(series.map(col_count))
            return ss

        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(df[col]) for col in todo_cols)
        if res:
            tmp = pd.concat(res,axis=1)
            tmp.columns = new_cols
            
            if df.shape[0] <= 2000000:
                df = pd.concat([df,tmp],axis=1)
            else:
                for col in tmp.columns:
                    df[col] = tmp[col]
                    
            X.update_data(df,col2type,col2groupby,col2source_cat=col2groupby)

    @timeclass(cls='CatCountDIY')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)
        
class KeysNumMeanOrder2MinusSelfNew(Feat):
    @timeclass(cls='KeysNumMeanOrder2MinusSelfNew')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysNumMeanOrder2MinusSelfNew')
    def transform(self,X,useful_cols=None):
        df = X.data

        todo_cols = X.user_cols+X.key_cols+X.session_cols
        num_cols = X.combine_num_cols
        
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_max]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_mean'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    exec_cols.append((col, num_col))
                    new_cols.append(new_col)
    
                    new_col = '{}_{}_meanMinusSelf'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
        else:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_max]
                for num_col in cur_num_cols:
                    new_col1 = '{}_{}_mean'.format(col, num_col)
                    new_col1 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col1,CONSTANT.NUMERICAL_TYPE)
                    if new_col1 not in useful_cols:
                        do1 = False
                    else:
                        do1 = True
                        new_cols.append(new_col1)
                        col2type[new_col1] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col1] = col
    
                    new_col = '{}_{}_meanMinusSelf'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    if new_col not in useful_cols:
                        do2 = False
                    else:
                        do2 = True 
                        new_cols.append(new_col)
                        col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col] = col
                        
                    if not (do1 | do2):
                        continue
                    
                    exec_cols.append((col, num_col))

        def func(df,useful_cols):
            col = df.columns[0]
            num_col = df.columns[1]
            
            df_new = pd.DataFrame(index=df.index)
            
            df[num_col] = df[num_col].astype('float32')
            
            means = df.groupby(col,sort=False)[num_col].mean()
            ss = df[col].map(means)
            
            df_new['new'] = downcast(ss)
            df_new['minus'] = downcast(ss-df[num_col])
            
            if useful_cols is None:
                return df_new
            
            new_col1 = '{}_{}_mean'.format(col, num_col)
            new_col1 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col1,CONSTANT.NUMERICAL_TYPE)
            new_col = '{}_{}_meanMinusSelf'.format(col, num_col)
            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
            if new_col1 in useful_cols and new_col in useful_cols:
                return df_new
            elif new_col1 not in useful_cols:
                return df_new['new']
            elif new_col not in useful_cols:
                return df_new['minus']
            else:
                return None

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]], useful_cols) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            del res
            gc.collect()
        
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
            
            del tmp
            gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []
    
    @timeclass(cls='KeysNumMeanOrder2MinusSelfNew')
    def fit_transform(self, X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    
class KeysNumMaxMinOrder2MinusSelfNew(Feat):
    @timeclass(cls='KeysNumMaxMinOrder2MinusSelfNew')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysNumMaxMinOrder2MinusSelfNew')
    def transform(self,X,useful_cols=None):
        df = X.data

        todo_cols = X.user_cols+X.key_cols+X.session_cols
        num_cols = X.combine_num_cols
        
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_maxmin]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_max'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    exec_cols.append((col, num_col))
                    new_cols.append(new_col)
                    
                    new_col = '{}_{}_min'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
                    
                    new_col = '{}_{}_maxMinusSelf'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
                    
                    new_col = '{}_{}_minMinusSelf'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
        else:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_maxmin]
                for num_col in cur_num_cols:
                    new_col1 = '{}_{}_max'.format(col, num_col)
                    new_col1 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col1,CONSTANT.NUMERICAL_TYPE)
                    if new_col1 not in useful_cols:
                        do1 = False
                    else:
                        do1 = True
                        new_cols.append(new_col1)
                        col2type[new_col1] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col1] = col
                        
                    new_col2 = '{}_{}_min'.format(col, num_col)
                    new_col2 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col2,CONSTANT.NUMERICAL_TYPE)
                    if new_col2 not in useful_cols:
                        do2 = False
                    else:
                        do2 = True
                        new_cols.append(new_col2)
                        col2type[new_col2] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col2] = col
                    
                    new_col3 = '{}_{}_maxMinusSelf'.format(col, num_col)
                    new_col3 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col3,CONSTANT.NUMERICAL_TYPE)
                    if new_col3 not in useful_cols:
                        do3 = False
                    else:
                        do3 = True
                        new_cols.append(new_col3)
                        col2type[new_col3] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col3] = col
                        
                    new_col4 = '{}_{}_minMinusSelf'.format(col, num_col)
                    new_col4 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col4,CONSTANT.NUMERICAL_TYPE)
                    if new_col4 not in useful_cols:
                        do4 = False
                    else:
                        do4 = True
                        new_cols.append(new_col4)
                        col2type[new_col4] = CONSTANT.NUMERICAL_TYPE
                        col2groupby[new_col4] = col
                        
                    if not (do1 | do2 | do3 | do4):
                        continue
                        
                    exec_cols.append((col, num_col))

        def func(df,useful_cols):
            col = df.columns[0]
            num_col = df.columns[1]
            
            df_new = pd.DataFrame(index=df.index)
            
            df[num_col] = df[num_col].astype('float32')
            
            maxs = df.groupby(col,sort=False)[num_col].max()
            mins = df.groupby(col,sort=False)[num_col].min()
            ss_max = df[col].map(maxs)
            ss_min = df[col].map(mins)
            
            new_col1 = '{}_{}_max'.format(col, num_col)
            new_col1 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col1,CONSTANT.NUMERICAL_TYPE)
            new_col2 = '{}_{}_min'.format(col, num_col)
            new_col2 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col2,CONSTANT.NUMERICAL_TYPE)
            new_col3 = '{}_{}_maxMinusSelf'.format(col, num_col)
            new_col3 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col3,CONSTANT.NUMERICAL_TYPE)
            new_col4 = '{}_{}_minMinusSelf'.format(col, num_col)
            new_col4 = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col4,CONSTANT.NUMERICAL_TYPE)
      
            df_new[new_col1] = downcast(ss_max)
            df_new[new_col2] = downcast(ss_min)
            df_new[new_col3] = downcast(ss_max-df[num_col])
            df_new[new_col4] = downcast(ss_min-df[num_col])
            
            if useful_cols is None:
                return df_new
            
            new_cols = [new_col1, new_col2, new_col3, new_col4]
            exec_cols = []
            for new_col in new_cols:
                if new_col in useful_cols:
                    exec_cols.append(new_col)
            
            if not exec_cols:
                return None
            return df_new[exec_cols]

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]], useful_cols) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            del res
            gc.collect()
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
            
            del tmp
            gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []
    
    @timeclass(cls='KeysNumMaxMinOrder2MinusSelfNew')
    def fit_transform(self, X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    
class KeysNumStd(Feat):
    @timeclass(cls='KeysNumStd')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysNumStd')
    def transform(self,X,useful_cols=None):
        df = X.data

        todo_cols = X.user_cols+X.key_cols+X.session_cols
        num_cols = X.combine_num_cols
        
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_std]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_std'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    exec_cols.append((col, num_col))
                    new_cols.append(new_col)

        else:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_num_std]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_std'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    if new_col not in useful_cols:
                        continue
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    new_cols.append(new_col)
                    exec_cols.append((col, num_col))

        def func(df,useful_cols):
            col = df.columns[0]
            num_col = df.columns[1]
            
            df[num_col] = df[num_col].astype('float32')
            
            std = df.groupby(col,sort=False)[num_col].std()
            ss = df[col].map(std)
            return downcast(ss)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]], useful_cols) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            del res
            gc.collect()
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
            
            del tmp
            gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []
    
    @timeclass(cls='KeysNumStd')
    def fit_transform(self, X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    
class KeysCatCntOrder2New(Feat):
    @timeclass(cls='KeysCatCntOrder2New')
    def fit(self,X,y):
        pass

    @timeclass(cls='KeysCatCntOrder2New')
    def transform(self,X,useful_cols=None):
        df = X.data

        todo_cols = X.user_cols+X.key_cols+X.session_cols
        cat_cols = X.combine_cat_cols
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for col in todo_cols:
                work_cols = X.get_groupby_cols(by=col,cols=cat_cols)
                work_cols = work_cols[:self.config.keys_order2_cat_max]
                for cat_col in work_cols:
                    new_col = '{}_{}_cnt'.format(col, cat_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = (col,cat_col)
                    exec_cols.append((col, cat_col))
                    new_cols.append(new_col)
        else:
            for col in todo_cols:
                work_cols = X.get_groupby_cols(by=col,cols=cat_cols)
                work_cols = work_cols[:self.config.keys_order2_cat_max]
                for cat_col in work_cols:
                    new_col = '{}_{}_cnt'.format(col, cat_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    if new_col not in useful_cols:
                        continue
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = (col,cat_col)
                    exec_cols.append((col, cat_col))
                    new_cols.append(new_col)
             
        def func(df):
            cats = gen_combine_cats(df, df.columns)
            cnt = cats.value_counts()
            ss = cats.map(cnt)
            return downcast(ss)
        
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            del res
            gc.collect()
            tmp.columns = new_cols
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
                gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []

    @timeclass(cls='KeysCatCntOrder2New')
    def fit_transform(self,X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    
class BinsCatCntOrder2DIYNew(Feat):
    @timeclass(cls='BinsCatCntOrder2DIYNew')
    def fit(self,X,y):
        pass

    @timeclass(cls='BinsCatCntOrder2DIYNew')
    def transform(self,X,useful_cols):
        df = X.data

        todo_cols = X.bin_cols
        cat_cols = X.combine_cat_cols + X.user_cols+X.key_cols+X.session_cols
        
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for col in todo_cols:
                work_cols = X.get_groupby_cols(by=col,cols=cat_cols)
                work_cols = work_cols[:self.config.keys_order2_bin_cat_max]
                for cat_col in work_cols:
                    new_col = '{}_{}_cnt'.format(col, cat_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = (col,cat_col)
                    exec_cols.append((col, cat_col))
                    new_cols.append(new_col)
        else:
            for col in todo_cols:
                work_cols = X.get_groupby_cols(by=col,cols=cat_cols)
                work_cols = work_cols[:self.config.keys_order2_bin_cat_max]
                for cat_col in work_cols:
                    new_col = '{}_{}_cnt'.format(col, cat_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    if new_col not in useful_cols:
                        continue
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = (col,cat_col)
                    exec_cols.append((col, cat_col))
                    new_cols.append(new_col)

        def func(df):
            cats = gen_combine_cats(df, df.columns)
            cnt = cats.value_counts()
            ss = cats.map(cnt)
            return downcast(ss)
        
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            del res
            gc.collect()
            tmp.columns = new_cols
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
                gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []
    
    @timeclass(cls='BinsCatCntOrder2DIYNew')
    def fit_transform(self, X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    
class BinsNumMeanOrder2DIYNew(Feat):
    @timeclass(cls='BinsNumMeanOrder2DIYNew')
    def fit(self,X,y):
        pass

    @timeclass(cls='BinsNumMeanOrder2DIYNew')
    def transform(self,X,useful_cols):
        df = X.data

        todo_cols = X.bin_cols
        num_cols = X.combine_num_cols
        
        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_bin_num_max]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_mean'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    exec_cols.append((col, num_col))
                    new_cols.append(new_col)
        else:
            for col in todo_cols:
                cur_num_cols = X.get_groupby_cols(by=col,cols=num_cols)
                cur_num_cols = cur_num_cols[:self.config.keys_order2_bin_num_max]
                for num_col in cur_num_cols:
                    new_col = '{}_{}_mean'.format(col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    if new_col not in useful_cols:
                        continue
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = col
                    exec_cols.append((col, num_col))
                    new_cols.append(new_col)
                
        def func(df):
            col = df.columns[0]
            num_col = df.columns[1]
            
            df[num_col] = df[num_col].astype('float32')
            
            means = df.groupby(col,sort=False)[num_col].mean()
            ss = df[col].map(means)
            return downcast(ss)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            tmp.columns = new_cols
            del res
            gc.collect()
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
            del tmp
            gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []

    @timeclass(cls='BinsNumMeanOrder2DIYNew')
    def fit_transform(self, X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    
class CatNumMeanOrder2DIYNew(Feat):
    @timeclass(cls='CatNumMeanOrder2DIYNew')
    def fit(self,X,y):
        pass

    @timeclass(cls='CatNumMeanOrder2DIYNew')
    def transform(self,X,useful_cols):
        df = X.data

        todo_cat_cols = X.combine_cat_cols[:self.config.all_order2_cat_max]
        todo_num_cols = X.combine_num_cols[:self.config.all_order2_num_max]

        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for cat_col in todo_cat_cols:
                cur_todo_cols = X.get_groupby_cols(by=cat_col,cols=todo_num_cols)
                for num_col in cur_todo_cols:
                    new_col = '{}_{}_mean'.format(cat_col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = (cat_col,num_col)
                    exec_cols.append((cat_col, num_col))
                    new_cols.append(new_col)
        else:
            for cat_col in todo_cat_cols:
                cur_todo_cols = X.get_groupby_cols(by=cat_col,cols=todo_num_cols)
                for num_col in cur_todo_cols:
                    new_col = '{}_{}_mean'.format(cat_col, num_col)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    if new_col not in useful_cols:
                        continue
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = (cat_col,num_col)
                    exec_cols.append((cat_col, num_col))
                    new_cols.append(new_col)

        def func(df):
            cat_col = df.columns[0]
            num_col = df.columns[1]
            
            df[num_col] = df[num_col].astype('float32')
            
            means = df.groupby(cat_col,sort=False)[num_col].mean()
            ss = df[cat_col].map(means)
            return downcast(ss)

        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            del res
            gc.collect()
            tmp.columns = new_cols
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
                gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []

    @timeclass(cls='CatNumMeanOrder2DIYNew')
    def fit_transform(self,X,y,useful_cols=None):
        return self.transform(X,useful_cols)
    
class CatCntOrder2DIYNew(Feat):
    @timeclass(cls='CatCntOrder2DIYNew')
    def fit(self,X,y):
        pass

    @timeclass(cls='CatCntOrder2DIYNew')
    def transform(self,X,useful_cols):
        df = X.data

        todo_cols = X.combine_cat_cols[:self.config.all_order2_cat_max]
        size = len(todo_cols)

        col2type = {}
        col2groupby = {}
        exec_cols = []
        new_cols = []
        
        if useful_cols is None:
            for i in range(size):
                col1 = todo_cols[i]
                cur_todo_cols = X.get_groupby_cols(by=col1,cols=todo_cols)
                cur_todo_cols = set(cur_todo_cols)
                for j in range(i+1, size):
                    col2 = todo_cols[j]
                    if col2 not in cur_todo_cols:
                        continue
                    new_col = '{}_{}_cnt'.format(col1, col2)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = (col1,col2)
                    exec_cols.append((col1, col2))
                    new_cols.append(new_col)
        else:
            for i in range(size):
                col1 = todo_cols[i]
                cur_todo_cols = X.get_groupby_cols(by=col1,cols=todo_cols)
                cur_todo_cols = set(cur_todo_cols)
                for j in range(i+1, size):
                    col2 = todo_cols[j]
                    if col2 not in cur_todo_cols:
                        continue
                    new_col = '{}_{}_cnt'.format(col1, col2)
                    new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                    if new_col not in useful_cols:
                        continue
                    col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                    col2groupby[new_col] = (col1,col2)
                    exec_cols.append((col1, col2))
                    new_cols.append(new_col)

        def func(df):
            cats = gen_combine_cats(df, df.columns)
            cnt = cats.value_counts()
            ss = cats.map(cnt)

            return downcast(ss)
        res = Parallel(n_jobs=CONSTANT.JOBS, require='sharedmem')(delayed(func)(df[[col1, col2]]) for col1, col2 in exec_cols)
        if res:
            tmp = pd.concat(res, axis=1)
            del res
            gc.collect()
            
            tmp.columns = new_cols
            for col in tmp.columns:
                df[col] = tmp[col]
                del tmp[col]
                gc.collect()
            X.update_data(df, col2type,col2groupby)
            X.add_wait_selection_cols(new_cols)
            return new_cols
        return []

    @timeclass(cls='CatCntOrder2DIYNew')
    def fit_transform(self, X,y,useful_cols=None):
        return self.transform(X,useful_cols)

class PreMcToNumpy(Feat):
    @timeclass(cls='PreMcToNumpy')
    def fit(self,X,y):
        pass
            
    @timeclass(cls='PreMcToNumpy')
    def transform(self,X):
        df = X.data
        multi_cols = X.multi_cat_cols
        
        col2muldatas = {}
        col2muldatalens = {}
        
        for col in multi_cols:
            vals = df[col].values
            datas,datalen = ac.get_need_data(vals)
        
            col2muldatas[col] = np.array(datas,dtype='int64').astype(np.int32)
            col2muldatalens[col] = np.array(datalen,dtype='int32')
            
        X.col2muldatas = col2muldatas
        X.col2muldatalens = col2muldatalens
        
    @timeclass(cls='PreMcToNumpy')   
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X) 
        
class McCatRank(Feat):
    @timeclass(cls='McCatRank')
    def fit(self,X,y):
        pass
            
    @timeclass(cls='McCatRank')
    def transform(self,X):
        df = X.data
        col2type = {}
        
        new_cols = []
        exec_cols = []
        
        max_count = 300
        for col1 in X.multi_cat_cols:
            if (col1 in X.col2block):
                block_id1 = X.col2block[col1]
                
                for col2 in (X.cat_cols+X.user_cols+X.key_cols+X.session_cols):
                    if (col2 in X.col2block):
                        block_id2 = X.col2block[col2]
                        
                        if block_id1 == block_id2:
                            exec_cols.append( (col1,col2) )
                            new_col =  col1+'_'+col2
                            new_col = FeatContext.gen_feat_name(namespace,self.__class__.__name__,new_col,CONSTANT.NUMERICAL_TYPE)
                            col2type[new_col] = CONSTANT.NUMERICAL_TYPE
                            new_cols.append(new_col)
                            if len(new_cols)>=max_count:
                                break
            
            if len(new_cols)>=max_count:
                break
        
        X.drop_data(X.multi_cat_cols)
        gc.collect()
        
        cat_cols = []
        for (col1,col2) in exec_cols:
            cat_cols.append(col2)
        
        cat_cols = sorted(list(set(cat_cols)))
        
        col2muldatas = X.col2muldatas
        col2muldatalens = X.col2muldatalens
        
        col2catdatas = {}
        
        for col in cat_cols:
            catdata = df[col].fillna(-1).astype('int32').values
            col2catdatas[col] = catdata

        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(ac.cat_rank_multi)(col2muldatas[col1],col2muldatalens[col1],col2catdatas[col2]) for (col1,col2) in exec_cols)
        
        del X.col2muldatas,col2muldatas
        del X.col2muldatalens,col2muldatalens
        del col2catdatas
        gc.collect()
        if res:
            index = df.index
            for i in range(len(res)):
                res[i] = downcast(pd.Series(res[i],index=index))
        
            tmp = pd.concat(res,axis=1) 
            del res
            gc.collect()
            
            tmp.columns = new_cols
            
            if df.shape[0] <= 2000000:
                df = pd.concat([df,tmp],axis=1)
            else:
                for col in tmp.columns:
                    df[col] = tmp[col]

            X.update_data(df,col2type,None,None)
         
    @timeclass(cls='McCatRank')   
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X) 