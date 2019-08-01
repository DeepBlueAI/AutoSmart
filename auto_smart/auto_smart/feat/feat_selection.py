# -*- coding: utf-8 -*-
from util import timeclass,log
import CONSTANT
from model_input import FeatOutput
from automl import autosample
import gc
import lightgbm as lgb
import pandas as pd
from .feat import Feat
import time
import numpy as np

def lgb_train(X,y):
    num_boost_round = 100
    num_leaves = 63

    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': "None",
        'learning_rate': 0.1,
        'num_leaves': num_leaves,
        'max_depth': -1,
        'min_child_samples': 20,
        'max_bin':255,
        'subsample': 0.9,
        'subsample_freq': 1,
        'colsample_bytree': 1,
        'min_child_weight': 0.001,
        'subsample_for_bin': 200000,
        'min_split_gain': 0.02,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'seed': CONSTANT.SEED,
        'nthread': CONSTANT.THREAD_NUM,
    }

    data = X.data

    y_train = y

    max_sample_num = min(len(y_train),50000)
    y_train = autosample.downsampling_y(y_train,max_sample_num)

    X_train = data.loc[y_train.index]

    X.data = X_train
    feat_output = FeatOutput()
    X_train,y_train,categories = feat_output.fit_transform_output(X,y_train)

    X.data = data
    gc.collect()

    feat_name_cols = list(X_train.columns)
    feat_name_maps = { feat_name_cols[i] : str(i)  for i in range(len(feat_name_cols)) }
    f_feat_name_maps = { str(i) : feat_name_cols[i] for i in range(len(feat_name_cols)) }
    new_feat_name_cols = [ feat_name_maps[i] for i in feat_name_cols ]
    X_train.columns = new_feat_name_cols

    dtrain = lgb.Dataset(X_train,y_train,feature_name=list(X_train.columns))
    model = lgb.train(params,dtrain,
                         num_boost_round=num_boost_round,
                         categorical_feature=[],
                         )

    df_imp = pd.DataFrame({'features': [ f_feat_name_maps[i] for i in model.feature_name() ] ,
             'importances':model.feature_importance()})

    df_imp.sort_values('importances',ascending=False,inplace=True)

    return df_imp

class LGBFeatureSelection(Feat):
    @timeclass(cls='LGBFeatureSelection')
    def fit(self,X,y):
        now = time.time()
        log(f'LGBFeatureSelection:{now-self.config.start_time}')

        threshold = 5
        df_imp = lgb_train(X,y)
        log(f'importances sum {df_imp["importances"].sum()}')
        if df_imp["importances"].sum() != 6200:
            keep_feats = list(df_imp.loc[df_imp['importances'] >= threshold,'features'])
            if len(keep_feats) < 150:
                useful_feats = list(df_imp.loc[df_imp['importances'] > 0,'features'])
                if len(useful_feats) <= 150:
                    keep_feats = useful_feats
                else:
                    df_imp_sorted = df_imp.sort_values(by='importances',ascending=False)
                    keep_feats = list(df_imp_sorted['features'].iloc[:150])
        else:
            keep_feats = list(df_imp.loc[df_imp['importances'] >= threshold,'features'])
        
        log(f'keep feats num {len(keep_feats)}')
        
        keep_cats = []

        keep_cats_set = set()
        cat_set = set(X.cat_cols)

        for feat in keep_feats:

            if X.col2type[feat] == CONSTANT.CATEGORY_TYPE:
                if feat in cat_set:
                    if feat not in keep_cats_set:
                        keep_cats_set.add(feat)
                        keep_cats.append(feat)

            elif feat in X.col2source_cat:
                keep_feat = X.col2source_cat[feat]
                if keep_feat in cat_set:
                    if keep_feat not in keep_cats_set:
                        keep_cats_set.add(keep_feat)
                        keep_cats.append(keep_feat)

        drop_feats = list(set(df_imp['features'].tolist()) - set(keep_feats))

        drop_feats = list(set(drop_feats) - keep_cats_set)
        self.drop_feats = drop_feats
        log(f'total feat num:{df_imp.shape[0]}, drop feat num:{len(self.drop_feats)}')

        keep_nums = []
        for feat in keep_feats:
            if X.col2type[feat] ==  CONSTANT.NUMERICAL_TYPE:
                keep_nums.append(feat)

        keep_binaries = []
        for feat in keep_feats:
            if X.col2type[feat] ==  CONSTANT.BINARY_TYPE:
                keep_binaries.append(feat)

        assert(len(set(keep_cats) & set(drop_feats))==0)
        assert(len(set(keep_nums) & set(drop_feats))==0)
        assert(len(set(keep_binaries) & set(drop_feats))==0)

        X.reset_combine_cols(keep_cats,keep_nums,keep_binaries)

    @timeclass(cls='LGBFeatureSelection')
    def transform(self,X):
        X.drop_data(self.drop_feats)
        return self.drop_feats
    
    @timeclass(cls='LGBFeatureSelection')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)
        return self.drop_feats

class LGBFeatureSelectionLast(Feat):
    @timeclass(cls='LGBFeatureSelectionLast')
    def fit(self,X,y):
        now = time.time()
        log(f'LGBFeatureSelectionLast:{now-self.config.start_time}')

        start_time = time.time()
        df_imp = lgb_train(X,y)
        
        data = X.data
        shape = data.shape
        y_pos = len(y[y==1])
        y_neg = len(y[y==0])
        unbalance_ratio = y_pos / y_neg if y_pos > y_neg else y_neg / y_pos
        memory_usage = pd.Series(np.zeros(shape[0]),dtype=np.float32).memory_usage() / 1024 / 1024 / 1024
        gc.collect()
        
        if unbalance_ratio >= 7:
            memory_constrain = 2
        elif unbalance_ratio >= 4:
            memory_constrain = 1.8
        else:
            memory_constrain = 1.6
            
        col_constrain =  int(memory_constrain / memory_usage)
        
        end_time = time.time()
        
        use_time = end_time-start_time
        user_time_rate = use_time / self.config.budget
               
        if user_time_rate > 0.1:
            threshold = 13
        elif user_time_rate > 0.09:
            threshold = 12
        elif user_time_rate > 0.08:
            threshold = 11
        elif user_time_rate > 0.07:
            threshold = 10
        elif user_time_rate > 0.06:
            threshold = 9
        elif user_time_rate > 0.05:
            threshold = 8
        elif user_time_rate > 0.04:
            threshold = 7
        elif user_time_rate > 0.03:
            threshold = 6
        else:
            threshold = 5
            
        log(f'LGBFeatureSelectionLast threshold {threshold}')
        
        log(f'importances sum {df_imp["importances"].sum()}')
        if df_imp["importances"].sum() != 6200:
            keep_feats = list(df_imp.loc[df_imp['importances'] >= threshold,'features'])
            if len(keep_feats) < 150:
                useful_feats = list(df_imp.loc[df_imp['importances'] > 0,'features'])
                if len(useful_feats) <= 150:
                    keep_feats = useful_feats
                else:
                    df_imp_sorted = df_imp.sort_values(by='importances',ascending=False)
                    keep_feats = list(df_imp_sorted['features'].iloc[:150])
        else:
            keep_feats = list(df_imp.loc[df_imp['importances'] >= threshold,'features'])
        
        keep_cats = []

        keep_cats_set = set()
        cat_set = set(X.cat_cols)

        for feat in keep_feats:

            if X.col2type[feat] == CONSTANT.CATEGORY_TYPE:
                if feat in cat_set:
                    if feat not in keep_cats_set:
                        keep_cats_set.add(feat)
                        keep_cats.append(feat)

            elif feat in X.col2source_cat:
                keep_feat = X.col2source_cat[feat]
                if keep_feat in cat_set:
                    if keep_feat not in keep_cats_set:
                        keep_cats_set.add(keep_feat)
                        keep_cats.append(keep_feat)

        drop_feats = list(set(df_imp['features'].tolist()) - set(keep_feats))
        
        drop_feats = list(set(drop_feats) - keep_cats_set)
        self.drop_feats = drop_feats
        log(f'total feat num:{df_imp.shape[0]}, drop feat num:{len(self.drop_feats)}')

        keep_nums = []
        for feat in keep_feats:
            if X.col2type[feat] ==  CONSTANT.NUMERICAL_TYPE:
                keep_nums.append(feat)

        keep_binaries = []
        for feat in keep_feats:
            if X.col2type[feat] ==  CONSTANT.BINARY_TYPE:
                keep_binaries.append(feat)

        assert(len(set(keep_cats) & set(drop_feats))==0)
        assert(len(set(keep_nums) & set(drop_feats))==0)
        assert(len(set(keep_binaries) & set(drop_feats))==0)

        X.reset_combine_cols(keep_cats,keep_nums,keep_binaries)

        rest_cols = len(df_imp) - len(self.drop_feats)
        if rest_cols > col_constrain:
            real_keep_feats = set(df_imp['features'].iloc[:col_constrain].tolist())
            real_drop_feats = list(set(df_imp['features'].tolist()) - real_keep_feats)
            self.drop_feats = real_drop_feats

    @timeclass(cls='LGBFeatureSelectionLast')
    def transform(self,X):
        X.drop_data(self.drop_feats)
        return self.drop_feats
    
    @timeclass(cls='LGBFeatureSelectionLast')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)
        return self.drop_feats

class LGBFeatureSelectionWait(Feat):
    @timeclass(cls='LGBFeatureSelectionWait')
    def fit(self,X,y):
        now = time.time()
        log(f'LGBFeatureSelection:{now-self.config.start_time}')

        threshold = 5
        df_imp = lgb_train(X,y)
        drop_feats = set(df_imp.loc[df_imp['importances'] < threshold,'features'])
        keep_feats = list(df_imp.loc[df_imp['importances'] >= threshold,'features'])

        df_imp.set_index('features',inplace=True)
        for cols in X.wait_selection_cols:
            drops = df_imp.loc[cols].sort_values(by='importances',ascending=False).index[self.config.wait_feat_selection_num:]
            drops = set(drops)
            drop_feats = drop_feats | drops

        keep_cats = []

        keep_cats_set = set()
        cat_set = set(X.cat_cols)
        for feat in keep_feats:

            if X.col2type[feat] == CONSTANT.CATEGORY_TYPE:
                if feat in cat_set:
                    if feat not in keep_cats_set:
                        keep_cats_set.add(feat)
                        keep_cats.append(feat)

            elif feat in X.col2source_cat:
                keep_feat = X.col2source_cat[feat]
                if keep_feat in cat_set:
                    if keep_feat not in keep_cats_set:
                        keep_cats_set.add(keep_feat)
                        keep_cats.append(keep_feat)


        drop_feats = drop_feats - keep_cats_set
        drop_feats = list(drop_feats)
        self.drop_feats = drop_feats
        X.empty_wait_selection_cols()
        log(f'total feat num:{df_imp.shape[0]}, drop feat num:{len(self.drop_feats)}')

        assert(len(set(keep_cats) & set(drop_feats))==0)

    @timeclass(cls='LGBFeatureSelectionWait')
    def transform(self,X):
        X.drop_data(self.drop_feats)
        return self.drop_feats
    
    @timeclass(cls='LGBFeatureSelectionWait')
    def fit_transform(self,X,y):
        self.fit(X,y)
        self.transform(X)
        return self.drop_feats
