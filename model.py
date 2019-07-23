import os

os.system("pip3 install hyperopt")
os.system("pip3 install lightgbm")
os.system("pip3 install pandas==0.24.2")
os.system('pip3 install joblib')

import numpy as np
import pandas as pd
import copy
import CONSTANT
from util import log, timeclass
from table.graph import Graph
from sklearn.metrics import roc_auc_score
from feat.merge_feat_pipeline import DeafultMergeFeatPipeline
from feat.feat_pipeline import DefaultFeatPipeline

from merger import Merger
from feat_engine import FeatEngine
from model_input import FeatOutput
from automl.model_selection import time_train_test_split
from automl.auto_lgb import AutoLGB
from PATHS import feature_importance_path,version
from datetime import datetime
import gc
from config import Config
import time

class Model:
    auc = []
    ensemble_auc = []
    ensemble_train_auc = []

    def __init__(self, info):
        self.info = copy.deepcopy(info)
        self.tables = None

    def shuffle(self,X,y,random_state):
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X.iloc[idx]
        y = y.iloc[idx]
        return X,y

    def release_tables(self,Xs,graph):
        
        for name in graph.tables:
            del Xs[name]
            del graph.name2table[name]

        gc.collect()

    @timeclass(cls='Model')
    def my_fit(self, Xs, y, time_ramain,X_test):
        np.random.seed(CONSTANT.SEED)

        split = CONSTANT.SPLIT

        self.split = split

        log(f'split {split}')

        if split == -1:
            config = Config(time.time(),self.info['time_budget'])
            
            X_test.index = -X_test.index-1

            main_shape = Xs[CONSTANT.MAIN_TABLE_NAME].shape[0]
            main_max_shape = 2888888
            main_min_shape = min( main_shape,100000 )
            
            test_shape = X_test.shape[0]
            max_accept_shape = 3999999
            
            if main_shape + test_shape > max_accept_shape: 
                sample_main_shape = max_accept_shape - test_shape
                if sample_main_shape > main_max_shape:
                    sample_main_shape = main_max_shape
                if sample_main_shape < main_min_shape:
                    sample_main_shape = main_min_shape
                log(f'start sample main table. origin main shape {main_shape} test shape {test_shape} sample rows num {sample_main_shape}')
                if 'time_col' in self.info:
                    key_time_col = self.info['time_col']
                    if key_time_col in Xs[CONSTANT.MAIN_TABLE_NAME].columns:
                        Xs[CONSTANT.MAIN_TABLE_NAME].sort_values(by=key_time_col,inplace=True)
                Xs[CONSTANT.MAIN_TABLE_NAME] = Xs[CONSTANT.MAIN_TABLE_NAME].iloc[-sample_main_shape:]
                gc.collect()


            Xs[CONSTANT.MAIN_TABLE_NAME] = pd.concat([Xs[CONSTANT.MAIN_TABLE_NAME], X_test])

            X_test.drop(X_test.columns,axis=1,inplace=True)
            gc.collect()

            graph = Graph(self.info,Xs)
            graph.sort_tables()
            train_index = Xs[CONSTANT.MAIN_TABLE_NAME].index[Xs[CONSTANT.MAIN_TABLE_NAME].index>=0]
            y = y.loc[train_index]
            test_index = Xs[CONSTANT.MAIN_TABLE_NAME].index[Xs[CONSTANT.MAIN_TABLE_NAME].index<0]

            graph.preprocess_fit_transform()
            gc.collect()

            merge_feat_pipeline = DeafultMergeFeatPipeline()
            merger = Merger(merge_feat_pipeline)

            merger.merge_table(graph)
            main_table = merger.merge_to_main_fit_transform(graph)
            self.release_tables(Xs,graph)
            del merger
            del graph
            gc.collect()

            feat_pipeline = DefaultFeatPipeline()
            feat_engine = FeatEngine(feat_pipeline,config)
            feat_engine.fit_transform_order1(main_table,y)
            
            sample_for_combine_features = True
            
            if sample_for_combine_features:
                main_data = main_table.data
                train_data = main_data.loc[main_data.index>=0]

                del main_data

                sample_num = CONSTANT.SAMPLE_NUM
                train_shape = train_data.shape 
                
                if train_shape[0] <= sample_num:
                    sample_for_combine_features = False
                else:
                    data_tail_new = train_data.iloc[-sample_num:]
                    
                    gc.collect()
                    
                    y_tail_new = y.loc[data_tail_new.index]
                    
                    table_tail_new = copy.deepcopy(main_table)
                    table_tail_new.data = data_tail_new
                    
                    del data_tail_new
                    gc.collect()

                    feat_engine.fit_transform_all_order2(table_tail_new,y_tail_new,sample=True)
                    feat_engine.fit_transform_keys_order2(table_tail_new,y_tail_new,sample=True)
                    
                    del table_tail_new,y_tail_new
                    gc.collect()

                    feat_engine.fit_transform_all_order2(main_table,y,selection=False)
                    feat_engine.fit_transform_keys_order2(main_table,y,selection=False)

                    feat_engine.fit_transform_post_order1(main_table,y)
                    
            if not sample_for_combine_features:
                gc.collect()

                feat_engine.fit_transform_all_order2(main_table,y)
                feat_engine.fit_transform_keys_order2(main_table,y)
                
                feat_engine.fit_transform_keys_order3(main_table,y)
                feat_engine.fit_transform_post_order1(main_table,y)


            del feat_engine
            gc.collect()


            X_test = main_table.data.loc[test_index]
            main_table.data = main_table.data.loc[train_index]

            gc.collect()

            test_table = copy.deepcopy(main_table)
            test_table.data = X_test
            self.test_table = test_table
            len_test = X_test.shape[0]
            gc.collect()

            feat_engine = FeatEngine(feat_pipeline,config)
            feat_engine.fit_transform_merge_order1(main_table,y)
            self.feat_engine = feat_engine

            feat_output = FeatOutput()
            self.feat_output = feat_output
            X,y,categories = feat_output.final_fit_transform_output(main_table,y)

            del main_table
            gc.collect()
            
            lgb = AutoLGB()
            
            lgb.param_compute(X,y,categories,config)
            X_train,y_train,X_test,y_test = time_train_test_split(X,y,test_rate=0.2)
            
            lgb.param_opt_new(X_train,y_train,X_test,y_test,categories)
            
            gc.collect()
            
            del X_train,y_train,X_test,y_test

            gc.collect()

            X,y = self.shuffle(X,y,2019)
            gc.collect()
            
            lgb.ensemble_train(X,y,categories,config,len_test)
                
            gc.collect()

            importances = lgb.get_ensemble_importances()

            self.model = lgb
            del X,y
            
        elif split == -2:

            config = Config(time.time(),self.info['time_budget'])

            Xs[CONSTANT.MAIN_TABLE_NAME] = pd.concat([Xs[CONSTANT.MAIN_TABLE_NAME], ])

            gc.collect()

            graph = Graph(self.info,Xs)
            graph.sort_tables()
            train_index = Xs[CONSTANT.MAIN_TABLE_NAME].index[Xs[CONSTANT.MAIN_TABLE_NAME].index>=0]
            y = y.loc[train_index]

            graph.preprocess_fit_transform()
            gc.collect()

            merge_feat_pipeline = DeafultMergeFeatPipeline()
            merger = Merger(merge_feat_pipeline)

            merger.merge_table(graph)
            main_table = merger.merge_to_main_fit_transform(graph)
            self.release_tables(Xs,graph)
            del merger
            del graph
            gc.collect()

            feat_pipeline = DefaultFeatPipeline()
            feat_engine = FeatEngine(feat_pipeline,config)
            feat_engine.fit_transform_order1(main_table,y)
            
            sample_for_combine_features = True

            if sample_for_combine_features:
                main_data = main_table.data
                train_data = main_data.loc[main_data.index>=0]

                del main_data

                sample_num = CONSTANT.SAMPLE_NUM
                train_shape = train_data.shape 
                
                if train_shape[0] <= sample_num:
                    sample_for_combine_features = False
                else:
                    data_tail_new = train_data.iloc[-sample_num:]

                    gc.collect()
                    log(f'sample data shape {data_tail_new.shape}')
                    
                    y_tail_new = y.loc[data_tail_new.index]
                    
                    table_tail_new = copy.deepcopy(main_table)
                    table_tail_new.data = data_tail_new
                    
                    del data_tail_new
                    gc.collect()
                    
                    feat_engine.fit_transform_all_order2(table_tail_new,y_tail_new,sample=True)
                    feat_engine.fit_transform_keys_order2(table_tail_new,y_tail_new,sample=True)
                    
                    del table_tail_new,y_tail_new
                    gc.collect()
    
                    feat_engine.fit_transform_all_order2(main_table,y,selection=False)
                    feat_engine.fit_transform_keys_order2(main_table,y,selection=False)
                    feat_engine.fit_transform_post_order1(main_table,y)
                    
            if not sample_for_combine_features:
                gc.collect()

                feat_engine.fit_transform_all_order2(main_table,y)
                feat_engine.fit_transform_keys_order2(main_table,y)
                feat_engine.fit_transform_keys_order3(main_table,y)
                feat_engine.fit_transform_post_order1(main_table,y)

            del feat_engine
            gc.collect()

            main_table.data = main_table.data.loc[train_index]

            gc.collect()

            def split_table(table,y):
                X = table.data
                X_train,y_train,X_test,y_test = time_train_test_split(X,y,shuffle=False,test_rate=0.2)
                table1 = copy.deepcopy(table)
                table1.data = X_train
                table2 = copy.deepcopy(table)
                table2.data = X_test
                return table1,y_train,table2,y_test

            table1,y_train,table2,y_test = split_table(main_table,y)

            feat_engine = FeatEngine(feat_pipeline,config)
            feat_engine.fit_transform_merge_order1(table1,y_train)
            self.feat_engine = feat_engine

            feat_output = FeatOutput()
            self.feat_output = feat_output

            X_train,y_train,categories = feat_output.fit_transform_output(table1,y_train)

            gc.collect()
            self.feat_engine.transform_merge_order1(table2)
            X_test = self.feat_output.transform_output(table2)
            
            lgb = AutoLGB()
            
            lgb.param_compute(X_train,y_train,categories,config)
            
            lgb.param_opt_new(X_train,y_train,X_test,y_test,categories)
                
            len_test = X_test.shape[0]

            lgb.ensemble_train(X_train,y_train,categories,config,len_test)
            gc.collect()

            pred,pred0 = lgb.ensemble_predict_test(X_test)

            auc = roc_auc_score(y_test,pred0)
            print('source AUC:',auc)
            
            auc = roc_auc_score(y_test,pred)
            Model.ensemble_auc.append(auc)
            print('ensemble AUC:',auc)
            
            importances = lgb.get_ensemble_importances()

            self.model = lgb

            del X_train,y_train,X_test,y_test
            gc.collect()

        paths = os.path.join(feature_importance_path,version)
        if not os.path.exists(paths):
            os.makedirs(paths)
        importances.to_csv(os.path.join(paths,'{}_importances.csv'.format(datetime.now().strftime('%Y%m%d%H%M%S'))),index=False)

    @timeclass(cls='Model')
    def fit(self, Xs, y, time_remain):
        self.Xs = Xs
        self.y = y
        self.time_remain = time_remain

    @timeclass(cls='Model')
    def predict(self, X_test, time_remain):

        self.my_fit(self.Xs, self.y, self.time_remain, X_test)
        
        gc.collect()

        if self.split != -2:
            main_table = self.test_table
            self.feat_engine.transform_merge_order1(main_table)
            X = self.feat_output.transform_output(main_table)

            X.index = -(X.index+1)
            X.sort_index(inplace=True)

            result = self.model.ensemble_predict(X)
            return pd.Series(result)
        
        else:
            return pd.Series()
