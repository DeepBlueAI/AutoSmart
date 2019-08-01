# -*- coding: utf-8 -*-
import lightgbm as lgb
import numpy as np
import CONSTANT
from util import log, timeclass
from .automl import AutoML
import pandas as pd
import gc
from . import autosample
import time
import copy
from sklearn.metrics import roc_auc_score

class AutoLGB(AutoML):
    def __init__(self):
        self.params = {
            "boosting_type": "gbdt",
            "objective": "binary",
            "metric": "auc",
            "verbosity": 1,
            "seed": CONSTANT.SEED,
            "num_threads": CONSTANT.THREAD_NUM
        }

        self.hyperparams = {
            'num_leaves': 31,
            'max_depth': -1,
            'min_child_samples': 20,
            'max_bin':255,
            'subsample': 0.8,
            'subsample_freq': 1,
            'colsample_bytree': 0.8,
            'min_child_weight': 0.001,
            'subsample_for_bin': 200000,
            'min_split_gain': 0.02,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
        }

        self.early_stopping_rounds = 50

    @timeclass(cls='AutoLGB')
    def predict(self,X):
        X = X[self.columns]
        X.columns = self.new_feat_name_cols
        return self.model.predict(X)

    @timeclass(cls='AutoLGB')
    def ensemble_train(self,X,y,categories,config,len_test):
        feat_name = list(X.columns)
        self.ensemble_models = []
        self.ensemble_columns = []
        columns = list(X.columns)
        log(f'lgb training set shape: {X.shape}')
        pos = (y==1).sum()
        neg = (y==0).sum()
        log(f'pos {pos} neg {neg}')

        self.columns = columns
        max_sample_num = len(y)

        feat_name_cols = list(X.columns)
        feat_name_maps = { feat_name_cols[i] : str(i)  for i in range(len(feat_name_cols)) }
        f_feat_name_maps = { str(i) : feat_name_cols[i] for i in range(len(feat_name_cols)) }
        new_feat_name_cols = [ feat_name_maps[i] for i in feat_name_cols ]
        X.columns = new_feat_name_cols
        categories = [ feat_name_maps[i] for i in categories ]
        self.f_feat_name_maps = f_feat_name_maps
        self.new_feat_name_cols = new_feat_name_cols
        
        all_columns = list(X.columns)
        
        start_time = time.time()
        i = 0
        cur_columns = all_columns
        seed = np.random.randint(2019*i,2019*(i+1))
        X_train,y_train = autosample.downsampling(X,y,max_sample_num,seed)
        X_train = X_train[cur_columns]
        gc.collect()
        
        colset = set(X_train.columns)
        cur_categorical = [col for col in categories if col in colset]
        pos = (y_train==1).sum()
        neg = (y_train==0).sum()

        params = self.params
        hyperparams = self.hyperparams
        params['seed'] = seed
        
        X_train = X_train.astype(np.float32)
        gc.collect()
        y_train = y_train.astype(np.float32)
        gc.collect()
        X_train = X_train.values
        gc.collect()
        y_train = y_train.values
        gc.collect()
        
        train_data = lgb.Dataset(X_train, label=y_train,feature_name=feat_name)
        del X_train,y_train
        gc.collect()
        
        model = lgb.train({**params, **hyperparams},
                                train_data,
                                num_boost_round=self.best_iteration,
                                feature_name=cur_columns,
                                categorical_feature=cur_categorical,
                                learning_rates = self.learning_rates[:self.best_iteration])

        self.ensemble_columns.append(cur_columns)
        self.ensemble_models.append(model)
        end_time = time.time()
        
        model_use_time = end_time - start_time
        del train_data
        
        gc.collect()
        
        start_time = time.time()
        temp = X.iloc[:100000]
        
        temp = temp.astype(np.float32)
        gc.collect()
        temp = temp.values
        gc.collect()
        
        model.predict(temp)
        
        end_time = time.time()
        model_test_use_time = (end_time-start_time)
        model_test_use_time = len_test/temp.shape[0] * model_test_use_time
        model_use_time = model_use_time + model_test_use_time
        del temp,model
        
        rest_time = config.budget/10*9-(end_time-config.start_time)
        if rest_time <= 0:
            rest_model_num = 0
        else:
            rest_model_num = int(rest_time // model_use_time)
        
        if rest_model_num >= 50:
            rest_model_num = 50 
            
        if rest_model_num >= 1:
            rest_model_num -= 1

        if not CONSTANT.USE_ENSEMBLE:
            rest_model_num = 0
        
        for i in range(1,rest_model_num+1):

            seed = np.random.randint(2019*i,2019*(i+1))
            
            cur_columns = list(pd.Series(all_columns).sample(frac=0.85,replace=False,random_state=seed))

            X_train,y_train = autosample.downsampling(X,y,max_sample_num,seed)
            X_train = X_train[cur_columns]
            gc.collect()
            
            colset = set(X_train.columns)
            cur_categorical = [col for col in categories if col in colset]

            pos = (y_train==1).sum()
            neg = (y_train==0).sum()

            params = self.params
            hyperparams = self.hyperparams
            params['seed'] = seed
            
            num_leaves = hyperparams['num_leaves']
            num_leaves = num_leaves + np.random.randint(-int(num_leaves/10),int(num_leaves/10)+7)
            
            lrs = np.array(self.learning_rates)
            rands = 1 + 0.2*np.random.rand(len(lrs))
            lrs = list(lrs * rands)
            
            cur_iteration = self.best_iteration
            cur_iteration = cur_iteration + np.random.randint(-30,40)
            if cur_iteration > len(lrs):
                cur_iteration = len(lrs)
            
            if cur_iteration <= 10:
                cur_iteration = self.best_iteration
            
            cur_hyperparams = copy.deepcopy(hyperparams)
            cur_hyperparams['num_leaves'] = num_leaves
            
            X_train = X_train.astype(np.float32)
            gc.collect()
            y_train = y_train.astype(np.float32)
            gc.collect()
            X_train = X_train.values
            gc.collect()
            y_train = y_train.values
            gc.collect()
            
            train_data = lgb.Dataset(X_train, label=y_train,feature_name=cur_columns)
            del X_train,y_train
            gc.collect()
            
            model = lgb.train({**params, **cur_hyperparams},
                                    train_data,
                                    num_boost_round=cur_iteration,
                                    feature_name=cur_columns,
                                    categorical_feature=cur_categorical,
                                    learning_rates = lrs[:cur_iteration])


            self.ensemble_columns.append(cur_columns)
            self.ensemble_models.append(model)

            del train_data
            gc.collect()

        X.columns = self.columns


    @timeclass(cls='AutoLGB')
    def ensemble_predict(self,X):
        X = X[self.columns]
        gc.collect()
        
        X.columns = self.new_feat_name_cols

        preds = []
        for model,cur_cols in zip(self.ensemble_models,self.ensemble_columns):
            gc.collect()
            tX = X[cur_cols]
            gc.collect()
            tX = tX.astype(np.float32)
            gc.collect()
            tX = tX.values
            gc.collect()
            
            preds.append(model.predict( tX ))
            gc.collect()
            
        if len(preds) == 1:
            pred = preds[0]

        if len(preds) > 1:
            total_model_num = len(preds)
            
            main_model_weight = 8 / (8 + 2 * (total_model_num-1))
            rest_model_weight = 2 / (8 + 2 * (total_model_num-1))
            pred = preds[0] * main_model_weight
            for i in range(1,total_model_num):
                pred = pred + rest_model_weight * preds[i]
            
        return pred
    
    @timeclass(cls='AutoLGB')
    def ensemble_predict_test(self,X):
        X = X[self.columns]
        gc.collect()
        
        X.columns = self.new_feat_name_cols
        log(f'ensemble models {len(self.ensemble_models)}')
        preds = []
        for model,cur_cols in zip(self.ensemble_models,self.ensemble_columns):
            gc.collect()
            tX = X[cur_cols]
            gc.collect()
            tX = tX.astype(np.float32)
            gc.collect()
            tX = tX.values
            gc.collect()
            
            preds.append(model.predict( tX ))
            gc.collect()
            
        if len(preds) == 1:
            pred = preds[0]

        if len(preds) > 1:
            total_model_num = len(preds)
            
            main_model_weight = 8 / (8 + 2 * (total_model_num-1))
            rest_model_weight = 2 / (8 + 2 * (total_model_num-1))
            pred = preds[0] * main_model_weight
            for i in range(1,total_model_num):
                pred = pred + rest_model_weight * preds[i]
            
        return pred,preds[0]
    
    def get_log_lr(self,num_boost_round,max_lr,min_lr):
        learning_rates = [max_lr+(min_lr-max_lr)/np.log(num_boost_round)*np.log(i) for i in range(1,num_boost_round+1)]
        return learning_rates

    def set_num_leaves(self,X,y):
        t = len(y)
        t = X.shape[1]*(t/40000)
        level = t**0.225 + 1.5
        num_leaves = int(2**level) + 10
        num_leaves = min(num_leaves, 128)
        num_leaves = max(num_leaves, 32)
        self.hyperparams['num_leaves'] = num_leaves

    def set_min_child_samples(self, X,y ):
        min_child_samples = ( (X.shape[0]/20000)**0.6 ) *15
        min_child_samples = int(min_child_samples)
        min_child_samples = min(min_child_samples, 150)
        min_child_samples = max(min_child_samples, 15)

        self.hyperparams['min_child_samples'] = min_child_samples

    @timeclass(cls='AutoLGB')
    def lr_opt(self,train_data,valid_data,categories):
        params = self.params
        hyperparams = self.hyperparams

        max_lrs = [0.1,0.08,0.05,0.02]
        min_lrs = [0.04,0.02,0.01,0.005]

        num_boost_round = self.num_boost_round
        max_num_boost_round = min(400,num_boost_round)
        best_score = -1
        best_loop = -1
        lr = None

        scores = []
        lrs = []
        for max_lr,min_lr in zip(max_lrs,min_lrs):
            learning_rates = self.get_log_lr(num_boost_round,max_lr,min_lr)
            
            model = lgb.train({**params, **hyperparams}, train_data, num_boost_round=max_num_boost_round,\
                                  categorical_feature=categories,learning_rates = learning_rates[:max_num_boost_round]
                                  )
            pred = model.predict(valid_data.data)
            score = roc_auc_score(valid_data.label,pred)
            scores.append(score)
            lrs.append(learning_rates)
            del model, pred
            gc.collect()

        best_loop = np.argmax(scores)
        best_score = np.max(scores)
        lr = lrs[best_loop]
        log(f'scores {scores}')
        log(f'loop {best_loop}')
        log(f'lr max {lr[0]} min {lr[-1]}')
        log(f'lr best score {best_score}')
        return lr

    @timeclass(cls='AutoLGB')
    def num_leaves_opt(self,train_data,valid_data,categories):
        params = self.params
        hyperparams = self.hyperparams
        num_leaves = [31,63,127,255]

        num_boost_round = 500
        best_iteration = -1
        i = 0
        best_score = -1
        best_loop = -1
        best_num_leaves = None

        for leaves in num_leaves:
            hyperparams['num_leaves'] = leaves
            model = lgb.train({**params, **hyperparams}, train_data, num_boost_round=num_boost_round,\
                                  valid_sets=[valid_data], early_stopping_rounds=self.early_stopping_rounds, verbose_eval=100,\
                                  categorical_feature=categories,learning_rates = self.learning_rates
                                  )

            score = model.best_score["valid_0"][params["metric"]]
            if score > best_score:
                best_num_leaves = leaves
                best_iteration = model.best_iteration
                best_score = score
                best_loop = i

        return best_num_leaves

    @timeclass(cls='AutoLGB')
    def subsample_opt(self,num_samples):
        samples = num_samples
        if samples > 1000000:
            samples = 1000000

        if samples<200000:
            subsample = 0.95 - samples/1000000
            return subsample

        subsample = 0.85-samples/2500000
        return subsample

    @timeclass(cls='AutoLGB')
    def colsample_bytree_opt(self,num_feature):
        if num_feature > 500:
            num_feature = 500

        if num_feature > 100:
            colsample_bytree = 0.8 - num_feature/2000
        else:
            colsample_bytree = 0.95 - num_feature/500

        return colsample_bytree

    @timeclass(cls='AutoLGB')
    def param_compute(self,X,y,categories,config):
        feat_name = list(X.columns)
        colsample_bytree = self.colsample_bytree_opt(X.shape[1])
        self.hyperparams['colsample_bytree'] = colsample_bytree
        
        max_sample_num = len(y)
        subsample = self.subsample_opt(autosample.downsampling_y(y,max_sample_num).shape[0])
        self.hyperparams['subsample'] = subsample
        
        max_sample_num = min(len(y),50000)
        X_sample,y_sample = autosample.downsampling(X,y,max_sample_num)
        gc.collect()
        params = self.params
        
        start_time = time.time()
        X_sample = X_sample.astype(np.float32)
        gc.collect()
        y_sample = y_sample.astype(np.float32)
        gc.collect()
        X_sample = X_sample.values
        gc.collect()
        y_sample = y_sample.values
        gc.collect()
        end_time = time.time()
        transfer_time = end_time-start_time
        
        time_number_boost_round1 = 15
        start_time = time.time()
        train_data = lgb.Dataset(X_sample, label=y_sample,feature_name=feat_name)
        
        gc.collect()
        
        lgb.train({**params, **self.hyperparams}, train_data, num_boost_round=time_number_boost_round1,\
                                  categorical_feature=categories,)
        
        end_time = time.time()
        
        model_use_time1 = end_time - start_time
        
        time_number_boost_round2 = time_number_boost_round1*2
        
        del train_data
        gc.collect()
        
        start_time = time.time()
        train_data = lgb.Dataset(X_sample, label=y_sample,feature_name=feat_name)
        del X_sample,y_sample
        gc.collect()
        lgb.train({**params, **self.hyperparams}, train_data, num_boost_round=time_number_boost_round2,\
                                  categorical_feature=categories,)
        
        del train_data
        gc.collect()
        end_time = time.time()
        
        model_use_time2 = end_time - start_time
        
        boost_time = (model_use_time2 - model_use_time1)
        boost_round = time_number_boost_round2 - time_number_boost_round1
        preprocess_time = model_use_time1 - boost_time
        model_sample_time = 4 * (transfer_time + preprocess_time + (boost_time * (400/boost_round))) + 5
        
        max_sample_num = len(y)
        X,y = autosample.downsampling(X,y,max_sample_num)

        gc.collect()
        pos = (y==1).sum()
        neg = (y==0).sum()
        
        gc.collect()
        params = self.params
        
        time_number_boost_round1 = 15
        
        start_time = time.time()
        X = X.astype(np.float32)
        gc.collect()
        y = y.astype(np.float32)
        gc.collect()
        X = X.values
        gc.collect()
        y = y.values
        gc.collect()
        end_time = time.time()
        
        transfer_time = end_time-start_time
        
        start_time = time.time()
        train_data = lgb.Dataset(X, label=y,feature_name=feat_name)
        
        gc.collect()
        
        
        lgb.train({**params, **self.hyperparams}, train_data, num_boost_round=time_number_boost_round1,\
                                  categorical_feature=categories,)
        
        del train_data
        gc.collect()
        end_time = time.time()
        
        model_use_time1 = end_time - start_time
        
        time_number_boost_round2 = time_number_boost_round1*2
        
        start_time = time.time()
        train_data = lgb.Dataset(X, label=y,feature_name=feat_name)
        del X,y
        gc.collect()
        lgb.train({**params, **self.hyperparams}, train_data, num_boost_round=time_number_boost_round2,\
                                  categorical_feature=categories,)
        
        del train_data
        gc.collect()
        end_time = time.time()
        
        model_use_time2 = end_time - start_time
        
        boost_time = (model_use_time2 - model_use_time1)
        boost_round = time_number_boost_round2 - time_number_boost_round1
        preprocess_time = model_use_time1 - boost_time
        
        rest_time = config.budget/10*9-(end_time-config.start_time)-model_sample_time-10
        
        self.num_boost_round = 20
        for number_boost_round in [700,600,500,400,300,200,100,50]:
            real_model_time = (transfer_time + preprocess_time + (boost_time * (number_boost_round/boost_round)))
            if real_model_time > rest_time:
                continue
            else:
                self.num_boost_round = number_boost_round
                break
            
        gc.collect()
    
    @timeclass(cls='AutoLGB')
    def param_opt(self,X_train,y_train,X_valid,y_valid,categories):
        feat_name = list(X_train.columns)

        pos = (y_train==1).sum()
        neg = (y_train==0).sum()
        val_pos = (y_valid==1).sum()
        val_neg = (y_valid==0).sum()

        max_sample_num = min(len(y_train),50000)
        X,y = autosample.downsampling(X_train,y_train,max_sample_num)

        pos = (y==1).sum()
        neg = (y==0).sum()

        train_data = lgb.Dataset(X, label=y,feature_name=feat_name)
        del X,y
        gc.collect()

        valid_data = lgb.Dataset(X_valid, label=y_valid,feature_name=feat_name,free_raw_data=False)
        del X_valid,y_valid
        gc.collect()

        lr = self.lr_opt(train_data,valid_data,categories)
        self.learning_rates = lr
        
        self.best_iteration = self.num_boost_round
        
        del train_data
        gc.collect()
         
        num_boost_round = self.num_boost_round
        params = self.params
        max_sample_num = len(y_train)
         
        X,y = autosample.downsampling(X_train,y_train,max_sample_num)
        del X_train,y_train
         
        gc.collect()
        pos = (y==1).sum()
        neg = (y==0).sum()
        
        X = X.astype(np.float32)
        gc.collect()
        y = y.astype(np.float32)
        gc.collect()
        X = X.values
        gc.collect()
        y = y.values
        gc.collect()
        
        train_data = lgb.Dataset(X, label=y,feature_name=feat_name)
        
        del X,y
        gc.collect()
        
        model = lgb.train({**params, **self.hyperparams}, train_data, num_boost_round=num_boost_round,\
                                   valid_sets=[valid_data], early_stopping_rounds=self.early_stopping_rounds, verbose_eval=100,\
                                   categorical_feature=categories,learning_rates = self.learning_rates
                                   )
        gc.collect()
        
        best_model = model
         
        best_score = model.best_score["valid_0"][params["metric"]]
        
        if model.best_iteration > 50:
            self.best_iteration = model.best_iteration
        elif model.current_iteration() > 50:
            self.best_iteration = model.current_iteration()
        else:
            self.best_iteration = 50
        
        return best_model,best_score

    def get_importances(self):
        model = self.model
        importances = pd.DataFrame({'features':[ self.f_feat_name_maps[i] for i in model.feature_name() ] ,
                                'importances':model.feature_importance()})

        importances.sort_values('importances',ascending=False,inplace=True)

        return importances

    @timeclass(cls='AutoLGB')
    def ensemble_predict_train(self,X):
        X = X[X.columns]
        X.columns = self.new_feat_name_cols

        preds = []
        for model in self.ensemble_models:
            preds.append(model.predict(X))

        pred = np.stack(preds,axis=1).mean(axis=1)
        return pred

    def get_ensemble_importances(self):
        model = self.ensemble_models[0]
        importances = pd.DataFrame({'features':[ self.f_feat_name_maps[i] for i in model.feature_name() ] ,
                                'importances':model.feature_importance()})

        importances.sort_values('importances',ascending=False,inplace=True)

        return importances
    
    @timeclass(cls='AutoLGB')
    def param_opt_new(self,X_train,y_train,X_valid,y_valid,categories):
        feat_name = list(X_train.columns)

        pos = (y_train==1).sum()
        neg = (y_train==0).sum()
        val_pos = (y_valid==1).sum()
        val_neg = (y_valid==0).sum()
        log(f'training set pos {pos} neg {neg}')
        log(f'validation set pos {val_pos} neg {val_neg}')

        max_sample_num = min(len(y_train),50000)
        X,y = autosample.downsampling(X_train,y_train,max_sample_num)

        pos = (y==1).sum()
        neg = (y==0).sum()
        log(f'opt downsampling set pos {pos} neg {neg}')

        X = X.astype(np.float32)
        gc.collect()
        y = y.astype(np.float32)
        gc.collect()
        X = X.values
        gc.collect()
        y = y.values
        gc.collect()
        
        train_data = lgb.Dataset(X, label=y,feature_name=feat_name)
        del X,y
        gc.collect()

        valid_data = lgb.Dataset(X_valid, label=y_valid,feature_name=feat_name,free_raw_data=False)
        del X_valid,y_valid
        gc.collect()

        lr = self.lr_opt(train_data,valid_data,categories)
        del train_data
        gc.collect()
        self.learning_rates = lr
        
        self.best_iteration = self.num_boost_round
        log(f'pass round opt, use best iteration as {self.best_iteration}')
