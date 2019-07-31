# -*- coding: utf-8 -*-


from util import log, timeit, timeclass
import numpy as np
import gc
import sys

class FeatOutput:
    @timeclass(cls='FeatOutput')
    def transform_output(self,table):
        X = table.data

        self.drop_non_numerical_column(table,X)
        self.drop_post_drop_column(table,X)

        return X

    @timeclass(cls='FeatOutput')

    def fit_transform_output(self,table,y):
        X = table.data.copy()

        self.drop_non_numerical_column(table,X)
        self.drop_post_drop_column(table,X)

        categories =  self.get_categories(table,X)

        return X,y,categories

    def final_fit_transform_output(self,table,y):
        X = table.data


        self.drop_non_numerical_column(table,X)
        self.drop_post_drop_column(table,X)

        categories =  self.get_categories(table,X)

        return X,y,categories

    @timeclass(cls='FeatOutput')
    def fillna(self,table,X):
        for col in table.num_cols:
            X[col] = X[col].fillna(X[col].mean())


    def get_categories(self,table,X):
        categories = []
        col_set = set(X.columns)
        for col in table.cat_cols:
            if col in col_set:
                if X[col].nunique() <= 15:
                    categories.append(col)


        return categories

    @timeclass(cls='FeatOutput')
    def drop_non_numerical_column(self,table,X):
        if table.key_time_col is not None:

            X.drop(table.key_time_col,axis=1,inplace=True)
            gc.collect()

        if len(table.time_cols) != 0:
            X.drop(table.time_cols,axis=1,inplace=True)

        if len(table.multi_cat_cols) != 0:
            X.drop(table.multi_cat_cols,axis=1,inplace=True)

    @timeclass(cls='FeatOutput')
    def drop_post_drop_column(self,table,X):
        if len(table.post_drop_set) != 0:
            drop_cols = list(table.post_drop_set)
            X.drop(drop_cols,axis=1,inplace=True)
            log(f'post drop cols:{drop_cols}')

    @timeclass(cls='FeatOutput')
    def drop_cat_column(self,table,X):
        X.drop(list(set(table.session_cols + table.user_cols + table.key_cols + table.cat_cols)&set(X.columns)),axis=1,inplace=True)

    @timeclass(cls='FeatOutput')
    def cat_hash(self,table,X):
        for col in table.user_cols + table.key_cols + table.cat_cols:
            X[col] = X[col] % 15

    @timeclass(cls='FeatOutput')
    def cat_process(self,train_table,test_table):
        X = train_table

        train = train_table.data
        test = test_table.data
        for col in X.user_cols + X.key_cols + X.cat_cols:
            inter = set(train[col].unique()) & set(test[col].unique())
            train.loc[~(train[col].isin(inter)),col] = np.nan
            test.loc[~(test[col].isin(inter)),col] = np.nan

    @timeclass(cls='FeatOutput')
    def drop_tail(self,train_table,test_table):
        X = train_table

        train = train_table.data
        test = test_table.data
        for col in X.key_cols + X.cat_cols:
            vc = train[col].value_counts()
            vc.loc[vc==1] = np.nan
            train[col] = train[col].map(vc)
            test[col] = test[col].map(vc)
