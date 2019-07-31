# -*- coding: utf-8 -*-

from feat.feat_pipeline import FeatPipeline
from util import timeclass

class FeatEngine:
    def __init__(self, feat_pipeline: FeatPipeline, config):
        self.feat_pipeline = feat_pipeline
        self.config = config
        
    @timeclass(cls='FeatEngine')
    def fit_order1(self,table,y):
        self.feats_order1 = []
        for feat_cls in self.feat_pipeline.order1s:
            feat = feat_cls(self.config)
            feat.fit(table,y)
            self.feats_order1.append(feat)
            
    @timeclass(cls='FeatEngine')       
    def transform_order1(self,table):
        for feat in self.feats_order1:
            feat.transform(table)
    
    @timeclass(cls='FeatEngine')
    def fit_transform_order1(self,table,y):
        self.feats_order1 = []
        for feat_cls in self.feat_pipeline.order1s:
            feat = feat_cls(self.config)
            feat.fit_transform(table,y)
            self.feats_order1.append(feat)
    

    @timeclass(cls='FeatEngine')
    def fit_keys_order2(self,table,y):
        self.feats_keys_order2 = []
        for feat_cls in self.feat_pipeline.keys_order2s:
            feat = feat_cls(self.config)
            feat.fit(table,y)
            self.feats_keys_order2.append(feat)
            
    @timeclass(cls='FeatEngine')       
    def transform_keys_order2(self,table):
        for feat in self.feats_keys_order2:
            feat.transform(table)

    @timeclass(cls='FeatEngine')
    def fit_transform_keys_order2(self,table,y,sample=False,selection=True):
        if not self.feat_pipeline.keys_order2s:
            return
        
        if sample:
            self.feats_keys_order2 = []
            self.keys_order2_new_cols = []
            for feat_cls in self.feat_pipeline.keys_order2s[:-1]:
                feat = feat_cls(self.config)
                new_cols = feat.fit_transform(table,y)
                self.feats_keys_order2.append(feat)
                self.keys_order2_new_cols.append(set(new_cols))

            feat_cls = self.feat_pipeline.keys_order2s[-1]
            feat = feat_cls(self.config)
            drop_feats = set(feat.fit_transform(table,y))
            self.feats_keys_order2.append(feat)
            for i in range(len(self.keys_order2_new_cols)):
                self.keys_order2_new_cols[i] = (set(self.keys_order2_new_cols[i]) - drop_feats)

        if not sample:
            if selection:
                self.feats_keys_order2 = []
                for i,feat_cls in enumerate(self.feat_pipeline.keys_order2s):
                    feat = feat_cls(self.config)
                    feat.fit_transform(table,y)
                    self.feats_keys_order2.append(feat)
            if not selection:
                for i,feat_cls in enumerate(self.feat_pipeline.keys_order2s[:-1]):
                    feat = feat_cls(self.config)
                    feat.fit_transform(table,y,self.keys_order2_new_cols[i])
                    self.feats_keys_order2.append(feat)

    @timeclass(cls='FeatEngine')
    def fit_all_order2(self,table,y):
        self.feats_all_order2 = []
        for feat_cls in self.feat_pipeline.all_order2s:
            feat = feat_cls(self.config)
            feat.fit(table,y)
            self.feats_all_order2.append(feat)
            
    @timeclass(cls='FeatEngine')       
    def transform_all_order2(self,table):
        for feat in self.feats_all_order2:
            feat.transform(table)

    @timeclass(cls='FeatEngine')
    def fit_transform_all_order2(self,table,y,sample=False,selection=True):
        if not self.feat_pipeline.all_order2s:
            return
        
        if sample:
            self.feats_all_order2 = []
            self.all_order2_new_cols = []
            for feat_cls in self.feat_pipeline.all_order2s[:-1]:
                feat = feat_cls(self.config)
                new_cols = feat.fit_transform(table,y)
                self.feats_all_order2.append(feat)
                self.all_order2_new_cols.append(set(new_cols))

            feat_cls = self.feat_pipeline.all_order2s[-1]
            feat = feat_cls(self.config)
            drop_feats = set(feat.fit_transform(table,y))
            self.feats_all_order2.append(feat)
            for i in range(len(self.all_order2_new_cols)):
                self.all_order2_new_cols[i] = set(self.all_order2_new_cols[i]) - drop_feats

        if not sample:
            if selection:
                self.feats_all_order2 = []
                for i,feat_cls in enumerate(self.feat_pipeline.all_order2s):
                    feat = feat_cls(self.config)
                    feat.fit_transform(table,y)
                    self.feats_all_order2.append(feat)
            if not selection:
                for i,feat_cls in enumerate(self.feat_pipeline.all_order2s[:-1]):
                    feat = feat_cls(self.config)
                    feat.fit_transform(table,y,self.all_order2_new_cols[i])
                    self.feats_all_order2.append(feat)
    
    @timeclass(cls='FeatEngine')
    def fit_keys_order3(self,table,y):
        self.feats_keys_order3 = []
        for feat_cls in self.feat_pipeline.keys_order3s:
            feat = feat_cls(self.config)
            feat.fit(table,y)
            self.feats_keys_order3.append(feat)
            
    @timeclass(cls='FeatEngine')       
    def transform_keys_order3(self,table):
        for feat in self.feats_keys_order3:
            feat.transform(table)
    
    @timeclass(cls='FeatEngine')
    def fit_transform_keys_order3(self,table,y):
        self.feats_keys_order3 = []
        for feat_cls in self.feat_pipeline.keys_order3s:
            feat = feat_cls(self.config)
            feat.fit_transform(table,y)
            self.feats_keys_order3.append(feat)
    

    @timeclass(cls='FeatEngine')
    def fit_post_order1(self,table,y):
        self.feats_post_order1 = []
        for feat_cls in self.feat_pipeline.post_order1s:
            feat = feat_cls(self.config)
            feat.fit(table,y)
            self.feats_post_order1.append(feat)
            
    @timeclass(cls='FeatEngine')       
    def transform_post_order1(self,table):
        for feat in self.feats_post_order1:
            feat.transform(table)
    
    @timeclass(cls='FeatEngine')
    def fit_transform_post_order1(self,table,y):
        self.feats_post_order1 = []
        for feat_cls in self.feat_pipeline.post_order1s:
            feat = feat_cls(self.config)
            feat.fit_transform(table,y)
            self.feats_post_order1.append(feat)
        
    @timeclass(cls='FeatEngine')
    def fit_merge_order1(self,table,y):
        self.feats_merge_order1 = []
        for feat_cls in self.feat_pipeline.merge_order1s:
            feat = feat_cls(self.config)
            feat.fit(table,y)
            self.feats_merge_order1.append(feat)
            
    @timeclass(cls='FeatEngine')       
    def transform_merge_order1(self,table):
        for feat in self.feats_merge_order1:
            feat.transform(table)
    
    @timeclass(cls='FeatEngine')
    def fit_transform_merge_order1(self,table,y):
        self.feats_merge_order1 = []
        for feat_cls in self.feat_pipeline.merge_order1s:
            feat = feat_cls(self.config)
            feat.fit_transform(table,y)
            self.feats_merge_order1.append(feat)