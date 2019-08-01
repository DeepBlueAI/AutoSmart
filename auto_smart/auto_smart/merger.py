# -*- coding: utf-8 -*-
import pandas as pd

import CONSTANT
from util import log, timeclass
from feat.merge_feat_pipeline import MergeFeatPipeline
import copy
import gc
from data_tools import downcast

class Merger:
    def __init__(self,merge_feat_pipeline: MergeFeatPipeline):
        self.merge_feat_pipeline = merge_feat_pipeline



    @timeclass(cls='Merger')
    def merge(self,key,u,v,ttype,z2f):
        feats = []
        col2type = {}
        col2groupby = {}
        col2block = {}

        if u.key_time_col is not None and v.key_time_col is not None and ttype=='many_to_many':

            if z2f and self.merge_timem2m and (key in u.user_cols):
                self.merge_timem2m = False
                for merge_feat_cls in self.merge_feat_pipeline.newTimeM2Ms:
                    merge_feat = merge_feat_cls(key)
                    merge_feat.fit_transform(u,v)            
            
            for merge_feat_cls in self.merge_feat_pipeline.preM2Ms:
                merge_feat = merge_feat_cls(key)
                merge_feat.fit_transform(u,v)

            for merge_feat_cls in self.merge_feat_pipeline.M2Ms:
                merge_feat = merge_feat_cls(key)
                v_feat,v_col2type,v_col2block = merge_feat.fit_transform(u,v)
                feats.append(v_feat)
                col2type.update(v_col2type)
                col2block.update(v_col2block)

        elif ttype == 'one_to_one':
            for merge_feat_cls in self.merge_feat_pipeline.preO2Os:
                merge_feat = merge_feat_cls(key)
                merge_feat.fit_transform(u,v)

            for merge_feat_cls in self.merge_feat_pipeline.O2Os:
                merge_feat = merge_feat_cls(key)
                v_feat,v_col2type,v_col2block = merge_feat.fit_transform(u,v)
                feats.append(v_feat)
                col2type.update(v_col2type)
                col2block.update(v_col2block)

        elif ttype == 'many_to_one':
            for merge_feat_cls in self.merge_feat_pipeline.preM2Os:
                merge_feat = merge_feat_cls(key)
                merge_feat.fit_transform(u,v)

            for merge_feat_cls in self.merge_feat_pipeline.M2Os:
                merge_feat = merge_feat_cls(key)
                v_feat,v_col2type,v_col2block = merge_feat.fit_transform(u,v)
                feats.append(v_feat)
                col2type.update(v_col2type)
                col2block.update(v_col2block)

        elif ttype == 'one_to_many':
            for merge_feat_cls in self.merge_feat_pipeline.preO2Ms:
                merge_feat = merge_feat_cls(key)
                merge_feat.fit_transform(u,v)

            for merge_feat_cls in self.merge_feat_pipeline.O2Ms:
                merge_feat = merge_feat_cls(key)
                v_feat,v_col2type,v_col2block = merge_feat.fit_transform(u,v)
                feats.append(v_feat)
                col2type.update(v_col2type)
                col2block.update(v_col2block)

        elif ttype == 'many_to_many':
            for merge_feat_cls in self.merge_feat_pipeline.preM2Ms:
                merge_feat = merge_feat_cls(key)
                merge_feat.fit_transform(u,v)

            for merge_feat_cls in self.merge_feat_pipeline.M2Ms:
                merge_feat = merge_feat_cls(key)
                v_feat,v_col2type,v_col2block = merge_feat.fit_transform(u,v)
                feats.append(v_feat)
                col2type.update(v_col2type)
                col2block.update(v_col2block)
        if feats:
            feat = pd.concat(feats,axis=1)
            col2groupby = {col:key for col in feat.columns}

            del feats,v
            gc.collect()

            data = u.data
            index = data.index
            data.set_index(key,inplace=True)

            cols = list(feat.columns)
            data[cols] = feat
            data.reset_index(key,inplace=True)
            data[key] = downcast(data[key],accuracy_loss=False)
            data.index= index

            u.update_data(data,col2type,col2groupby,None,col2block,None)

    @timeclass(cls='Merger')
    def dfs(self,u_name, graph):
        depth = graph.depth
        name2table = graph.name2table
        rel_graph = graph.rel_graph

        u = name2table[u_name]
        log(f"enter {u_name}")
        for edge in rel_graph[u_name]:
            v_name = edge['to']
            if depth[v_name]['depth'] <= depth[u_name]['depth']:
                continue

            v = self.dfs(v_name, graph)
            key = edge['key']
            assert len(key) == 1
            key = key[0]
            type_ = edge['type']

            log(f"join {u_name} <--{type_}--t {v_name}")
            self.merge(key,u,v,type_,0)

            log(f"join {u_name} <--{type_}--nt {v_name}")

            del v

        log(f"leave {u_name}")
        return u

    @timeclass(cls='Merger')
    def merge_to_main_fit_transform(self,graph):
        depth = graph.depth
        name2table = graph.name2table

        u_name = CONSTANT.MAIN_TABLE_NAME
        u = name2table[u_name]
        rel_graph = graph.rel_graph

        table2feat = {}
        for edge in rel_graph[u_name]:
            v_name = edge['to']
            if depth[v_name]['depth'] <= depth[u_name]['depth']:
                continue

            v = name2table[v_name]
            key = edge['key']
            assert len(key) == 1
            key = key[0]
            type_ = edge['type']

            log(f"join {u_name} <--{type_}--t {v_name}")
            table2feat[v_name] = self.merge(key,u,v,type_,1)
            log(f"join {u_name} <--{type_}--nt {v_name}")

        self.table2feat = table2feat
        return u

    @timeclass(cls='Merger')
    def merge_table(self,graph):
        self.use_all_time_m2m = False
        if graph.M2M_relation_cnt < 3:
            self.use_all_time_m2m = True
        
        self.merge_timem2m = True
        
        graph.build_depth()

        depth = graph.depth
        u_name = CONSTANT.MAIN_TABLE_NAME
        rel_graph = graph.rel_graph

        for edge in rel_graph[u_name]:
            v_name = edge['to']
            if depth[v_name]['depth'] <= depth[u_name]['depth']:
                continue

            self.dfs(v_name,graph)
