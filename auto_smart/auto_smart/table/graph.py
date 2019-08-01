# -*- coding: utf-8 -*-

from .table import Table
from preprocessor.preprocessor import MSCatPreprocessor
import pandas as pd
import CONSTANT
from util import timeclass, log
from collections import defaultdict, deque
import gc
from joblib import Parallel, delayed

class Graph:
    def __init__(self,info,tables):
        
        self.info = info
        
        self.table2info = info['tables']
        self.relations = info['relations']
        self.key_time_col = info['time_col']
        
        self.M2M_relation_cnt = 0
        for relation in info['relations']:
            if relation['type'] == "many_to_many":
                self.M2M_relation_cnt = self.M2M_relation_cnt + 1
        
        self.key_col_set = None
        self.user_col = None
        
        self.name2table = {}
        self.tables = []
        
        key_col_set = set()
        for relation in info['relations']:
            key_col_set.update(relation['key'])
        self.key_col_set = key_col_set
        
        user_col = None
        for tname,table in tables.items():
            key_cols = []
            if tname == CONSTANT.MAIN_TABLE_NAME:
                for col in self.table2info[tname]:
                    if col in self.key_col_set:
                        key_cols.append(col)
                
                user_col = self.recognize_user_col(tables[tname],key_cols)
                
        self.user_col = user_col      
        del user_col
        
        main_cat_cols = []
        session_col = None
        for tname,table in tables.items():
            if tname == CONSTANT.MAIN_TABLE_NAME:
                for col in self.table2info[tname]:
                    type_ = self.table2info[tname][col]
                    if type_ == CONSTANT.CATEGORY_TYPE and col!=self.user_col and col not in key_col_set:
                        main_cat_cols.append(col)
                    
                session_cols = self.recognize_session_col(tables[tname],main_cat_cols,self.user_col)
        
        
        self.main_session_cols = session_cols
        del main_cat_cols
        del session_col
        
        for tname,table in tables.items():
            key_cols = []
            key_time_col = None
            user_cols = []

            for col in self.table2info[tname]:
                
                if col in self.key_col_set and col != self.user_col:
                    key_cols.append(col)
                    
                if col == self.user_col:
                    user_cols.append(col)
                
                if col == self.key_time_col:
                    key_time_col = col
            
            cat_cols = []
            for col in self.table2info[tname]:
                type_ = self.table2info[tname][col]
                if type_ == CONSTANT.CATEGORY_TYPE:
                    cat_cols.append(col)
            
            binary_cols = self.recognize_binary_col(tables[tname],cat_cols)
            for col in binary_cols:
                self.table2info[tname][col] = CONSTANT.BINARY_TYPE
                
            self.tables.append(tname)
            if tname == CONSTANT.MAIN_TABLE_NAME:
                self.name2table[tname] = Table(tables[tname],self.table2info[tname],self.main_session_cols,user_cols,key_cols,key_time_col,tname)
                
            else:
                self.name2table[tname] = Table(tables[tname],self.table2info[tname],[],user_cols,key_cols,key_time_col,tname)
                
            if tname == CONSTANT.MAIN_TABLE_NAME:
                self.main_key_cols = key_cols
                self.main_key_time_col = key_time_col
                self.main_user_col = user_cols
                self.main_table_info = self.table2info[tname]
                
        block2name,name2block = self.init_graph_to_blocks() 
        self.block2name = block2name
        self.name2block = name2block
        
        for tname in self.name2table:
            self.name2table[tname].block2name = block2name
            self.name2table[tname].name2block = name2block
        
        for tname in self.name2table:
            col2block = {}
            for col in self.name2table[tname].data.columns:
                name = tname + ':' + col
                
                if name in self.name2block:
                    block_id = self.name2block[name]
                    col2block[col] = block_id
            
            self.name2table[tname].col2block = col2block
        
        for tname in self.name2table:
            col2table = {}
            for col in self.name2table[tname].data.columns:
                col2table[col] = tname
                
            self.name2table[tname].col2table = col2table
            
    @timeclass(cls='Graph')
    def init_graph_to_blocks(self):
        mode = 'all'
        if mode == 'all':
            t_datas = []
            t_names = []

            for t_name in self.name2table:
                t_table = self.name2table[t_name]
                t_data = t_table.data
                t_data_num = t_data.shape[0]
                t_limit_num = 100000
                if t_limit_num > t_data_num:
                    t_limit_num = t_data_num
                t_sample_frac = t_limit_num / t_data_num
                t_data = t_data.sample(frac=t_sample_frac,random_state=CONSTANT.SEED)
                
                t_datas.append(t_data)
                t_names.append(t_name)
                    
            all_cat_cols = []
            all_cat2type = {}
            for t_data,t_name in zip(t_datas,t_names):
               
                for col in t_data.columns:
                    col2type = self.table2info[ t_name ][ col ]
                    new_col = t_name+':'+col
                    if col2type == CONSTANT.MULTI_CAT_TYPE or col2type == CONSTANT.CATEGORY_TYPE:
                        all_cat_cols.append(new_col)
                        all_cat2type[new_col] = col2type
            
            mc_graph = {}
            all_cat_len = len(all_cat_cols)
            for i in range(all_cat_len):
                name1 = all_cat_cols[i]
                mc_graph[name1] = {}
                for j in range(all_cat_len):
                    name2 = all_cat_cols[j]
                    mc_graph[name1][name2] = 0
            
            for t1 in range(len(t_datas)):
                t_data_1 = t_datas[t1]
                t_name_1 = t_names[t1]
                for col1 in t_data_1.columns:
                    if col1 in self.key_col_set:
                        name1 = t_name_1+':'+col1
                            
                        for t2 in range(len(t_datas)):
                            t_data_2 = t_datas[t2]
                            t_name_2 = t_names[t2]
                            for col2 in t_data_2.columns:
                                if col2 == col1: 
                                    name2 = t_name_2+':'+col2
                                    mc_graph[name1][name2] = 1
                                    mc_graph[name2][name1] = 1
            
            log('init mcgraph')      
            
            all_cat2set = {}

            for t_data,t_name in zip(t_datas,t_names):
                for col in t_data.columns:
                    new_col = t_name+':'+col
                    if new_col in all_cat2type:
                        cur_set = set()
                        if all_cat2type[new_col] == CONSTANT.MULTI_CAT_TYPE:
                            
                            for val in t_data[col]:
                                if type(val) == float:
                                    continue
                                cur_set.update(val.split(CONSTANT.MULTI_CAT_DELIMITER))
                            
                        elif all_cat2type[new_col] == CONSTANT.CATEGORY_TYPE:
                            cur_set = set(t_data[col].dropna())
                        
                        all_cat2set[new_col] = cur_set
            
            all_cat_len = len(all_cat_cols)
            for i in range(all_cat_len):
                for j in range(i+1,all_cat_len):
                    name1 = all_cat_cols[i]  
                    name2 = all_cat_cols[j]
                    
                    len1 = len(all_cat2set[name1])
                    len2 = len(all_cat2set[name2])

                    less_len = min(len1,len2)
                    if less_len <= 1:
                        continue
                    
                    if mc_graph[name1][name2]==1 or mc_graph[name2][name1] == 1:
                        continue
                    
                    if len(all_cat2set[name1] & all_cat2set[name2])/less_len > 0.1:
                        mc_graph[name1][name2] = 1
                        mc_graph[name2][name1] = 1
            
            block2name = {}

            block_id = 0
            vis = {}
            nodes = list(mc_graph.keys())
            def dfs(now,block_id):
                block2name[block_id].append(now)
                for nex in nodes:
                    if mc_graph[now][nex] and ( not (nex in vis) ):
                        vis[nex] = 1
                        dfs(nex,block_id)
            
            for now in nodes:
                if now in vis:
                    continue
                vis[now] = 1
                block_id += 1
                block2name[block_id] = []
                dfs(now,block_id)
            
            name2block = {}

            for block in block2name:
                for col in block2name[block]:
                    name2block[col] = block
            log(f'blocks: {block2name}')
            return block2name,name2block

        elif mode == 'part':
            pass

    @timeclass(cls='Graph')
    def sort_tables(self):
        for tname in self.name2table:
            table = self.name2table[tname]
            if table.key_time_col is not None:
                table.data.sort_values(by=table.key_time_col,inplace=True)
    
    @timeclass(cls='Graph')
    def sort_main_table(self):
        table = self.name2table[CONSTANT.MAIN_TABLE_NAME]
        if table.key_time_col is not None:
            table.data.sort_values(by=table.key_time_col,inplace=True)
    
    @timeclass(cls='Graph')
    def recognize_session_col(self,data,cat_cols,user_col):
        if user_col is None:
            return []
        
        user_nunique = data[user_col].nunique()
        session_cols = []
        
        def func(df,user_nunique):
            cat_col = df.columns[0]
            user_col = df.columns[1]
            cat_nunique = df[cat_col].nunique()
            
            if (cat_nunique <= user_nunique) or (cat_nunique >= df.shape[0]-10):
                return False
            
            if (df.groupby(cat_col)[user_col].nunique()>1).sum()>10:
                return False
            
            return True
        
        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(data[[col,user_col]],user_nunique) for col in cat_cols)
        
        for col,is_session in zip(cat_cols,res):
            if is_session:
                session_cols.append(col)

        return session_cols
    
    @timeclass(cls='Graph')
    def recognize_binary_col(self,data,cat_cols):
        def func(ss):
            ss = ss.unique()
            if len(ss) == 3:
                if pd.isna(ss).sum() == 1:
                    return True
            if len(ss) == 2:
                return True
            return False
        
        binary_cols = []
        
        res = Parallel(n_jobs=CONSTANT.JOBS,require='sharedmem')(delayed(func)(data[col]) for col in cat_cols)
        
        for col,is_binary in zip(cat_cols,res):
            if is_binary:
                binary_cols.append(col)
        
        return binary_cols
    
    @timeclass(cls='Graph')
    def recognize_user_col(self,data,key_cols):
        user_col = None
        nunique = -1
        for col in key_cols:
            nnum = data[col].nunique()
            if nnum > nunique:
                user_col = col
                nunique = nnum
        return user_col
    
    @timeclass(cls='Graph')
    def preprocess_fit_transform(self):
        log('start mscat')

        mscat_block2preprocessor = {}
        for block_id in range(1,len(self.block2name)+1):
            mscat_block2preprocessor[block_id] = MSCatPreprocessor()
        ss = {}
        for block_id in range(1,len(self.block2name)+1):
            ss[block_id] = pd.Series()
            
        t_datas = []
        t_names = []
        for t_name in self.name2table:
            t_table = self.name2table[t_name]
            t_data = t_table.data
            
            t_datas.append(t_data)
            t_names.append(t_name)
        
        for t in range(len(t_datas)):
                t_data = t_datas[t]
                t_name = t_names[t]
                for col in t_data.columns:
                    coltype = self.table2info[ t_name ][col] 
                    if coltype == CONSTANT.MULTI_CAT_TYPE or coltype == CONSTANT.CATEGORY_TYPE:
                        name = t_name + ':' + col
                        if name in self.name2block:
                            block_id = self.name2block[name]
                            ss[block_id] = pd.concat([ss[block_id],t_data[col].drop_duplicates()])

        for block_id in range(1,len(self.block2name)+1):
            mscat_block2preprocessor[block_id].fit(ss[block_id])
           
        for tname,table in self.name2table.items():
            table.preprocess_fit_transform(mscat_block2preprocessor)
            
        gc.collect()
            
    def set_main_table(self,table):
        tname = CONSTANT.MAIN_TABLE_NAME
        self.name2table[CONSTANT.MAIN_TABLE_NAME] = Table(table,self.main_table_info,self.main_session_cols,self.main_user_col,self.main_key_cols,self.main_key_time_col,tname)
        gc.collect()
        
    @timeclass(cls='Graph')
    def bfs(self,root_name, graph, depth):
        depth[CONSTANT.MAIN_TABLE_NAME]['depth'] = 0
        queue = deque([root_name])
        while queue:
            u_name = queue.popleft()
            for edge in graph[u_name]:
                v_name = edge['to']
                if 'depth' not in depth[v_name]:
                    depth[v_name]['depth'] = depth[u_name]['depth'] + 1
                    queue.append(v_name)
    
    @timeclass(cls='Graph')
    def build_depth(self):
        rel_graph = defaultdict(list)
        depth = {}
        
        for tname in self.tables:
            depth[tname] = {}
            
        for rel in self.relations:
            ta = rel['table_A']
            tb = rel['table_B']
            rel_graph[ta].append({
                "to": tb,
                "key": rel['key'],
                "type": rel['type']
            })
            rel_graph[tb].append({
                "to": ta,
                "key": rel['key'],
                "type": '_'.join(rel['type'].split('_')[::-1])
            })
        self.bfs(CONSTANT.MAIN_TABLE_NAME, rel_graph, depth)
        
        self.rel_graph = rel_graph
        self.depth = depth
    
    
    