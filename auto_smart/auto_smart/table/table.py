# -*- coding: utf-8 -*-
from preprocessor.preprocessor import *
import CONSTANT
from util import timeclass,log
import gc

class Table:
    def __init__(self,data,table_info,session_cols,user_cols,key_cols,key_time_col,tname):
        self.name = tname
        
        self.col2type = {}
        self.col2groupby = {}
        self.col2block = {}
        self.col2istime = {}

        self.col2muldatas = {}
        self.col2muldatalens = {}
        
        self.user_cols = user_cols
        self.session_cols = []
        
        self.block2name = {}
        self.name2block = {}
        
        for col in session_cols:
            if len(self.user_cols) > 0:
                self.session_cols.append(col)
                self.col2groupby[col] = self.user_cols[0]
        
        self.key_time_col = key_time_col
        self.key_cols = key_cols
        
        self.cat_cols = None
        
        self.binary_cols = None
        self.multi_cat_cols = None
        self.num_cols = None
        
        self.time_cols = None
        
        self.bin_cols = []
        
        self.update_data(data,table_info,None)
        
        log(f'session_cols:{self.session_cols}')
        log(f'user_cols:{self.user_cols}')
        log(f'key_cols:{self.key_cols}')
        log(f'cat_cols:{self.cat_cols}')
        log(f'binary_cols:{self.binary_cols}')
        log(f'multi_cat_cols:{self.multi_cat_cols}')
        log(f'key_time_col:{self.key_time_col}')
        log(f'time_cols:{self.time_cols}')
        log(f'num_cols:{self.num_cols}')
        
        self.apart_cat_set = set()
        self.post_drop_set = set()
        
        self.col2source_cat = {}
        
        self.combine_cat_cols = []
        self.combine_num_cols = []
        self.combine_binary_cols = []
        self.wait_selection_cols = []
    
    def add_session_col(self,col):
        self.session_cols.append(col)
        self.col2type[col] = CONSTANT.CATEGORY_TYPE
        if len(self.user_cols) > 0:
            self.col2groupby[col] = self.user_cols[0]
        
    def get_groupby_cols(self,by,cols):
        new_cols = []
        bys = set()
        bys.add(by)
        while by in self.col2groupby:
            by = self.col2groupby[by]
            bys.add(by)
        
        for col in cols:
            is_skip = False
            cur = col
            while True:
                if cur in bys:
                    is_skip = True
                    break
                
                if cur in self.col2groupby:
                    cur = self.col2groupby[cur]
                else:
                    break
                
            if not is_skip:
                new_cols.append(col)
        
        return new_cols
    
    def get_not_apart_cat_cols(self,cols):
        new_cols = []
        for col in cols:
            if col not in self.apart_cat_set:
                new_cols.append(col)
        return new_cols       
    
    def drop_data(self,cols):
        drop_cols = []
        for col in cols:
            if col not in self.session_cols\
            and col not in self.user_cols\
            and col not in self.key_cols\
            and col != self.key_time_col:
                drop_cols.append(col)
        if len(drop_cols)>0:
            self.data.drop(drop_cols,axis=1,inplace=True)
            self.drop_data_cols(drop_cols)
            
    def drop_data_cols(self,drop_cols):
        for col in drop_cols:
            self.col2type.pop(col)
            if col in self.col2groupby:
                self.col2groupby.pop(col)
            
        self.type_reset()
        self.drop_combine_cols(drop_cols)

    def drop_combine_cols(self,drop_cols):
        drop_cols_set = set(drop_cols)
        
        combine_cat_cols = []
        combine_num_cols = []
        combine_binary_cols = []
        
        for col in self.combine_cat_cols:
            if col not in drop_cols_set:
                combine_cat_cols.append(col)
        
        for col in self.combine_num_cols:
            if col not in drop_cols_set:
                combine_num_cols.append(col)
        
        for col in self.combine_binary_cols:
            if col not in drop_cols_set:
                combine_binary_cols.append(col)
        
        self.combine_cat_cols = combine_cat_cols
        self.combine_num_cols = combine_num_cols
        self.combine_binary_cols = combine_binary_cols
    
    def add_apart_cat_cols(self,cols):
        self.apart_cat_set.update(cols)
    
    def add_post_drop_cols(self,cols):
        self.post_drop_set.update(cols)
    
    def add_wait_selection_cols(self,cols):
        self.wait_selection_cols.append(cols)
    
    def empty_wait_selection_cols(self):
        self.wait_selection_cols = []
    
    def update_data(self,data,col2type,col2groupby,col2source_cat=None,col2block=None,col2istime=None):
        
        self.data = data
        self.update_col2type(col2type)
        if col2groupby is not None:
            self.update_col2groupby(col2groupby)
        
        if col2block is not None:
            self.update_col2block(col2block)
        if col2istime is not None:
            self.update_col2istime(col2istime)
        
        if col2source_cat is not None:
            self.update_col2source_cat(col2source_cat)
        gc.collect()
        
    def update_col2block(self,col2block):
        self.col2block.update(col2block)
    
    def update_col2istime(self,col2istime):
        self.col2istime.update(col2istime)
    
    def update_col2groupby(self,col2groupby):
        self.col2groupby.update(col2groupby)
    
    def update_col2source_cat(self,col2source_cat):
        self.col2source_cat.update(col2source_cat)
    
    def update_col2type(self,col2type):
        self.col2type.update(col2type)
        self.type_reset()
        
    def reset_combine_cols(self,combine_cat_cols=None,combine_num_cols=None,combine_binary_cols=None):
        self.combine_cat_cols = combine_cat_cols
        self.combine_num_cols = combine_num_cols
        self.combine_binary_cols = combine_binary_cols
    
    def type_reset(self):
        
        cat_cols = []
        binary_cols = []
        multi_cat_cols = []
        num_cols = []
        time_cols = []

        for cname,ctype in self.col2type.items():
            if (ctype == CONSTANT.CATEGORY_TYPE) \
            and (cname not in self.key_cols)\
            and (cname not in self.user_cols)\
            and (cname not in self.session_cols):
                cat_cols.append(cname)
            elif ctype == CONSTANT.BINARY_TYPE:
                binary_cols.append(cname)
            elif ctype == CONSTANT.MULTI_CAT_TYPE:
                multi_cat_cols.append(cname)
            elif ctype == CONSTANT.NUMERICAL_TYPE:
                num_cols.append(cname)
            elif ctype == CONSTANT.TIME_TYPE and cname != self.key_time_col:
                time_cols.append(cname)
        
        self.cat_cols = sorted(cat_cols)
        self.binary_cols = sorted(binary_cols)
        self.num_cols = sorted(num_cols)
        self.multi_cat_cols = sorted(multi_cat_cols)
        self.time_cols = sorted(time_cols)  
    
    @timeclass(cls='Table')
    def preprocess_fit_transform(self,mscat_group2preprocessor):
        
        for col in (self.cat_cols+self.multi_cat_cols+self.user_cols+self.key_cols+self.session_cols):
            name = self.name+':'+col
            if name in self.name2block:
                block_id = self.name2block[name]
                self.data[col] = mscat_group2preprocessor[block_id].transform(self.data[col],self.col2type[col])
                
        unique_preprocessor = UniquePreprocessor()
        unique_preprocessor.fit_transform(self)
        
        all_diff_preprocessor = AllDiffPreprocessor()
        all_diff_preprocessor.fit_transform(self)
        
        binary_preprocessor = BinaryPreprocessor()
        binary_preprocessor.fit_transform(self)
        
        num_preprocess = NumPreprocessor()
        num_preprocess.fit_transform(self)

        general_preprocessor = GeneralPreprocessor()
        general_preprocessor.fit_transform(self)
