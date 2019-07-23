# -*- coding: utf-8 -*-

class Config:
    def __init__(self, start_time,budget):
        if budget >= 1000:
            self.keys_order2_cat_max = 50
            self.keys_order2_num_max = 50
            
            self.keys_order2_cat_maxmin = 10
            self.keys_order2_num_maxmin = 10
            self.keys_order2_num_std = 5
            
            self.keys_order2_bin_num_max = 20
            self.keys_order2_bin_cat_max = 20
            
            self.all_order2_cat_max = 7
            self.all_order2_num_max = 7

            
            self.keys_order3_num_max = 10
            self.keys_order3_cat_max = 10
            
            self.wait_feat_selection_num = 30
            self.wait_feat_selection_num_all = 20
            
            self.start_time = start_time
            self.budget = budget
        else:
            self.keys_order2_cat_max = 40
            self.keys_order2_num_max = 40
            
            self.keys_order2_cat_maxmin = 10
            self.keys_order2_num_maxmin = 10
            self.keys_order2_num_std = 5
            
            self.keys_order2_bin_num_max = 10
            self.keys_order2_bin_cat_max = 10
            
            self.all_order2_cat_max = 7
            self.all_order2_num_max = 7

            self.keys_order3_num_max = 10
            self.keys_order3_cat_max = 10
            
            self.wait_feat_selection_num = 30
            self.wait_feat_selection_num_all = 20
            
            self.start_time = start_time
            self.budget = budget

