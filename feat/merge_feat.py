# -*- coding: utf-8 -*-

class MergeFeat:
    def __init__(self,key):
        self.key = key
    
    def fit(self,U,V):
        pass

    def transform(self,U,V):
        pass
    
    def fit_transform(self,U,V):
        pass
    
class PreTimeM2M(MergeFeat):
    pass

class PreO2O(MergeFeat):
    pass

class PreM2O(MergeFeat):
    pass

class PreO2M(MergeFeat):
    pass

class PreM2M(MergeFeat):
    pass

class O2O(MergeFeat):
    pass

class M2O(MergeFeat):
    pass

class O2M(MergeFeat):
    pass


class M2M(MergeFeat):
    pass

class TimeM2M(MergeFeat):
    pass

class CmjTimeM2M(MergeFeat):
    def __init__(self,key,time_key,u_key_time_col):
        self.key = key
        self.time_key = time_key
        self.u_key_time_col = u_key_time_col
    
    def fit(self,T):
        pass

    def transform(self,T):
        pass
    
    def fit_transform(self,T):
        pass