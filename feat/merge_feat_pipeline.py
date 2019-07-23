# -*- coding: utf-8 -*-
from .default_merge_feat import *

class MergeFeatPipeline:
    def __init__(self):
        self.preM2Ms = []
        self.preO2Ms = []

        self.TimeM2Ms = []
        self.newTimeM2Ms = []

        self.O2Ms = []
        self.M2Ms = []

        self.preM2Os = []
        self.preO2Os = []

        self.O2Os = []
        self.M2Os = []


class DeafultMergeFeatPipeline(MergeFeatPipeline):
    def __init__(self):
        super(DeafultMergeFeatPipeline,self).__init__()

        self.main_init()

    def main_init(self):
        
        self.newTimeM2Ms = [TimeM2MnewLastData]
        
        self.preM2Ms = []
        self.M2Ms = [M2MKeyCount, M2MNumMean,M2MDataLast]

        self.preO2Ms = []
        self.O2Ms = [M2MKeyCount, M2MNumMean,M2MDataLast]

        self.preO2Os = []
        self.O2Os = [M2OJoin]
        
        self.preM2Os = []
        self.M2Os = [M2OJoin]
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        