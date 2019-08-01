# -*- coding: utf-8 -*-

from .default_feat import *
from .feat_selection import LGBFeatureSelection,LGBFeatureSelectionWait,LGBFeatureSelectionLast

class FeatPipeline:
    def __init__(self):
        self.order1s = []

class DefaultFeatPipeline(FeatPipeline):
    def __init__(self):
        super(DefaultFeatPipeline,self).__init__()
        self.main_init()

    def main_init(self):
        self.order1s = [
                PreMcToNumpy,McCatRank,
                
                OriginSession,\

                ApartCatRecognize,\

                KeysCountDIY,
                UserKeyCntDIY,SessionKeyCntDIY,\
                
                KeysTimeDiffAndFuture,
                
                UserSessionNuniqueDIY,\
                UserSessionCntDivNuniqueDIY,\
                UserKeyNuniqueDIY, SessionKeyNuniqueDIY,\
                UserKeyCntDivNuniqueDIY,SessionKeyCntDivNuniqueDIY,\

                KeysCumCntRateAndReverse,
                
                UserKeyCumCntRateAndReverse,
                
                KeyTimeDate,
                KeyTimeBin,
                KeysBinCntDIY,
                
                CatCountDIY,
                LGBFeatureSelection,\
                ]

        self.keys_order2s = [
                KeysNumMeanOrder2MinusSelfNew,
                KeysNumMaxMinOrder2MinusSelfNew,
                KeysNumStd,
                KeysCatCntOrder2New,

                LGBFeatureSelectionWait,
        ]
        
        self.all_order2s = [
                BinsCatCntOrder2DIYNew,
                BinsNumMeanOrder2DIYNew,
                CatNumMeanOrder2DIYNew,
                CatCntOrder2DIYNew,

                LGBFeatureSelectionWait
        ]

        self.post_order1s = [
                TimeNum,
        ]

        self.merge_order1s = [
                CatSegCtrOrigin,
                CatMeanEncoding,

                LGBFeatureSelectionLast,
        ]

