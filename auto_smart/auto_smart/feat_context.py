# -*- coding: utf-8 -*-
import CONSTANT

class FeatContext:
    @staticmethod
    def gen_feat_name(namespace,cls_name,feat_name,feat_type):
        prefix = CONSTANT.type2prefix[feat_type]


        return f"{prefix}{cls_name}:{feat_name}:{namespace}"

    @staticmethod
    def gen_merge_name(table_name,feat_name,feat_type):
        prefix = CONSTANT.type2prefix[feat_type]
        return f"{prefix}{table_name}.({feat_name})"

    @staticmethod
    def gen_merge_feat_name(namespace,cls_name,feat_name,feat_type,table_name):
        feat_name = FeatContext.gen_feat_name(namespace,cls_name,feat_name,feat_type)
        return FeatContext.gen_merge_name(table_name,feat_name,feat_type)
