import glob
import os
import re

cur_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(cur_dir, 'data')

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

def get_min_loss_model_path_prefix(checkpoint_dir):
    models = glob.glob(os.path.join(checkpoint_dir,'*.meta'))
    model_path = ''

    if models:
        sorted_models = sorted(models, key=lambda path:re.findall("model-(\d+\.\d+)-(0\.\d+)-(\d+).meta",path)[0])
        model_path = sorted_models[-1].strip('.meta')
    return model_path   # 取出最新的神经网络结构文件*.meta

def get_checkpoint_prefix(use_crf, char_base):
    if use_crf:
        attach = '-crf'
    else:
        attach = ''
    if char_base:
        attach += '-char'
    else:
        attach += '-word'

    return os.path.join(data_dir,'checkpoint'+ attach)



