import random
import os
import numpy as np
import torch.nn as nn

def set_device(logger, id=-1):
    logger.info('*' * 100)

    if id == -1:
        tmp_file_name = 'tmp%s'%(random.random())
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >%s'%(tmp_file_name))
        memory_gpu=[int(x.split()[2]) for x in open(tmp_file_name,'r').readlines()]
        id = np.argmax(memory_gpu)
        os.system('rm %s'%(tmp_file_name))

    logger.info('process runs on gpu %d', id)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(id)
    logger.info('*'*100)
    return id

def update_values(dict_from, dict_to):
    for key, value in dict_from.items():
        if isinstance(value, dict):
            update_values(dict_from[key], dict_to[key])
        elif value is not None:
            dict_to[key] = dict_from[key]


def LoggerInfo(logger, title, data):
    logger.info('*' * 100)
    logger.info(title)
    logger.info(data)

def StatisticsPrint(statistics, title):
    print(title,":")
    print('\t'.join(str(k) for k in statistics[title].keys()))
    print('\t'.join(str(v) for v in statistics[title].values()))
    if title in ['mIoU']:
        key = list(statistics[title].keys())
        val = list(statistics[title].values())
        print('Max mIoU:', max(val), '\tEpoch', key[val.index(max(val))])

def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.normalization.LayerNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif len(list(m.children())) == 0:
            group_decay.extend([*m.parameters()])

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups