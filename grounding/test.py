import argparse
import numpy
import logging
import sys
import os
import time
import yaml

import torch
from torch.utils.data import DataLoader

from util.model_saver import ModelSaver
from util.helper_function import set_device, StatisticsPrint, LoggerInfo, update_values, group_weight
from loss import *
from IoU_eval import retrieval_eval
from train import perpare_data
from model.SpanGroundMatchDisc import GMD


def constract_model(params, logger):
    if params['start_from'] is not None:
        state_dict = torch.load(params['start_from'],
                                map_location=lambda storage, location: storage)
        logger.warn('use checkpoint: %s', params['start_from'])

    video_seq_set = {}
    video_seq_set['name'] = params['video_encoder']
    video_seq_set['input_dim'] = params['video_feature_dim']
    video_seq_set['rnn_hidden_dim'] = params['video_rnn_hiddendim']
    video_seq_set['rnn_layers'] = params['video_rnn_layers']
    video_seq_set['rnn_cell'] = params['video_rnn_cell']
    video_seq_set['mask'] = params['mask']
    video_seq_set['drop_out'] = params['dropout']
    video_seq_set['T'] = params['video_len']
    video_seq_set['nblocks'] = 2

    sent_seq_set = {}
    sent_seq_set['name'] = params['sent_encoder']
    sent_seq_set['input_dim'] = 300
    sent_seq_set['rnn_hidden_dim'] = params['sent_rnn_hiddendim']
    sent_seq_set['rnn_layers'] = params['sent_rnn_layers']
    sent_seq_set['rnn_cell'] = params['sent_rnn_cell']
    sent_seq_set['drop_out'] = params['dropout']

    # grounding
    grounding_set = {}
    grounding_set['cross_name'] = params['crossmodal']
    grounding_set['name'] = params['predictor']
    grounding_set['lstm_hidden_dim'] = params['span_hidden_dim']
    grounding_set['mlp_hidden_dim'] = params['mlp_hidden_dim']

    # matching
    matching_set= {}
    cross_set, temporal_set, predict_set = {}, {}, {}
    cross_set['name'] = params['m_cross']
    temporal_set['name'] = params['m_temp']
    temporal_set['hidden_dim'] = 256
    temporal_set['layers'] = 2
    temporal_set['dropout'] = params['dropout']

    predict_set['name'] = params['m_pred']
    predict_set['activation'] = params['m_pred_activ']
    predict_set['hidden_dim'] = params['m_pred_hidden']

    matching_set['cross'] = cross_set
    matching_set['temporal'] = temporal_set
    matching_set['predict'] = predict_set

    model = GMD(video_seq_set, sent_seq_set, grounding_set, matching_set, logger, params['dropout'])

    logger.info('*' * 120)
    sys.stdout.flush()
    print('Model' + '*' * 110)
    print(model)

    if params['start_from'] is not None:
        model.load_state_dict(state_dict)
        print("load over.", params['start_from'])

    return model

def test(model, data_loader, params, logger, step, saver, dataset):
    model.eval()

    _start_time = time.time()
    accumulate_loss = 0
    accumulate_iou = 0
    pred_dict = {'version': 'V0',
                 'results': {},
                 'external_data': {
                     'used': True,
                     'details': 'provided i3D feature'
                 },
                 'params': params}

    logger.info('testing:' + '*' * 106)

    for idx, batch_data in enumerate(data_loader):
        batch_time = time.time()

        sent_list, sent_feat, sent_len, sent_mask, \
        video_duration, vid_list, \
        video_feat, nfeats, video_mask, gt, \
        _,_,_,_ = perpare_data(batch_data)

        B, T, _ = video_feat.size()

        with torch.no_grad():

            span_prob = model.module.eval_forward(video_feat, sent_feat, video_mask, sent_mask)

            loss_g = span_ground_loss(span_prob['start'], span_prob['end'], gt['framestps'])

            loss = loss_g

            pred_time, score = span_pred(span_prob['start'].cpu(), span_prob['end'].cpu())
            pred_time = dataset.frame2sec(pred_time.float(), duration=video_duration, nfeats=nfeats)
            miou = compute_mean_iou(pred_time.float().data, gt['timestps'].data)

        accumulate_loss += loss.cpu().item()
        accumulate_iou += miou.cpu().item()

        if params['batch_log_interval'] != -1 and idx % params['batch_log_interval'] == 0:
            logger.info('test: epoch[%03d], batch[%04d/%04d], elapsed time=%0.2fs, loss: %03.3f, miou: %03.3f',
                        step, idx, len(data_loader), time.time() - batch_time, loss.cpu().item(), miou)

        # # submits
        pred_time = pred_time.cpu().data.numpy()
        score = score.cpu().data.numpy()
        ts_time = gt['timestps'].cpu().data.numpy()
        video_duration = video_duration.cpu().data.numpy()
        for idx in range(B):
            video_key = vid_list[idx]
            if video_key not in pred_dict['results']:
                pred_dict['results'][video_key] = list()
            pred_dict['results'][video_key].append({
                'sentence': sent_list[idx],
                'timestamp': pred_time[idx].tolist(),
                'gt_timestamp': ts_time[idx].tolist(),
                'score': score[idx].tolist(),
                'video_duration': video_duration[idx].tolist(),
            })

    submit_filename = saver.save_submits(pred_dict, step, 'test_data')
    logger.info('epoch [%03d]: elapsed time:%0.4fs, avg loss: %03.3f, miou: %03.3f',
                step, time.time() - _start_time,
                accumulate_loss / len(data_loader), accumulate_iou / len(data_loader))
    logger.info('*' * 100)

    return accumulate_iou / len(data_loader), submit_filename


def select_dataset_and_cfn(dataset_name):
    if dataset_name in ['charades', 'charades_cd']:
        from dataset.charades_pair_aug import collate_fn, CharadesVideoAugVideoPair
        data_class = CharadesVideoAugVideoPair
        cfn = collate_fn
    elif dataset_name in ['anet', 'anet_cd']:
        from dataset.anet_pair_aug import collate_fn, ANetVideoAugVideoPair
        data_class = ANetVideoAugVideoPair
        cfn = collate_fn
    else:
        assert False, 'Error datasetname' + dataset_name
    return data_class, cfn

def main(params):
    logging.basicConfig()
    logger = logging.getLogger(params['alias'])
    gpu_id = set_device(logger, params['gpu_id'])
    logger = logging.getLogger(params['alias'] + '(%d)' % gpu_id)
    set_device(logger, params['gpu_id'])
    logger.setLevel(logging.INFO)

    saver = ModelSaver(params, os.path.abspath('./third_party/densevid_eval'))
    model = constract_model(params, logger)
    model = torch.nn.DataParallel(model).cuda()

    data_class, source_cfn = select_dataset_and_cfn(params['test'])
    test_set = data_class(
        params['test_data'],
        params['test_featpath'],
        params,
        logger
    )
    test_loader = DataLoader(test_set, batch_size=params['batch_size'][0],
                              shuffle=False, num_workers=4, collate_fn=source_cfn)

    iou, submit_filename = test(model, test_loader, params, logger, 0, saver, test_set)

    retrieval_eval(submit_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False,
                        help='option to debug')

    # Datasets
    parser.add_argument('--feature_type', type=str, default='i3d',
                        help='feature type')
    parser.add_argument('--vfeat_fn', type=str, default='raw',
                        help='feature type')
    parser.add_argument('--cfg', type=str, default='charades_cd_i3d.yml',
                        help='domain adaptation configure')

    parser.add_argument('--train', type=str, default='charades',
                        help='source dataset')
    parser.add_argument('--valid', type=str, default='charades',
                        help='source dataset')

    parser.add_argument('--train_data', type=str, default='../../data/Charades/train.json',
                        help='source data path')
    parser.add_argument('--val_data', type=str, default='../../data/Charades/test.json',
                        help='validation data path')

    parser.add_argument('--train_featpath', type=str, default='../../data/Charades/i3d_feature',
                        help='feature path')
    parser.add_argument('--valid_featpath', type=str, default='../../data/Charades/i3d_feature',
                        help='feature path')

    parser.add_argument('--wordtoix_path', type=str, default='words/wordtoix.npy',
                        help='wordtoix_path')
    parser.add_argument('--ixtoword_path', type=str, default='words/ixtoword.npy',
                        help='ixtoword_path')
    parser.add_argument('--word_fts_path', type=str, default='words/word_glove_fts_init.npy',
                        help='word_fts_path')

    # Data_aug
    parser.add_argument('--if_aug', action='store_true', default=False,
                        help='data augment')
    parser.add_argument('--aug_percentage', type=float, default=1.0,
                        help='aug_percentage')
    parser.add_argument('--aug_mode', type=str, default='gt_translate',
                        help='checkpoint')

    # Load and Save
    parser.add_argument('--start_from', type=str, default=None,
                        help='checkpoint')

    # Interval
    parser.add_argument('--save_model_interval', type=int, default=1,
                        help='save the model parameters every a certain step')
    parser.add_argument('--batch_log_interval', type=int, default=50,
                        help='log interval')
    parser.add_argument('--batch_log_interval_test', type=int, default=50,
                        help='log interval')
    parser.add_argument('--test_interval', type=int, default=1,
                        help='test interval between training')

    # Training Setting
    parser.add_argument('-b', '--batch_size', default=[32, 28, 64], type=int, nargs="+",
                        metavar='N', help='mini-batch size ([train, valid, test])')
    parser.add_argument('--epoch', type=int, default=30,
                        help='training epochs in total')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='used in data loader(only 1 is supported because of bugs in h5py)')
    parser.add_argument('--alias', type=str, default='test',
                        help='alias used in model/checkpoint saver')
    parser.add_argument('--runs', type=str, default='runs',
                        help='folder where models are saved')
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help='the id of gup used to train the model, -1 means automatically choose the best one')

    # Loss
    parser.add_argument('--loss_disc_lambda', type=float, default=1.0,
                        help='weight of loss_d in final loss')
    parser.add_argument('--loss_m1_lambda', type=float, default=1,
                        help='loss_da in final loss')
    parser.add_argument('--loss_m2_lambda', type=float, default=1,
                        help='loss_da in final loss')

    # Optim and Lr
    parser.add_argument('--optim', type=str, default='adam',
                        help='optimizer')
    parser.add_argument('--lr_schd', type=str, default='ms',
                        help='optimizer')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate used to train the model')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay learning rate by this value every decay step')
    parser.add_argument('--lr_step', type=int, nargs='+', default=[20, 40],
                        help='lr_steps used to decay the learning_rate')
    parser.add_argument('--momentum', type=float, default=0.8,
                        help='momentum used in the process of learning')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay, i.e. weight normalization')
    parser.add_argument('--grad_clip', action='store_true', default=False,
                        help='gradient clip threshold(not used)')
    parser.add_argument('--grad_clip_max', type=float, default=1.0,
                        help='gradient clip threshold(not used)')
    parser.add_argument('--group_weight', action='store_true', default=False,
                        help='group_weight')

    # Model
    parser.add_argument('--model', type=str, default="GMD",
                        help='the model to be used')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='rnn_dropout')

    # Language
    parser.add_argument('--sent_encoder', type=str, default='rnn',
                        help='sent encoder')
    parser.add_argument('--sent_embedding_dim', type=int, default=300)
    parser.add_argument('--sent_rnn_hiddendim', type=int, default=256,
                        help='hidden dimension of rnn')
    parser.add_argument('--sent_rnn_layers', type=int, default=2,
                        help='layers number of rnn')
    parser.add_argument('--sent_rnn_cell', type=str, default='lstm',
                        help='rnn cell used in the model')
    parser.add_argument('--sent_len', type=int, default=20,
                        help='layers number of rnn')

    # Video
    parser.add_argument('--video_encoder', type=str, default='query_aware_encoder',
                        help='video encoder')
    parser.add_argument('--video_len', type=int, default=128,
                        help='vdieo len')
    parser.add_argument('--video_feature_dim', type=int, default=1024)
    parser.add_argument('--video_rnn_hiddendim', type=int, default=256,
                        help='hidden dimension of rnn')
    parser.add_argument('--video_rnn_layers', type=int, default=2,
                        help='layers number of rnn')
    parser.add_argument('--video_rnn_cell', type=str, default='lstm',
                        help='rnn cell used in the model')
    parser.add_argument('--mask', action='store_true', default=False,
                        help='seq mask')

    # Cross-Modal Interaction
    parser.add_argument('--crossmodal', type=str, default='vs',
                        help='video-sent fusion manner')

    # Span Predictor
    parser.add_argument('--predictor', type=str, default="mlp",
                        help='the predictor to be used')
    parser.add_argument('--mlp_hidden_dim', type=int, default=256,
                        help='hidden dimension of mlp')
    parser.add_argument('--span_hidden_dim', type=int, default=128,
                        help='hidden dimension of rnn')

    # Matching setting
    parser.add_argument('--m_cross', type=str, default="concat",
                        help='')
    parser.add_argument('--m_temp', type=str, default="none",
                        help='')
    parser.add_argument('--m_pred', type=str, default="mlp",
                        help='')
    parser.add_argument('--m_pred_activ', type=str, default="relu",
                        help='')
    parser.add_argument('--m_pred_hidden', type=int, default=1024,
                        help='')


    params = parser.parse_args()
    params = vars(params)

    cfgs_file = params['cfg']
    cfgs_file = os.path.join('cfgs',cfgs_file)
    with open(cfgs_file, 'r') as handle:
        options_yaml = yaml.load(handle, Loader=yaml.FullLoader)
    update_values(options_yaml, params)
    # print(params)


    main(params)
    print('Testing finished successfully!')