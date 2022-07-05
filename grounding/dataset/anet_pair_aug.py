import logging
import numpy as np
import string
import torch
import argparse
import os

from torch.utils.data import Dataset, DataLoader

from .anet import ANetDataSentence
from .charades_pair_aug import collate_fn, Sequence_mask

class ANetVideoAugVideoPair(ANetDataSentence):
    def __init__(self, annotation_file, feature_file, params, logger):
        params['aug_mode'] = 'gt_translate'
        params['aug_percentage'] = 1
        super(ANetVideoAugVideoPair, self).__init__(annotation_file, feature_file, params, logger)
        self.if_aug = True

    def __getitem__(self, idx):
        vid = self.sen_vid[idx]
        sidx = self.sen_idx_in_video[idx]

        # video information
        video_duration = self.annotaion[vid]['duration']

        # sentence
        sentence = self.sentences[idx]
        sentence_len = self.sentence_lens[idx]
        sentence_idx = self.pad_sentence_idxes[idx]
        sentence_features = list(map(lambda x: self.word_emb_init[x], sentence_idx))
        sentence_features = np.vstack(sentence_features)
        sent_mask = Sequence_mask(self.MAX_SENTENCE_LEN, [0, sentence_len])

        # timestamp:
        timestamps = self.annotaion[vid]['timestamps'][sidx]

        #
        if self.feature_type == 'c3d':
            feature_path = os.path.join(self.feature_dir, vid + '.npy')
            video_feature = np.load(feature_path, 'r')
        elif self.feature_type == 'i3d':
            feature_path = os.path.join(self.feature_dir, vid + '.npy')
            video_feature = np.load(feature_path, 'r')
        else:
            raise NotImplementedError()
        raw_video_feature, raw_framestamps, raw_nfeats = self.vfeat_fn(video_feature, timestamps, video_duration)

        raw_video_mask = Sequence_mask(self.SAMPLE_LEN, [0, raw_nfeats])
        raw_temporal_labels = Sequence_mask(self.SAMPLE_LEN, raw_framestamps)
        raw_fore_mask = Sequence_mask(self.SAMPLE_LEN, [0, raw_framestamps[0]])
        raw_back_mask = Sequence_mask(self.SAMPLE_LEN, [raw_framestamps[1], raw_nfeats])

        aug_framestamps, aug_nfeats, aug_video_feature = self.data_aug.aug_data(raw_framestamps, raw_nfeats,
                                                                                raw_video_feature)
        aug_timestamps = aug_framestamps
        aug_video_mask = Sequence_mask(self.SAMPLE_LEN, [0, aug_nfeats])
        aug_temporal_labels = Sequence_mask(self.SAMPLE_LEN, aug_framestamps)
        aug_fore_mask = Sequence_mask(self.SAMPLE_LEN, [0, aug_framestamps[0]])
        aug_back_mask = Sequence_mask(self.SAMPLE_LEN, [aug_framestamps[1], aug_nfeats])

        # Shuffle in segments
        # _, raw_nfeats, raw_video_feature = self.data_aug.shuffel_temporal_order_by_short_segments2(raw_framestamps, raw_nfeats,
        #                                                                         raw_video_feature, seg_len=24)

        return (sentence, sentence_len, sentence_features, sent_mask,
                video_duration, vid,
                raw_video_feature, timestamps, raw_framestamps, raw_nfeats, raw_video_mask,
                raw_temporal_labels, raw_fore_mask, raw_back_mask,
                aug_video_feature, aug_timestamps, aug_framestamps, aug_nfeats, aug_video_mask,
                aug_temporal_labels, aug_fore_mask, aug_back_mask)


if __name__ == '__main__':
    feature_type='c3d'

    parser = argparse.ArgumentParser()
    if feature_type == 'c3d':
        parser.add_argument('--train_data', type=str, default='../../data/ANet/train.json',
                            help='training data path')
        parser.add_argument('--val_data', type=str, default='../../data/ANet/val_merge.json',
                            help='validation data path')
        parser.add_argument('--feature_path', type=str, default='../../data/ANet/c3d_feature',
                            help='feature path')
        parser.add_argument('--feature_type', type=str, default='c3d',
                            help='feature_type')
    else:  # i3d
        parser.add_argument('--train_data', type=str, default='../../data/ANet/train_f.json',
                            help='training data path')
        parser.add_argument('--val_data', type=str, default='../../data/ANet/val_merge_f.json',
                            help='validation data path')
        parser.add_argument('--feature_path', type=str,
                            default='../../data/ANet/i3d_feature',
                            help='feature path')
        parser.add_argument('--feature_type', type=str, default='i3d',
                            help='feature_type')

    parser.add_argument('--wordtoix_path', type=str, default='../../data/ANet/words/wordtoix.npy',
                        help='wordtoix_path')
    parser.add_argument('--ixtoword_path', type=str, default='../../data/ANet/words/ixtoword.npy',
                        help='ixtoword_path')
    parser.add_argument('--word_fts_path', type=str, default='../../data/ANet/words/word_glove_fts_init.npy',
                        help='word_fts_path')

    parser.add_argument('--video_len', type=int, default=240,
                        help='vdieo len')
    parser.add_argument('--sent_len', type=int, default=20,
                        help='sent len')

    parser.add_argument('--vfeat_fn', type=str, default='raw',
                        help='feature type')

    # Data_aug
    parser.add_argument('--if_aug', action='store_true', default=False,
                        help='data augment')
    parser.add_argument('--aug_percentage', type=float, default=1,
                        help='aug_percentage')
    parser.add_argument('--aug_mode', type=str, default=None,
                        help='checkpoint')

    params = parser.parse_args()
    params = vars(params)

    logging.basicConfig()
    logger = logging.getLogger('dataset')
    logger.setLevel(logging.INFO)


    dataset = ANetVideoAugVideoPair(params['train_data'],
                       params['feature_path'],
                       params,
                       logger,
                       )
    data_loader = DataLoader(dataset, batch_size=1,
                             shuffle=True, num_workers=0, collate_fn=collate_fn)
    count = 0
    for dt in data_loader:
        sent_list, sent_feat, sent_len, sent_mask, \
        video_duration, vid_list, \
        raw_video_feat, raw_nfeats, raw_video_mask, raw_gt, \
        aug_video_feat, aug_nfeats, aug_video_mask, aug_gt = dt

        print('vid',vid_list)
        print('sent',sent_list)
        print('duration',video_duration.detach().numpy().tolist())

        print('Raw')
        print('raw_timestps', list(map(lambda x: (round(x[0], 2), round(x[1], 2)), raw_gt['timestps'].numpy().tolist()))[0])
        print('raw_framestps', raw_gt['framestps'][0])
        print('nfeats', raw_nfeats.item())
        framestamps = torch.from_numpy(np.array(raw_gt['framestps'])).float()
        pred_time = dataset.frame2sec(framestamps, duration=video_duration, nfeats=raw_nfeats)
        print(pred_time[0].numpy().tolist())

        print('Aug')
        print('aug_timestps', list(map(lambda x: (round(x[0], 2), round(x[1], 2)), aug_gt['timestps'].numpy().tolist()))[0])
        print('aug_framestps', aug_gt['framestps'][0])
        print('nfeats', aug_nfeats.item())
        framestamps = torch.from_numpy(np.array(aug_gt['framestps'])).float()
        pred_time = dataset.frame2sec(framestamps, duration=video_duration, nfeats=aug_nfeats)
        print(pred_time[0].numpy().tolist())

        print('*' * 80)
        # if video_duration[0] < ts_time[0][0] or video_duration[0] < ts_time[0][1]:
        #     print('vid',vid_list, 'duration',video_duration.detach().numpy().tolist(), 'framestamps', ts_time.numpy().tolist())
        #     count +=1
        #break
    logger.info('test_done')
    print(count)