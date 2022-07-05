import json
from torch.utils.data import Dataset, DataLoader
import h5py
import logging
import numpy as np
import string
import torch
import argparse
import os
from .charades import CharadesDataSentence, Sequence_mask

def collate_fn(batch):
    batch_size = len(batch)
    sent_list, sent_len, sent_feat, sent_mask, \
        video_duration, vid_list, \
        raw_video_feat, timestamps, raw_framestamps, raw_nfeats, raw_video_mask, \
        raw_temporal_labels, raw_fore_mask, raw_back_mask, \
        aug_video_feat, aug_timestamps, aug_framestamps, aug_nfeats, aug_video_mask, \
        aug_temporal_labels, aug_fore_mask, aug_back_mask = zip(*batch)

    raw_video_tensor = torch.from_numpy(np.vstack(raw_video_feat)).float()
    aug_video_tensor = torch.from_numpy(np.vstack(aug_video_feat)).float()
    raw_video_mask_tensor = torch.from_numpy(np.stack(raw_video_mask, axis=0))
    aug_video_mask_tensor = torch.from_numpy(np.stack(aug_video_mask, axis=0))
    raw_nfeats = torch.from_numpy(np.array(raw_nfeats))
    aug_nfeats = torch.from_numpy(np.array(aug_nfeats))

    sent_len_tensor = torch.from_numpy(np.array(sent_len))
    sent_feat_tensor = torch.from_numpy(np.stack(sent_feat, axis=0)).float()
    sent_mask_tensor = torch.from_numpy(np.stack(sent_mask, axis=0))

    video_duration_tensor = torch.from_numpy(np.array(video_duration))

    # Gt
    timestamps_tensor = torch.from_numpy(np.array(timestamps)).float()
    aug_timestamps_tensor = torch.from_numpy(np.array(aug_timestamps)).float()
    raw_temporal_labels = torch.from_numpy(np.stack(raw_temporal_labels, axis=0))
    aug_temporal_labels = torch.from_numpy(np.stack(aug_temporal_labels, axis=0))
    raw_fore_mask = torch.from_numpy(np.stack(raw_fore_mask, axis=0))
    aug_fore_mask = torch.from_numpy(np.stack(aug_fore_mask, axis=0))
    raw_back_mask = torch.from_numpy(np.stack(raw_back_mask, axis=0))
    aug_back_mask = torch.from_numpy(np.stack(aug_back_mask, axis=0))
    raw_gt, aug_gt = {}, {}
    raw_gt['timestps'] = timestamps_tensor
    raw_gt['framestps'] = raw_framestamps
    raw_gt['temporal_labels'] = raw_temporal_labels
    raw_gt['fore_masks'] = raw_fore_mask
    raw_gt['back_masks'] = raw_back_mask
    aug_gt['timestps'] = aug_timestamps_tensor
    aug_gt['framestps'] = aug_framestamps
    aug_gt['temporal_labels'] = aug_temporal_labels
    aug_gt['fore_masks'] = aug_fore_mask
    aug_gt['back_masks'] = aug_back_mask

    return sent_list, sent_feat_tensor, sent_len_tensor, sent_mask_tensor, \
           video_duration_tensor, vid_list, \
           raw_video_tensor, raw_nfeats, raw_video_mask_tensor, raw_gt, \
           aug_video_tensor, aug_nfeats, aug_video_mask_tensor, aug_gt

class CharadesVideoAugVideoPair(CharadesDataSentence):
    def __init__(self, annotation_file, feature_file, params, logger):
        params['aug_mode'] = 'gt_translate'
        params['aug_percentage'] = 1
        super(CharadesVideoAugVideoPair, self).__init__(annotation_file, feature_file, params, logger)
        self.if_aug = True

    def __getitem__(self, idx):
        vid = self.sen_vid[idx]
        sidx = self.sen_idx_in_video[idx]

        # video information
        video_duration = self.annotaion[vid]['video_duration']
        raw_total_frame = video_duration * self.annotaion[vid]['decode_fps']

        # sentence
        sentence = self.sentences[idx]
        sentence_len = self.sentence_lens[idx]
        sentence_idx = self.pad_sentence_idxes[idx]
        sentence_features = list(map(lambda x: self.word_emb_init[x], sentence_idx))
        sentence_features = np.vstack(sentence_features)
        sent_mask = Sequence_mask(self.MAX_SENTENCE_LEN, [0, sentence_len])

        # timestamp:
        timestamps = self.annotaion[vid]['timestamps'][sidx]

        if self.feature_type == 'i3d':
            feature_path = os.path.join(self.feature_file, vid + '.npy')
            video_feature = np.load(feature_path, 'r')
        elif self.feature_type.lower() in ['lgi3d']:
            feature_path = os.path.join(self.feature_file, vid + '.npy')
            video_feature = np.load(feature_path, 'r')
            video_feature = np.resize(video_feature, (-1, 1024))

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

        # _, raw_nfeats, raw_video_feature = self.data_aug.shuffel_temporal_order_by_short_segments2(raw_framestamps,
        #                                                                                            raw_nfeats,
        #                                                                                            raw_video_feature,
        #                                                                                            seg_len=8)

        return (sentence, sentence_len, sentence_features, sent_mask,
                video_duration, vid,
                raw_video_feature, timestamps, raw_framestamps, raw_nfeats, raw_video_mask,
                raw_temporal_labels, raw_fore_mask, raw_back_mask,
                aug_video_feature, aug_timestamps, aug_framestamps, aug_nfeats, aug_video_mask,
                aug_temporal_labels, aug_fore_mask, aug_back_mask)


if __name__ == '__main__':
    feature_type='i3d'

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str, default='../../data/Charades/train.json',
                        help='training data path')
    parser.add_argument('--val_data', type=str, default='../../data/Charades/test.json',
                        help='validation data path')
    if feature_type =='i3d':
        parser.add_argument('--feature_path', type=str, default='../../data/Charades/charades_i3d_rgb.hdf5',
                            help='feature path')
        parser.add_argument('--feature_type', type=str, default='i3d',
                            help='feature type')
    elif feature_type == 'LG_i3d_finetuned':
        parser.add_argument('--feature_path', type=str, default='../../data/Charades/LG_i3d_finetuned',
                            help='feature path')
        parser.add_argument('--feature_type', type=str, default='LG_i3d_finetuned',
                            help='feature type')
    else:
        parser.add_argument('--feature_path', type=str, default='../../data/Charades/charades_features_finetune',
                            help='feature path')
        parser.add_argument('--feature_type', type=str, default='i3d_finetune',
                            help='feature type')


    parser.add_argument('--wordtoix_path', type=str, default='../../data/Charades/words/wordtoix.npy',
                        help='wordtoix_path')
    parser.add_argument('--ixtoword_path', type=str, default='../../data/Charades/words/ixtoword.npy',
                        help='ixtoword_path')
    parser.add_argument('--word_fts_path', type=str, default='../../data/Charades/words/word_glove_fts_init.npy',
                        help='word_fts_path')

    parser.add_argument('--video_len', type=int, default=64,
                        help='vdieo len')
    parser.add_argument('--sent_len', type=int, default=15,
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


    dataset = CharadesVideoAugVideoPair(params['train_data'],
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
        print('temp_labels', raw_gt['temporal_labels'][0][raw_gt['framestps'][0][0]:raw_gt['framestps'][0][1]+1].numpy().tolist())
        # if torch.sum(raw_gt['temporal_labels'][0][raw_gt['framestps'][0][0]:raw_gt['framestps'][0][1]+1]) != raw_gt['framestps'][0][1] - raw_gt['framestps'][0][0] +1:
        #     count +=1

        print('Aug')
        print('aug_timestps', list(map(lambda x: (round(x[0], 2), round(x[1], 2)), aug_gt['timestps'].numpy().tolist()))[0])
        print('aug_framestps', aug_gt['framestps'][0])
        print('nfeats', aug_nfeats.item())
        framestamps = torch.from_numpy(np.array(aug_gt['framestps'])).float()
        pred_time = dataset.frame2sec(framestamps, duration=video_duration, nfeats=aug_nfeats)
        print(pred_time[0].numpy().tolist())
        aug_s, aug_e = aug_gt['framestps'][0][0], aug_gt['framestps'][0][1]
        print('temp_labels',
              aug_gt['temporal_labels'][0][aug_s:aug_e + 1].numpy().tolist())
        print(torch.sum(aug_gt['temporal_labels'][0][aug_s:aug_e+1]))
        if torch.sum(aug_gt['temporal_labels'][0][aug_s:aug_e+1]) != aug_e - aug_s +1:
            count +=1

        print('*' * 80)
        # if video_duration[0] < ts_time[0][0] or video_duration[0] < ts_time[0][1]:
        #     print('vid',vid_list, 'duration',video_duration.detach().numpy().tolist(), 'framestamps', ts_time.numpy().tolist())
        #     count +=1
        #break
    logger.info('test_done')
    print(count)