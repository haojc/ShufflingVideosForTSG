import json
from torch.utils.data import Dataset, DataLoader
import logging
import numpy as np
import string
import torch
import argparse
import math
import os

from .data_augment import DataAugmentForTSG
from .charades import Sequence_mask, collate_fn

class ANetData(Dataset):
    def __init__(self, annotation_file, feature_file, params, logger):
        super(ANetData,self).__init__()

        self.feature_type = params['feature_type']
        self.SAMPLE_LEN = params['video_len']
        self.MAX_SENTENCE_LEN = params['sent_len']
        logger.info('Video length, %d Sentence Length %d', self.SAMPLE_LEN, self.MAX_SENTENCE_LEN)

        self.annotaion = json.load(open(annotation_file, 'r'))
        self.keys = list(self.annotaion.keys())  # list of 'v_vdieo_id'

        anno_prefix = os.path.splitext(os.path.split(annotation_file)[-1])[0]
        if anno_prefix in ['train', 'train_f', 'anet_train']:
            self.split = 'train'
        elif anno_prefix in ['val_2', 'val_2_f']:
            self.split = 'val_2'
        elif anno_prefix in ['val_1', 'val_1_f']:
            self.split = 'val_1'
        elif anno_prefix in ['anet_test_iid']:
            self.split = 'test_iid'
        elif anno_prefix in ['anet_test_ood']:
            self.split = 'test_ood'
        elif anno_prefix in ['anet_val']:
            self.split = 'val'
        else:
            self.split = 'val_m'
        logger.info('%s, load captioning file, %d videos loaded', self.split, len(self.keys))

        if self.feature_type == 'c3d':
            self.feature_dir = feature_file

        elif self.feature_type == 'i3d':
            self.feature_dir = feature_file


        # data_aug
        self.if_aug = params['if_aug']
        self.data_aug = DataAugmentForTSG(seed=123, aug_percentage=params['aug_percentage'],
                                          mode=params['aug_mode'])

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self,idx):
        raise NotImplementedError()

class ANetDataSentence(ANetData):
    def __init__(self, annotation_file, feature_file, params, logger, num_dataload=None):
        super(ANetDataSentence, self).__init__(annotation_file, feature_file, params, logger)

        self.num_dataload = num_dataload

        self.vfeat_fname = params['vfeat_fn']
        if self.feature_type in ['i3d']:
            self.vfeat_fn = self.sample_1to1_video_feat
            logger.info('vfeat_fn %s', 'sample_1to1_video_feat')
        elif self.vfeat_fname in ['raw']:
            self.vfeat_fn = self.sample_frame2second
        elif self.vfeat_fname in ['114']:
            self.vfeat_fn = self.sample_frame2second_114
        elif self.vfeat_fname in ['lg']:
            self.vfeat_fn = self.lg_get_fixed_length_feat
        else:
            self.vfeat_fname = str(114)
            self.vfeat_fn = self.sample_frame2second_114

        # list of all sentences
        self.sentences = []
        self.sen_idx_in_video = []
        self.sen_vid = []
        for vid in self.annotaion:
            annotation = self.annotaion[vid]
            for idx,sentence in enumerate(annotation['sentences']):
                self.sentences.append(sentence.lower().strip())
                self.sen_vid.append(vid)
                self.sen_idx_in_video.append(idx)

        print('Total %d sentences for training/testing', len(self.sentences))
        for c in string.punctuation:
            if c == ',':
                self.sentences = list(map(lambda x: x.replace(c, ' '), self.sentences))
            else:
                self.sentences = list(map(lambda x: x.replace(c, ''), self.sentences))
        self.sentences = list(map(lambda x: ' '.join(x.replace('\n', '').split()), self.sentences))

        self.ixtoword = np.load(params['ixtoword_path'], allow_pickle=True).tolist()
        self.wordtoix = np.load(params['wordtoix_path'], allow_pickle=True).tolist()
        self.word_emb_init = np.array(np.load(params['word_fts_path']).tolist(), np.float64)

        self.sentence_idxes = list(
            map(lambda x: [self.wordtoix[word] for word in x.lower().split(' ') if word in self.wordtoix], self.sentences))
        self.sentence_lens = list(map(lambda x:len(x), self.sentence_idxes))
        self.pad_sentence_idxes = list(map(
            lambda x: np.pad(np.array(x), (0, self.MAX_SENTENCE_LEN - len(x))).tolist() if len(
                x) < self.MAX_SENTENCE_LEN else np.array(x)[:self.MAX_SENTENCE_LEN],
            self.sentence_idxes))

        assert len(self.pad_sentence_idxes) == len(self.sentences) and len(self.sentence_lens)== len(self.sentences)

        if self.num_dataload is not None:
            self._parse_list()

    def _parse_list(self):
        # repeat the list if the length is less than num_dataload (especially for target data)
        n_repeat = self.num_dataload // len(self.sen_vid)
        n_left = self.num_dataload % len(self.sen_vid)
        self.sen_vid = self.sen_vid * n_repeat + self.sen_vid[:n_left]
        self.sen_idx_in_video = self.sen_idx_in_video * n_repeat + self.sen_idx_in_video[:n_left]
        self.sentences = self.sentences * n_repeat + self.sentences[:n_left]
        self.sentence_lens = self.sentence_lens * n_repeat + self.sentence_lens[:n_left]
        self.pad_sentence_idxes = self.pad_sentence_idxes * n_repeat + self.pad_sentence_idxes[:n_left]


    def __len__(self):
        return len(self.sentences)


    def __getitem__(self, idx):
        vid = self.sen_vid[idx]
        sidx = self.sen_idx_in_video[idx]

        #video information
        video_duration = self.annotaion[vid]['duration']

        #sentence
        sentence = self.sentences[idx]
        sentence_len = self.sentence_lens[idx]
        sentence_idx = self.pad_sentence_idxes[idx]
        sentence_features = list(map(lambda x: self.word_emb_init[x], sentence_idx))
        sentence_features = np.vstack(sentence_features)
        sent_mask = Sequence_mask(self.MAX_SENTENCE_LEN, [0, sentence_len])

        # timestamp:
        timestamps = self.annotaion[vid]['timestamps'][sidx]


        if self.feature_type == 'c3d':
            feature_path = os.path.join(self.feature_dir, vid + '.npy')
            video_feature = np.load(feature_path, 'r')
        elif self.feature_type == 'i3d':
            feature_path = os.path.join(self.feature_dir, vid+'.npy')
            video_feature = np.load(feature_path, 'r')
        else:
            raise NotImplementedError()
        video_normal_feature, framestamps, nfeats = self.vfeat_fn(video_feature, timestamps, video_duration)
        if self.split in ['train'] and self.if_aug:
            framestamps, nfeats, video_normal_feature = self.data_aug.aug_data(framestamps,nfeats,video_normal_feature)

        video_mask = Sequence_mask(self.SAMPLE_LEN, [0, nfeats])
        temporal_labels = Sequence_mask(self.SAMPLE_LEN, framestamps)
        fore_mask = Sequence_mask(self.SAMPLE_LEN, [0, framestamps[0]])
        back_mask = Sequence_mask(self.SAMPLE_LEN, [framestamps[1], nfeats])

        return (sentence, sentence_len, sentence_features, sent_mask,
                video_duration, vid,
                video_normal_feature, timestamps, framestamps, nfeats, video_mask,
                temporal_labels, fore_mask, back_mask)

    def sample_frame2second(self, video_fts, timestamps, duration):
        framestamps = list(map(lambda x: int(x) if int(x) < self.SAMPLE_LEN else self.SAMPLE_LEN-1, timestamps))

        video_fts_shape = np.shape(video_fts)
        video_clip_num = video_fts_shape[0]
        video_fts_dim = video_fts_shape[1]

        output_video_fts = np.zeros([1, self.SAMPLE_LEN, video_fts_dim]) + 0.0

        sample_frame_rate = video_clip_num / duration

        add = 0
        for i in range(self.SAMPLE_LEN):
            if i < duration:
                start_ = max(0, math.floor(i * sample_frame_rate))
                output_video_fts[0, i, :] = video_fts[start_, :]
                add = add + 1

        return output_video_fts, framestamps, add

    def sample_1to1_video_feat(self, video_fts, timestamps, video_duration):
        framestamps = list(map(lambda x: int(x) if int(x) < self.SAMPLE_LEN else self.SAMPLE_LEN-1, timestamps))

        video_fts_shape = np.shape(video_fts)
        video_clip_num = video_fts_shape[0]
        video_fts_dim = video_fts_shape[1]
        output_video_fts = np.zeros([1, self.SAMPLE_LEN, video_fts_dim]) + 0.0
        add = 0
        for i in range(video_clip_num):
            output_video_fts[0, add, :] = video_fts[i, :]
            add += 1
            if add == self.SAMPLE_LEN:
                return output_video_fts, framestamps, add
        #print(add)

        return output_video_fts, framestamps, add

    def sample_frame2second_114(self, video_fts, timestamps, duration):
        framestamps = list(map(lambda x: int(x) if int(x) < self.SAMPLE_LEN else self.SAMPLE_LEN-1, timestamps))

        video_fts_shape = np.shape(video_fts)
        video_clip_num = video_fts_shape[0]
        video_fts_dim = video_fts_shape[1]

        output_video_fts = np.zeros([1, self.SAMPLE_LEN, video_fts_dim]) + 0.0

        sample_frame_rate = video_clip_num / duration

        for i in range(self.SAMPLE_LEN):
            if i < duration:
                start_ = min(video_clip_num-1, max(0, int(i * sample_frame_rate + 0.5)))
                end_ = int((i+1) * sample_frame_rate + 0.5)
                if end_ > video_clip_num or end_ <= start_:
                    output_video_fts[0, i, :] = video_fts[start_, :]
                else:
                    output_video_fts[0, i, :] = np.mean(video_fts[start_:end_, :], 0)

        return output_video_fts, framestamps, video_clip_num

    def lg_get_fixed_length_feat(self, feat, timestamps, video_duration):
        start_pos = min(max(timestamps[0]/video_duration, 0),1)
        end_pos = min(max(timestamps[1]/video_duration, 0),1)
        self.S = self.SAMPLE_LEN
        num_segment = self.SAMPLE_LEN

        nfeats = feat[:, :].shape[0]
        if nfeats <= self.S:
            stride = 1
        else:
            stride = nfeats * 1.0 / num_segment
        if self.split != "train":
            spos = 0
        else:
            random_end = -0.5 + stride
            if random_end == np.floor(random_end):
                random_end = random_end - 1.0
            spos = np.random.random_integers(0, random_end)
        s = np.round(np.arange(spos, nfeats - 0.5, stride)).astype(int)
        start_pos = float(nfeats - 1.0) * start_pos
        end_pos = float(nfeats - 1.0) * end_pos

        if not (nfeats < self.S and len(s) == nfeats) \
                and not (nfeats >= self.S and len(s) == num_segment):
            s = s[:num_segment]  # ignore last one
        assert (nfeats < self.S and len(s) == nfeats) \
               or (nfeats >= self.S and len(s) == num_segment), \
            "{} != {} or {} != {}".format(len(s), nfeats, len(s), num_segment)

        start_index, end_index = None, None
        for i in range(len(s) - 1):
            if s[i] <= end_pos < s[i + 1]:
                end_index = i
            if s[i] <= start_pos < s[i + 1]:
                start_index = i

        if start_index is None:
            start_index = 0
        if end_index is None:
            end_index = num_segment - 1

        cur_feat = feat[s, :]
        nfeats = min(nfeats, num_segment)
        out = np.zeros((1,num_segment, cur_feat.shape[1]))
        out[0, :nfeats, :] = cur_feat
        return out, (start_index, end_index), nfeats

    def _lg_frame2sec(self, framestps, duration, nfeats):
        pos = framestps / nfeats.unsqueeze(1)
        return pos * duration.unsqueeze(1)

    def frame2sec(self, framestps, duration, nfeats):
        if self.vfeat_fname in ['raw', '114']:
            return framestps
        elif self.vfeat_fname in ['lg']:
            return self._lg_frame2sec(framestps, duration, nfeats)
        else:
            raise NotImplementedError

if __name__ == '__main__':

    feature_type = 'c3d'
    parser = argparse.ArgumentParser()
    # c3d
    if feature_type == 'c3d':
        parser.add_argument('--train_data', type=str, default='../../data/ANet/train.json',
                            help='training data path')
        parser.add_argument('--val_data', type=str, default='../../data/ANet/val_merge.json',
                            help='validation data path')
        parser.add_argument('--feature_path', type=str, default='../../data/ANet/c3d_feature',
                            help='feature path')
        parser.add_argument('--feature_type', type=str, default='c3d',
                            help='feature_type')
    else:   # i3d
        parser.add_argument('--train_data', type=str, default='../../data/ANet/train_f.json',
                            help='training data path')
        parser.add_argument('--val_data', type=str, default='../../data/ANet/val_merge_f.json',
                            help='validation data path')
        parser.add_argument('--feature_path', type=str,
                            default= '../../data/ANet/i3d_feature',
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

    parser.add_argument('--vfeat_fn', type=str, default='114',
                        help='')
    params = parser.parse_args()
    params = vars(params)


    logging.basicConfig()
    logger = logging.getLogger('dataset')
    logger.setLevel(logging.INFO)

    dataset = ANetDataSentence(params['train_data'],
                               params['feature_path'],
                               params,
                               logger)
    data_loader = DataLoader(dataset, batch_size=4,
                             shuffle=True, num_workers=0, collate_fn=collate_fn)
    count =0
    for dt in data_loader:
        video_feat, sent_feat, sent_len, ts_time, video_duration, \
        sent_list, vid_list, framestamps, nfeats = dt
        print(video_feat.size())
        print(list(map(lambda x: (round(x[0], 2), round(x[1], 2)), ts_time.numpy().tolist())))
        print('framestamps', framestamps)
        print('vid', vid_list)
        print('sent', sent_list)
        print('duration', video_duration.data)
        print('nfeat', nfeats.data)
        framestamps = torch.from_numpy(np.array(framestamps)).float()
        pred_time = dataset.frame2sec(framestamps, duration=video_duration, nfeats=nfeats)
        print(pred_time)
        print('*' * 80)
        # if torch.floor(video_duration[0]) < torch.floor(ts_time[0][0]) or torch.floor(video_duration[0])< torch.floor(ts_time[0][1]):
        #     print('vid',vid_list, 'duration',video_duration.detach().numpy().tolist(), 'framestamps', ts_time.numpy().tolist())
        #     count += 1
        # break
    logger.info('test_done')