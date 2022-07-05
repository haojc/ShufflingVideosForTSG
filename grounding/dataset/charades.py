import json
from torch.utils.data import Dataset, DataLoader
import h5py
import logging
import numpy as np
import string
import torch
import argparse
import os
from .data_augment import DataAugmentForTSG

def Sequence_mask(max_len, temporal_boundary):
    st, et = temporal_boundary
    mask = np.zeros(shape=[max_len], dtype=np.int32)
    st_ = max(0, st)
    et_ = min(et, max_len - 1)
    mask[st_:(et_ + 1)] = 1
    return mask

def collate_fn(batch):
    batch_size = len(batch)
    sent_list, sent_len, sent_feat, sent_mask, \
    video_duration, vid_list, \
    video_feat, timestamps, framestamps, nfeats, video_mask, \
    temporal_labels, fore_mask, back_mask = zip(*batch)

    sent_len_tensor = torch.from_numpy(np.array(sent_len))
    sent_feat_tensor = torch.from_numpy(np.stack(sent_feat, axis=0)).float()
    sent_mask_tensor = torch.from_numpy(np.stack(sent_mask, axis=0))

    video_tensor = torch.from_numpy(np.vstack(video_feat)).float()
    video_mask_tensor = torch.from_numpy(np.stack(video_mask, axis=0))
    video_duration_tensor = torch.from_numpy(np.array(video_duration))
    nfeats = torch.from_numpy(np.array(nfeats))

    # Gt
    timestamps_tensor = torch.from_numpy(np.array(timestamps)).float()
    temporal_labels = torch.from_numpy(np.stack(temporal_labels, axis=0))
    fore_mask = torch.from_numpy(np.stack(fore_mask, axis=0))
    back_mask = torch.from_numpy(np.stack(back_mask, axis=0))
    gt = {}
    gt['timestps'] = timestamps_tensor
    gt['framestps'] = framestamps
    gt['temporal_labels'] = temporal_labels
    gt['fore_masks'] = fore_mask
    gt['back_masks'] = back_mask

    return sent_list, sent_feat_tensor, sent_len_tensor, sent_mask_tensor, \
           video_duration_tensor, vid_list, \
           video_tensor, nfeats, video_mask_tensor, gt

class CharadesData(Dataset):
    def __init__(self, annotation_file, feature_file, params, logger):
        super(CharadesData,self).__init__()
        self.feature_type = params['feature_type']
        self.SAMPLE_LEN = params['video_len']
        self.MAX_SENTENCE_LEN = params['sent_len']
        logger.info('Video length, %d Sentence Length %d', self.SAMPLE_LEN, self.MAX_SENTENCE_LEN)

        anno_prefix = os.path.splitext(os.path.split(annotation_file)[-1])[0]
        if anno_prefix in ['train', 'train_f', 'charades_train']:
            self.split = 'train'
        elif anno_prefix in ['test', 'test_f', 'charades_test_iid']:
            self.split = 'test'
        elif anno_prefix in ['test_ood', 'charades_test_ood']:
            self.split = 'test_ood'
        else:
            self.split = 'val'
        self.annotaion = json.load(open(annotation_file, 'r'))
        self.keys = list(self.annotaion.keys())  # list of 'v_vdieo_id'
        logger.info('%s, load captioning file, %d videos loaded', self.split, len(self.keys))

        if self.feature_type.lower() in ['c3d', 'c3d_cbp']:
            self.feature_file = h5py.File(feature_file, 'r')
            logger.info('load video feature file, %d video feature obj(%s) loaded',
                        len(self.feature_file.keys()),
                        self.feature_file[self.keys[0]]['c3d_fc6_features'][0].shape)
        elif self.feature_type.lower() in ['i3d', 'i3d_finetune', 'lg_i3d_finetuned', 'lg_i3d_finetune', 'lgi3d']:
            self.feature_file = feature_file
        #glove word2idx
        self.ixtoword = np.load(params['ixtoword_path'], allow_pickle=True).tolist()
        self.wordtoix = np.load(params['wordtoix_path'], allow_pickle=True).tolist()
        self.word_emb_init = np.array(np.load(params['word_fts_path']).tolist(), np.float32)

        # data_aug
        self.if_aug = params['if_aug']
        self.data_aug = DataAugmentForTSG(seed=123, aug_percentage=params['aug_percentage'], mode=params['aug_mode'])
        if self.split in ['train'] and self.if_aug:
            logger.info('Data Augmentation Applied %f  mode: %s', params['aug_percentage'], params['aug_mode'])

    def __len__(self):
        raise NotImplementedError()

    def __getitem(self,idx):
        raise NotImplementedError()
class CharadesDataSentence(CharadesData):
    def __init__(self, annotation_file, feature_file, params, logger):
        super(CharadesDataSentence, self).__init__(annotation_file, feature_file, params, logger)


        self.vfeat_fname = params['vfeat_fn']
        if self.vfeat_fname.lower() == 'lg':
            self.vfeat_fn = self.lg_get_fixed_length_feat
        elif self.feature_type.lower() in ['lgi3d']:
            self.vfeat_fn = self.lg_generate_video_fts_data
        else:
            self.vfeat_fn = self.generate_video_fts_data

        # list of all sentences
        self.sentences = []
        self.sen_idx_in_video = []
        self.sen_vid = []
        for vid in self.annotaion:
            annotation = self.annotaion[vid]
            for idx,sentence in enumerate(annotation['sentences']):
                self.sentences.append(sentence)
                self.sen_vid.append(vid)
                self.sen_idx_in_video.append(idx)

        for c in string.punctuation:
            self.sentences = list(map(lambda x: x.replace(c, ' '), self.sentences))
        logger.info('Total %d sentences for %s', len(self.sentences), self.split)

        self.sentence_idxes = list(
            map(lambda x: [self.wordtoix[word] for word in x.lower().split(' ') if word in self.wordtoix], self.sentences))
        self.sentence_lens = list(map(lambda x:len(x), self.sentence_idxes))
        self.pad_sentence_idxes = list(map(lambda x: np.pad(np.array(x),(0,self.MAX_SENTENCE_LEN-len(x))).tolist(), self.sentence_idxes))

        assert len(self.pad_sentence_idxes) == len(self.sentences) and len(self.sentence_lens)== len(self.sentences)


    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        vid = self.sen_vid[idx]
        sidx = self.sen_idx_in_video[idx]

        #video information
        video_duration = self.annotaion[vid]['video_duration']
        raw_total_frame = video_duration * self.annotaion[vid]['decode_fps']

        #sentence
        sentence = self.sentences[idx]
        sentence_len = self.sentence_lens[idx]
        sentence_idx = self.pad_sentence_idxes[idx]
        sentence_features = list(map(lambda x: self.word_emb_init[x], sentence_idx))
        sentence_features = np.vstack(sentence_features)
        sent_mask = Sequence_mask(self.MAX_SENTENCE_LEN, [0, sentence_len])

        # timestamp:
        timestamps = self.annotaion[vid]['timestamps'][sidx]

        #
        if self.feature_type == 'i3d':
            feature_path = os.path.join(self.feature_file, vid + '.npy')
            video_feature = np.load(feature_path, 'r')
        elif self.feature_type.lower() in ['lgi3d']:
            feature_path = os.path.join(self.feature_file, vid + '.npy')
            video_feature = np.load(feature_path, 'r')
            video_feature = np.resize(video_feature, (-1, 1024))

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

    def generate_video_fts_data(self, video_fts, timestamps, video_duration):
        framestamps = list(map(lambda x: int(x) if int(x) < self.SAMPLE_LEN else self.SAMPLE_LEN-1, timestamps))

        video_fts_shape = np.shape(video_fts)
        video_clip_num = video_fts_shape[0]
        video_fts_dim = video_fts_shape[1]
        output_video_fts = np.zeros([1, self.SAMPLE_LEN, video_fts_dim]) + 0.0
        add = 0
        for i in range(video_clip_num):
            if i % 2 == 0 and i + 1 <= video_clip_num - 1:
                output_video_fts[0, add, :] = np.mean(video_fts[i:i + 2, :], 0)
                add += 1
            elif i % 2 == 0 and i + 1 > video_clip_num - 1:
                output_video_fts[0, add, :] = video_fts[i, :]
                add += 1
            if add == self.SAMPLE_LEN:
                return output_video_fts, framestamps, add
        #print(add)

        return output_video_fts, framestamps, add

    def lg_get_fixed_length_feat(self, feat, timestamps, video_duration):
        start_pos = min(max(timestamps[0] / video_duration, 0), 1)
        end_pos = min(max(timestamps[1] / video_duration, 0), 1)
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
        out = np.zeros((1, num_segment, cur_feat.shape[1]))
        out[0, :nfeats, :] = cur_feat
        return out, (start_index, end_index), nfeats

    def lg_generate_video_fts_data(self, video_fts, timestamps, video_duration):
        """   1frame corresponds to 0.33s """
        framestamps = list(map(lambda x: int(x) if int(x) < self.SAMPLE_LEN else self.SAMPLE_LEN-1, timestamps))

        video_fts_shape = np.shape(video_fts)
        video_clip_num = video_fts_shape[0]
        video_fts_dim = video_fts_shape[1]
        output_video_fts = np.zeros([1, self.SAMPLE_LEN, video_fts_dim]) + 0.0

        add = 0
        for i in range(video_clip_num):
            if i % 3 == 0 and i + 1 <= video_clip_num - 1 and i + 2 <= video_clip_num -1:
                output_video_fts[0, add, :] = np.mean(video_fts[i:i + 3, :], 0)
                add += 1
            elif i % 3 == 0 and i + 1 <= video_clip_num - 1 and i + 2 > video_clip_num -1:
                output_video_fts[0, add, :] = np.mean(video_fts[i:i + 2, :], 0)
                add += 1
            elif i % 3 == 0 and i + 1 > video_clip_num - 1:
                output_video_fts[0, add, :] = video_fts[i, :]
                add += 1

            if add == self.SAMPLE_LEN:
                return output_video_fts, framestamps, add

        return output_video_fts, framestamps, add

    def _lg_frame2sec(self, framestps, duration, nfeats):
        pos = framestps / nfeats.unsqueeze(1)
        return pos * duration.unsqueeze(1)

    def frame2sec(self, framestps, duration, nfeats):
        if self.vfeat_fname in ['lg']:
            return self._lg_frame2sec(framestps, duration, nfeats)
        else:
            return framestps


if __name__ == '__main__':
    feature_type='LG_i3d_finetuned'

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
    parser.add_argument('--aug_percentage', type=float, default=0.5,
                        help='aug_percentage')
    parser.add_argument('--aug_mode', type=str, default=None,
                        help='checkpoint')

    params = parser.parse_args()
    params = vars(params)

    logging.basicConfig()
    logger = logging.getLogger('dataset')
    logger.setLevel(logging.INFO)


    dataset = CharadesDataSentence(params['train_data'],
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
        video_feat, nfeats, video_mask, gt = dt
        ts_time, framestamps = gt['timestps'], gt['framestps']
        # print(list(map(lambda x: (round(x[0],2), round(x[1],2)), ts_time.numpy().tolist())))
        # print('vid',vid_list)
        # print('sent',sent_list)
        # print('duration',video_duration.detach().numpy().tolist())
        # print('nfeats', nfeats.item())
        # print('framestamps', framestamps)
        # framestamps = torch.from_numpy(np.array(framestamps)).float()
        # pred_time = dataset.frame2sec(framestamps, duration=video_duration, nfeats=nfeats)
        # print(pred_time)
        # print('*' * 80)
        # if video_duration[0] < ts_time[0][0] or video_duration[0] < ts_time[0][1]:
        #     print('vid',vid_list, 'duration',video_duration.detach().numpy().tolist(), 'framestamps', ts_time.numpy().tolist())
        #     count +=1
        #break
    logger.info('test_done')
    print(count)