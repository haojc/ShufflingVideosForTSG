import numpy as np
import time
import random

class DataAugmentForTSG():
    def __init__(self, seed, aug_percentage, mode='all'):
        np.random.seed(seed)
        self.aug_percentage = aug_percentage
        self.protected_ratio = 0.2
        self.count = 0
        self.aug_mode = mode

        if self.aug_mode in ['all']:
            self.fn_candidate = [self.protected_gt_moment_crop, self.gt_moment_cropout]
        elif self.aug_mode in ['gt_crop']:
            self.fn_candidate = [self.gt_moment_crop]
        elif self.aug_mode in ['gt_cropout']:
            self.fn_candidate = [self.gt_moment_cropout]
        elif self.aug_mode in ['prot_gt_crop']:
            self.fn_candidate = [self.protected_gt_moment_crop]
        elif self.aug_mode in ['gt_translate']:
            self.fn_candidate = [self.gt_moment_translate]
        elif self.aug_mode in ['shuffle_temporal']:
            self.fn_candidate = [self.shuffel_temporal_order_by_short_segments]
        else:
            self.fn_candidate = [self.gt_moment_crop, self.gt_moment_cropout]

    def aug_data(self, framestps, nfeats, video_feat, min_crop_width_ratio=0.2, max_crop_width_ratio=0.5):
        aug_prob = np.random.rand(1)[0]
        if aug_prob > self.aug_percentage:
            self.count +=1
            return framestps, nfeats, video_feat

        # fn_idx = np.random.random_integers(len(self.fn_candidate)-1) if len(self.fn_candidate) > 1 else 0
        fn_idx = random.randint(0, len(self.fn_candidate)-1) if len(self.fn_candidate) > 1 else 0
        return self.fn_candidate[fn_idx](framestps, nfeats, video_feat, min_crop_width_ratio, max_crop_width_ratio)

    def gt_moment_crop(self, framestps, nfeats, video_feat, min_crop_width_ratio=0.2, max_crop_width_ratio=0.5,
                       crop_width=None, crop_start=None):
        raw_start, raw_end = framestps
        gt_moment_len = raw_end - raw_start + 1
        if gt_moment_len <= 1:
            return framestps, nfeats, video_feat

        if crop_width is None or crop_width >= gt_moment_len:
            max_crop_width = np.ceil(gt_moment_len * max_crop_width_ratio)
            min_crop_width = np.ceil(gt_moment_len * min_crop_width_ratio)
            # crop_width = np.random.random_integers(min_crop_width, max_crop_width)
            crop_width = random.randint(min_crop_width, max_crop_width)

        if crop_start is None or crop_start < raw_start or crop_start > raw_end:
            # crop_start = np.random.random_integers(raw_start, raw_end - crop_width)
            crop_start = random.randint(raw_start, raw_end - crop_width + 1)
        crop_end = crop_start + crop_width - 1

        crop_video_feat = np.delete(video_feat.copy(), [a for a in range(crop_start, crop_end + 1)], 1)
        new_video_feat = np.zeros(video_feat.shape) + 0.0
        new_video_feat[0, :crop_video_feat.shape[1], :] = crop_video_feat[0, :crop_video_feat.shape[1], :]
        new_start = raw_start
        new_end = raw_end - crop_width
        new_nfeats = nfeats - crop_width

        return [new_start, new_end], new_nfeats, new_video_feat

    def protected_gt_moment_crop(self, framestps, nfeats, video_feat, min_crop_width_ratio=0.2,
                                 max_crop_width_ratio=0.5,
                                 crop_width=None, crop_start=None):
        raw_start, raw_end = framestps
        gt_moment_len = raw_end - raw_start + 1
        if gt_moment_len <= 1:
            return framestps, nfeats, video_feat
        protected_start = raw_start + np.ceil(gt_moment_len * self.protected_ratio)
        protected_end = raw_end - np.ceil(gt_moment_len * self.protected_ratio)

        if crop_width is None or crop_width > gt_moment_len:
            max_crop_width = np.ceil((protected_end - protected_start) * max_crop_width_ratio)
            min_crop_width = np.ceil((protected_end - protected_start) * min_crop_width_ratio)
            # crop_width = np.random.random_integers(min_crop_width, max_crop_width)
            crop_width = random.randint(min_crop_width, max_crop_width)

        if crop_start is None or crop_start < raw_start or crop_start > raw_end:
            # crop_start = np.random.random_integers(protected_start, protected_end - crop_width)
            crop_start = random.randint(protected_start, protected_end - crop_width + 1)
        crop_end = crop_start + crop_width - 1

        crop_video_feat = np.delete(video_feat.copy(), [a for a in range(crop_start, crop_end + 1)], 1)
        new_video_feat = np.zeros(video_feat.shape) + 0.0
        new_video_feat[0, :crop_video_feat.shape[1], :] = crop_video_feat[0, :crop_video_feat.shape[1], :]
        new_start = raw_start
        new_end = raw_end - crop_width
        new_nfeats = nfeats - crop_width

        return [new_start, new_end], new_nfeats, new_video_feat

    def gt_moment_cropout(self, framestps, nfeats, video_feat, min_crop_width_ratio=0.2, max_crop_width_ratio=0.5, ):
        raw_start, raw_end = framestps
        gt_moment_len = raw_end - raw_start + 1
        if gt_moment_len <= 1:
            return framestps, nfeats, video_feat

        protected_start_l = raw_start - np.ceil(gt_moment_len * self.protected_ratio)
        protected_start_r = raw_start + np.ceil(gt_moment_len * self.protected_ratio)
        protected_end_l = raw_end - np.ceil(gt_moment_len * self.protected_ratio)
        protected_end_r = raw_end + np.ceil(gt_moment_len * self.protected_ratio)

        # determine internal region in gt_moment to be croped
        max_crop_width = np.ceil((protected_end_l - protected_start_r) * max_crop_width_ratio)
        min_crop_width = np.ceil((protected_end_l - protected_start_r) * min_crop_width_ratio)
        # crop_width = np.random.random_integers(min_crop_width, max_crop_width)
        crop_width = random.randint(min_crop_width, max_crop_width)
        if crop_width <= 0:
            return self.gt_moment_crop(framestps, nfeats, video_feat, min_crop_width_ratio, max_crop_width_ratio)
        # cropout_start = np.random.random_integers(protected_start_r, protected_end_l - crop_width)
        cropout_start = random.randint(protected_start_r, protected_end_l - crop_width + 1)
        cropout_end = cropout_start + crop_width - 1

        # determine outernal region to be croped in
        candidate = []
        if protected_start_l >= crop_width:
            candidate += [a for a in range(int(protected_start_l))]
        if nfeats - 1 - protected_end_r >= crop_width:
            candidate += [a for a in range(int(protected_end_r), nfeats - crop_width)]
        if len(candidate) == 0:
            return self.gt_moment_crop(framestps, nfeats, video_feat, min_crop_width_ratio, max_crop_width_ratio,
                                       crop_width, cropout_start)
        # cropin_start = candidate[np.random.random_integers(len(candidate) - 1) if len(candidate) > 1 else 0]
        cropin_start = candidate[random.randint(0, len(candidate) - 1) if len(candidate) > 1 else 0]
        cropin_end = cropin_start + crop_width - 1

        new_video_feat = video_feat.copy()
        new_video_feat[0, cropout_start:cropout_end+1, :] = video_feat[0, cropin_start:cropin_end+1]

        return framestps, nfeats, new_video_feat

    def gt_moment_translate(self, framestps, nfeats, video_feat, *args):
        raw_start, raw_end = framestps
        gt_moment_len = raw_end - raw_start + 1
        if gt_moment_len <= 1 or gt_moment_len >= nfeats:
            return framestps, nfeats, video_feat

        wo_len = nfeats - gt_moment_len
        video_feat_wo_gt = np.zeros(video_feat.shape) + 0.0
        video_feat_wo_gt[0, 0:raw_start, :] = video_feat[0, 0:raw_start]
        if raw_start < wo_len:
            # print(video_feat.shape, raw_start, raw_end, nfeats, wo_len)
            video_feat_wo_gt[0, raw_start:wo_len, :] = video_feat[0, raw_end + 1:nfeats]

        # cropin_start = np.random.random_integers(0, wo_len+1)
        cropin_start = random.randint(0, wo_len)
        insert_index = [cropin_start for a in range(gt_moment_len)]

        video_feat_shuffle_gt = np.insert(video_feat_wo_gt, insert_index, video_feat[0, raw_start: raw_end + 1], 1)
        new_video_feat = np.zeros(video_feat.shape) + 0.0
        new_video_feat[0, :video_feat.shape[1]] = video_feat_shuffle_gt[0, :video_feat.shape[1], :]

        return [cropin_start, cropin_start + gt_moment_len - 1], nfeats, new_video_feat

    def shuffel_temporal_order_by_short_segments(self, framestps, nfeats, video_feat, seg_len, *args):
        _, T, D = video_feat.shape
        T_ = T//seg_len
        reshaped_video_feat = np.reshape(video_feat, (T_, seg_len, D))
        shuffle_ix = np.random.permutation(np.arange(T_))
        new_video_feat = reshaped_video_feat[shuffle_ix].reshape((1, T, D))
        return framestps, nfeats, new_video_feat

    def shuffel_temporal_order_by_short_segments_pad(self, framestps, nfeats, video_feat, seg_len, *args):
        _, raw_T, D = video_feat.shape
        video_feat = self.pad_vfeat(video_feat, seg_len)
        _, T, D = video_feat.shape
        T_ = T // seg_len
        reshaped_video_feat = np.reshape(video_feat, (T_, seg_len, D))
        shuffle_ix = np.random.permutation(np.arange(T_))
        new_video_feat = reshaped_video_feat[shuffle_ix].reshape((1, T, D))
        return framestps, nfeats, new_video_feat[:, :raw_T]

    def pad_vfeat(self, video_feat, seg_len):
        _, T, D = video_feat.shape
        pad = T % seg_len
        if pad == 0:
            return video_feat

        pad = seg_len - pad
        new_video_feat = np.zeros((1, T + pad, D))
        new_video_feat[:, :T] = video_feat
        return new_video_feat

    def shuffel_temporal_order_by_short_segments2(self, framestps, nfeats, video_feat, seg_len, *args):
        _, raw_T, D = video_feat.shape
        video_feat = video_feat[:,:nfeats]
        video_feat = self.pad_vfeat(video_feat, seg_len)
        _, T, D = video_feat.shape
        T_ = T // seg_len
        reshaped_video_feat = np.reshape(video_feat, (T_, seg_len, D))
        shuffle_ix = np.random.permutation(np.arange(T_))
        # print(shuffle_ix)
        shuffled_feat = reshaped_video_feat[shuffle_ix].reshape((1, T, D))
        new_video_feat = np.zeros((1, raw_T,D))
        new_nfeats = min(raw_T, T)
        new_video_feat[0,:new_nfeats] = shuffled_feat[0,:new_nfeats]
        return framestps, T, new_video_feat

if __name__ == '__main__':
    dataaug = DataAugmentForTSG(seed=3, aug_percentage=0, mode='gt_translate')

    T= 40
    dim=1
    video_normal_feat = np.arange(0,T).repeat(dim).reshape((1, T, dim))
    nfeats = 40
    print('Raw', video_normal_feat.reshape((T)))

    framestps_lists = [[10, 20], [0, 1], [0, 2], [0, nfeats - 2], [0, nfeats - 1], [38, 39], [37, 39]]

    # fn = dataaug.gt_moment_crop
    # # fn = dataaug.protected_gt_moment_crop
    fn = dataaug.gt_moment_translate
    for fstps in framestps_lists:
        new_ftps, new_nfeats, new_video_feat = fn(fstps, nfeats, video_normal_feat)
        # sum = np.sum(new_video_feat)
        print(fstps, '->', new_ftps, new_nfeats, new_video_feat.reshape((T)))

    for i in range(100):
        fstps = framestps_lists[i%len(framestps_lists)]
        new_ftps, new_nfeats, new_video_feat = dataaug.aug_data(fstps, nfeats, video_normal_feat)

    print(dataaug.count)

