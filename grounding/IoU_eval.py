import numpy as np
import json
import pandas as pd
import argparse

pred_fields = ['results', 'version', 'external_data']

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / (segments_union + 1e-4)
    return tIoU

def wrapper_segment_iou(target_segments, candidate_segments):
    """Compute intersection over union btw segments
    Parameters
    ----------
    target_segments : ndarray
        2-dim array in format [m x 2:=[init, end]]
    candidate_segments : ndarray
        2-dim array in format [n x 2:=[init, end]]
    Outputs
    -------
    tiou : ndarray
        2-dim array [n x m] with IOU ratio.
    Note: It assumes that candidate-segments are more scarce that target-segments
    """
    if candidate_segments.ndim != 2 or target_segments.ndim != 2:
        raise ValueError('Dimension of arguments is incorrect')

    n, m = candidate_segments.shape[0], target_segments.shape[0]
    tiou = np.empty((n, m))
    for i in range(m):
        tiou[:, i] = segment_iou(target_segments[i,:], candidate_segments)

    return tiou

def import_retrieval_proposal(proposal_filename):
    with open(proposal_filename, 'r') as fobj:
        data = json.load(fobj)
    # Checking format...
    if not all([field in data.keys() for field in pred_fields]):
        raise IOError('Please input a valid proposal file.')

    # Read predictions.
    gt_video_lst, gt_start_lst, gt_end_lst = [], [], []
    video_lst, t_start_lst, t_end_lst =[], [],[]
    score_lst = []
    for v_id, v in data['results'].items():
        for idx, result in enumerate(v):
            videoid = v_id+"_"+str(idx)
            gt_video_lst.append(videoid)
            gt_start_lst.append(result['gt_timestamp'][0])
            gt_end_lst.append(result['gt_timestamp'][1])
            video_lst.append(videoid)
            t_start_lst.append(result['timestamp'][0])
            t_end_lst.append(result['timestamp'][1])
            # score_lst.append(result['score'])
            score_lst.append(1)
    proposals = pd.DataFrame({'video-id': video_lst,
                             't-start': t_start_lst,
                             't-end': t_end_lst,
                             'score': score_lst})
    groundtruth = pd.DataFrame(
        {'video-id': gt_video_lst,
         'gt-start': gt_start_lst,
         'gt-end': gt_end_lst,}
    )

    return proposals,groundtruth

def retrieval_eval(filename):
    proposal_filename = filename
    proposals, groundtruth = import_retrieval_proposal(proposal_filename)
    print("=> Proposal loaded over.", proposal_filename)

    topn_lst = [1]
    tIoU_lst = [0.1, 0.3, 0.5, 0.7, 0.9]

    total_sentence = groundtruth.shape[0]
    sentence_lst = proposals['video-id'].unique()
    proposals_gbvn = proposals.groupby('video-id')
    ground_truth_gbvn = groundtruth.groupby('video-id')
    score_lst = []
    for videoid in sentence_lst:
        proposals_videoid = proposals_gbvn.get_group(videoid)
        ground_truth_videoid = ground_truth_gbvn.get_group(videoid)

        this_sentence_ground_truth = ground_truth_videoid.loc[:,
                                     ['gt-start', 'gt-end']].values

        this_sentence_proposals = proposals_videoid.loc[:,
                                  ['t-start', 't-end']].values

        # sort_idx = proposals_videoid['score'].argsort()[::-1]
        # this_sentence_proposals = this_sentence_proposals[sort_idx, :]

        if this_sentence_proposals.ndim != 2:
            this_sentence_proposals = np.expand_dims(this_sentence_proposals, axis=0)
        if this_sentence_ground_truth.ndim != 2:
            this_sentence_ground_truth = np.expand_dims(this_sentence_ground_truth,
                                                        axis=0)

        tiou = wrapper_segment_iou(this_sentence_proposals,
                                   this_sentence_ground_truth)
        score_lst.append(tiou[0])

    # R@k, tIoU = tiou
    positives = np.empty((len(topn_lst), len(tIoU_lst)))
    # baseline_pos = np.empty((len(topn_lst), len(tIoU_lst)))
    for i, topn in enumerate(topn_lst):
        for j, tiou in enumerate(tIoU_lst):
            for s, score in enumerate(score_lst):
                true_positives_tiou = score > tiou
                match = np.max(true_positives_tiou[:topn])
                positives[i, j] += match
    recall = positives / total_sentence

    top1_IoU = np.empty(total_sentence)
    for s, score in enumerate(score_lst):
        top1_IoU[s] = score[0]

    mIoU = round(top1_IoU.mean() * 100, 2)

    print('\tmIoU\t', '\t'.join([str(i) for i in tIoU_lst]))

    print('\n => ')
    for i, topn in enumerate(topn_lst):
        print(topn, '\t', mIoU, '\t', '\t'.join([str(round(i * 100, 2)) for i in recall[i].tolist()]))

    print('mIoU\t{:.4f}'.format(mIoU))

def main(params):
    retrieval_eval(params['submit'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--submit', type=str,
                        default='runs/charades'
                                '/'
                                'TB_baseline_128T_i3d_Vval_QAVE_TEMPnone_4/submits'
                                '/'
                                'TB_baseline_128T_i3d_Vval_QAVE_TEMPnone_4_00016_test'
                                '.json',
                        help='submit file')

    params = parser.parse_args()
    params = vars(params)

    main(params)
