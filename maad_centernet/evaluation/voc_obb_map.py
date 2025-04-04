# code originally from: https://github.com/CAPTAIN-WHU/DOTA_devkit/blob/master/dota-v1.5_evaluation_task1.py
# Modified by: Ronja Güldenring, 2022

# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

import numpy as np
from shapely.geometry import Polygon


def parse_gt(filename):
    """

    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(" ")
                object_struct = {}
                if len(splitlines) < 9:
                    continue
                object_struct["name"] = splitlines[8]
                object_struct["difficult"] = 0
                object_struct["bbox"] = [
                    float(splitlines[0]),
                    float(splitlines[1]),
                    float(splitlines[2]),
                    float(splitlines[3]),
                    float(splitlines[4]),
                    float(splitlines[5]),
                    float(splitlines[6]),
                    float(splitlines[7]),
                ]
                objects.append(object_struct)
            else:
                break
    return objects


def voc_ap(rec, prec, use_07_metric=False):
    """ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(
    detpath,
    annopath,
    imagesetfile,
    classname,
    # cachedir,
    ovthresh=0.5,
    use_07_metric=False,
):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # read list of images
    with open(imagesetfile, "r") as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        # print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj["name"] == classname]
        bbox = np.array([x["bbox"] for x in R])
        difficult = np.array([x["difficult"] for x in R]).astype(bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {
            "bbox": bbox,
            "difficult": difficult,
            "det": det,
        }

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, "r") as f:
        lines = f.readlines()

    splitlines = [x.strip().split(" ") for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)

    ## note the usage only in numpy not for list
    if len(sorted_ind) > 0:
        BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R["bbox"].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            BBGT_xmin = np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih

            # union
            uni = (
                (bb_xmax - bb_xmin + 1.0) * (bb_ymax - bb_ymin + 1.0)
                + (BBGT_xmax - BBGT_xmin + 1.0) * (BBGT_ymax - BBGT_ymin + 1.0)
                - inters
            )

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]

            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    # overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    bb_pred_shapely = Polygon(
                        [
                            (bb[0], bb[1]),
                            (bb[2], bb[3]),
                            (bb[4], bb[5]),
                            (bb[6], bb[7]),
                        ]
                    )
                    gt_shapely = Polygon(
                        [
                            (GT[0], GT[1]),
                            (GT[2], GT[3]),
                            (GT[4], GT[5]),
                            (GT[6], GT[7]),
                        ]
                    )
                    intersect = bb_pred_shapely.intersection(gt_shapely).area
                    union = bb_pred_shapely.union(gt_shapely).area
                    overlap = intersect / union
                    overlaps.append(overlap)
                return overlaps

            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]
        if ovmax > ovthresh:
            if not R["difficult"][jmax]:
                if not R["det"][jmax]:
                    tp[d] = 1.0
                    R["det"][jmax] = 1
                else:
                    fp[d] = 1.0
        else:
            fp[d] = 1.0

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def dota_mAP(detpath, annopath, imagesetfile, classnames, use_07_metric=True):
    classaps = []
    map = 0
    summary_string = "["
    for classname in classnames:
        rec, prec, ap = voc_eval(
            detpath,
            annopath,
            imagesetfile,
            classname,
            ovthresh=0.5,
            use_07_metric=use_07_metric,
        )
        map = map + ap
        summary_string += f"{classname}: {ap}, "
        classaps.append(ap)
    summary_string += "]\n"

    map = map / len(classnames)
    summary_string += (
        f"mAP (VOC, 11-point interpolation: {use_07_metric}): {map}"
    )
    return map, summary_string
