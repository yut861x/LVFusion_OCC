import torch
import numpy as np

from . import iou3d_cuda


def boxes_iou_bev(boxes_a, boxes_b):
    """Calculate boxes IoU in the bird view.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5).

    Returns:
        ans_iou (torch.Tensor): IoU result with shape (M, N).
    """
    ans_iou = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))

    iou3d_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(),
                                 ans_iou)

    return ans_iou


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """Nms function with gpu implementation.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (int): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    order = scores.sort(0, descending=True)[1]

    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh, boxes.device.index)
    keep = order[keep[:num_out].cuda(boxes.device)].contiguous()

    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_cpu(boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
    """CPU version of NMS for rotated boxes.
    
    Args:
        boxes (torch.Tensor): Input boxes with the shape of [N, 5]
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of [N].
        thresh (float): Threshold.
        pre_maxsize (int): Max size of boxes before nms. Default: None.
        post_maxsize (int): Max size of boxes after nms. Default: None.

    Returns:
        torch.Tensor: Indexes after nms.
    """
    # 转换为numpy进行CPU计算
    if isinstance(boxes, torch.Tensor):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = boxes
    if isinstance(scores, torch.Tensor):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = scores
    
    # 排序
    order = scores_np.argsort()[::-1]  # 从高到低
    
    if pre_maxsize is not None:
        order = order[:pre_maxsize]
    
    boxes_sorted = boxes_np[order]
    keep = []
    suppressed = np.zeros(len(boxes_sorted), dtype=np.bool_)
    
    for i in range(len(boxes_sorted)):
        if suppressed[i]:
            continue
        keep.append(order[i])
        
        # 计算IoU并抑制重叠的框
        for j in range(i + 1, len(boxes_sorted)):
            if suppressed[j]:
                continue
            
            # 简化的IoU计算（对于旋转框，这里使用简化的BEV IoU）
            box_i = boxes_sorted[i]
            box_j = boxes_sorted[j]
            
            # 计算BEV IoU（简化版本）
            x1_i, y1_i, x2_i, y2_i = box_i[0], box_i[1], box_i[2], box_i[3]
            x1_j, y1_j, x2_j, y2_j = box_j[0], box_j[1], box_j[2], box_j[3]
            
            # 计算交集
            inter_x1 = max(x1_i, x1_j)
            inter_y1 = max(y1_i, y1_j)
            inter_x2 = min(x2_i, x2_j)
            inter_y2 = min(y2_i, y2_j)
            
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                iou = 0.0
            else:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                area_i = (x2_i - x1_i) * (y2_i - y1_i)
                area_j = (x2_j - x1_j) * (y2_j - y1_j)
                union_area = area_i + area_j - inter_area
                iou = inter_area / union_area if union_area > 0 else 0.0
            
            if iou > thresh:
                suppressed[j] = True
    
    keep = torch.tensor(keep, dtype=torch.long, device=boxes.device if isinstance(boxes, torch.Tensor) else torch.device('cpu'))
    
    if post_max_size is not None:
        keep = keep[:post_max_size]
    
    return keep


def nms_normal_gpu(boxes, scores, thresh):
    """Normal non maximum suppression on GPU.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5).
        scores (torch.Tensor): Scores of predicted boxes with shape (N).
        thresh (torch.Tensor): Threshold of non maximum suppression.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.zeros(boxes.size(0), dtype=torch.long)
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh,
                                        boxes.device.index)
    return order[keep[:num_out].cuda(boxes.device)].contiguous()
