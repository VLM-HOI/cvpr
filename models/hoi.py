from scipy.optimize import linear_sum_assignment

import torch
from torch import nn
import torch.nn.functional as F

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, box_xyxy_to_cxcywh, box_iou,masks_to_boxes, box_area
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import numpy as np
from queue import Queue
import math

from .backbone import build_backbone
from .matcher import build_matcher
from .muren import build_muren

from transformers import AutoProcessor, BlipForImageTextRetrieval
import transformers
import time

class MURENHOI(nn.Module):
    def __init__(self, backbone, transformer, num_obj_classes, num_verb_classes, num_queries, aux_loss=False, args=None):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.query_embed_human = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_obj = nn.Embedding(num_queries, hidden_dim)
        self.query_embed_rel = nn.Embedding(num_queries, hidden_dim)

        self.obj_class_embed = nn.Linear(hidden_dim, num_obj_classes + 1)
        self.verb_class_embed = nn.Linear(hidden_dim, num_verb_classes)
        self.sub_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.obj_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss

        self.use_matching = args.use_matching
        self.dec_layers = args.dec_layers

        if self.use_matching:
            self.matching_embed = MLP(hidden_dim*2, hidden_dim, 2, 3)

    def forward(self, samples: NestedTensor):
        # samples = NestedTensor(samples,samples2)
        if not isinstance(samples, NestedTensor):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None

        trans_outputs = self.transformer(self.input_proj(src), mask,
                                                     self.query_embed_human.weight, self.query_embed_obj.weight, self.query_embed_rel.weight, pos[-1])

        if len(trans_outputs) == 4:
            sub_out, obj_out, rel_out, _ = trans_outputs
        elif len(trans_outputs) == 5:
            sub_out, obj_out, rel_out, cross_out,_ = trans_outputs
        else:
            sub_out, obj_out, rel_out, _, sub_att, obj_att, rel_att = trans_outputs

        outputs_sub_coord = self.sub_bbox_embed(sub_out).sigmoid()
        outputs_obj_coord = self.obj_bbox_embed(obj_out).sigmoid()
        outputs_obj_class = self.obj_class_embed(obj_out)
        if self.use_matching:
            outputs_matching = self.matching_embed(torch.cat([sub_out,obj_out],dim=-1))

        outputs_verb_class = self.verb_class_embed(rel_out)

        out = {'pred_obj_logits': outputs_obj_class[-1], 'pred_verb_logits': outputs_verb_class[-1],
               'pred_sub_boxes': outputs_sub_coord[-1], 'pred_obj_boxes': outputs_obj_coord[-1], 'sub_out': sub_out, 'obj_out': obj_out, 'rel_out': rel_out}

        if self.use_matching:
            out['pred_matching_logits'] = outputs_matching[-1]

        if self.aux_loss:
            if self.use_matching:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord,
                                                        outputs_matching)
            else:
                out['aux_outputs'] = self._set_aux_loss(outputs_obj_class, outputs_verb_class,
                                                        outputs_sub_coord, outputs_obj_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_obj_class, outputs_verb_class, outputs_sub_coord, outputs_obj_coord, outputs_matching=None):
        min_dec_layers_num = self.dec_layers
        if self.use_matching:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, \
                     'pred_obj_boxes': d, 'pred_matching_logits': e}
                    for a, b, c, d, e in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_verb_class[-min_dec_layers_num : -1], \
                                             outputs_sub_coord[-min_dec_layers_num : -1], outputs_obj_coord[-min_dec_layers_num : -1], \
                                             outputs_matching[-min_dec_layers_num : -1])]
        else:
            return [{'pred_obj_logits': a, 'pred_verb_logits': b, 'pred_sub_boxes': c, 'pred_obj_boxes': d}
                    for a, b, c, d in zip(outputs_obj_class[-min_dec_layers_num : -1], outputs_verb_class[-min_dec_layers_num : -1], \
                                          outputs_sub_coord[-min_dec_layers_num : -1], outputs_obj_coord[-min_dec_layers_num : -1])]





class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class SetCriterionHOI(nn.Module):

    def __init__(self, num_obj_classes, num_queries, num_verb_classes, matcher, weight_dict, eos_coef, losses, args):
        super().__init__()

        self.args = args
        self.num_obj_classes = num_obj_classes
        self.num_queries = num_queries
        self.num_verb_classes = num_verb_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_obj_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.name_actions = args.action_names.copy()
        self.name_objects = args.object_names.copy()

        self.alpha = args.alpha

        
        self.itm_model = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco", torch_dtype=torch.float16).to("cuda")
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip-itm-base-coco")

        for p in self.itm_model.parameters():
            p.requires_grad = False

    def loss_obj_labels(self, outputs, targets, indices, num_interactions, raw_img, log=True):
        assert 'pred_obj_logits' in outputs
        src_logits = outputs['pred_obj_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['obj_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_obj_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o


        obj_weights = self.empty_weight

        loss_obj_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, obj_weights)
        losses = {'loss_obj_ce': loss_obj_ce}

        if log:
            losses['obj_class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_obj_cardinality(self, outputs, targets, indices, num_interactions, raw_img):
        pred_logits = outputs['pred_obj_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['obj_labels']) for v in targets], device=device)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'obj_cardinality_error': card_err}
        return losses

    def loss_verb_labels(self, outputs, targets, indices, num_interactions, raw_img):
        assert 'pred_verb_logits' in outputs
        src_logits = outputs['pred_verb_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['verb_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.zeros_like(src_logits)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.sigmoid()
        loss_verb_ce = self._neg_loss(src_logits, target_classes, alpha=self.alpha)

        losses = {'loss_verb_ce': loss_verb_ce}
        return losses


    def loss_sub_obj_boxes(self, outputs, targets, indices, num_interactions, raw_img):
        assert 'pred_sub_boxes' in outputs and 'pred_obj_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_sub_boxes = outputs['pred_sub_boxes'][idx]
        src_obj_boxes = outputs['pred_obj_boxes'][idx]
        target_sub_boxes = torch.cat([t['sub_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_obj_boxes = torch.cat([t['obj_boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        exist_obj_boxes = (target_obj_boxes != 0).any(dim=1)

        losses = {}
        if src_sub_boxes.shape[0] == 0:
            losses['loss_sub_bbox'] = src_sub_boxes.sum()
            losses['loss_obj_bbox'] = src_obj_boxes.sum()
            losses['loss_sub_giou'] = src_sub_boxes.sum()
            losses['loss_obj_giou'] = src_obj_boxes.sum()
        else:
            loss_sub_bbox = F.l1_loss(src_sub_boxes, target_sub_boxes, reduction='none')
            loss_obj_bbox = F.l1_loss(src_obj_boxes, target_obj_boxes, reduction='none')
            losses['loss_sub_bbox'] = loss_sub_bbox.sum() / num_interactions
            losses['loss_obj_bbox'] = (loss_obj_bbox * exist_obj_boxes.unsqueeze(1)).sum() / (exist_obj_boxes.sum() + 1e-4)
            loss_sub_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_sub_boxes),
                                                               box_cxcywh_to_xyxy(target_sub_boxes)))
            loss_obj_giou = 1 - torch.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_obj_boxes),
                                                               box_cxcywh_to_xyxy(target_obj_boxes)))
            losses['loss_sub_giou'] = loss_sub_giou.sum() / num_interactions
            losses['loss_obj_giou'] = (loss_obj_giou * exist_obj_boxes).sum() / (exist_obj_boxes.sum() + 1e-4)
        return losses

    def loss_matching_labels(self, outputs, targets, indices, num_interactions, log=True):
        assert 'pred_matching_logits' in outputs
        src_logits = outputs['pred_matching_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t['matching_labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_matching = F.cross_entropy(src_logits.transpose(1, 2), target_classes)
        losses = {'loss_matching': loss_matching}

        if log:
            losses['matching_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    def loss_itm(self, outputs, targets, hoi_indices, num_boxes, raw_img):
        # assert 'pred_matching_logits' in outputs

        start = time.time()
        src_actions = outputs['pred_verb_logits']
        idx = self._get_src_permutation_idx(hoi_indices)

        # Construct Target --------------------------------------------------------------------------------------------------------------
        target_classes_v = torch.cat([t["verb_labels"][J] for t, (_, J) in zip(targets, hoi_indices)])
        target_classes = torch.full(src_actions.shape, 0, dtype=torch.float32, device=src_actions.device)
        target_classes[..., -1] = 1 # the last index for no-interaction is '1' if a label exists

        pos_classes = torch.full(target_classes[idx].shape, 0, dtype=torch.float32, device=src_actions.device) # else, the last index for no-interaction is '0'
        pos_classes[:] = target_classes_v.float()
        target_classes[idx] = pos_classes

        # Parse the target action classes to text format --------------------------------------------------------------------------------------
        target_classes_dict = self.args.action_names.copy()
        target_classes_dict.append('no interaction')
        logits = src_actions.sigmoid()

        pred_text = []
        max_idx = torch.argmax(logits[..., :-1], dim=-1)
        for i in range(max_idx.shape[0]):
            temp = []
            for j in range(max_idx.shape[1]):
                temp.append('person ' + target_classes_dict[max_idx[i][j].item()].split(' ')[0])
            pred_text.append(temp)

        # Parse the predicted object classes to text format --------------------------------------------------------------------------------------
        obj_classes_dict = self.name_objects
        obj_logits = outputs['pred_obj_logits']
        
        pred_o_idx = []
        pred_text_o = []
        max_idx_o = torch.argmax(obj_logits, dim=-1)
        for i in range(max_idx_o.shape[0]):
            temp = []
            pos_temp = []
            for j in range(max_idx_o.shape[1]):
                if max_idx_o[i][j].item() == obj_logits.shape[-1]-1 or max_idx_o[i][j].item() == 1:
                    continue
                else:
                    pos_temp.append(max_idx_o[i][j].item())
                    temp.append(obj_classes_dict[max_idx_o[i][j].item()])
            pred_o_idx.append(list(set(pos_temp)))
            pred_text_o.append(list(set(temp)))
        # pred_text_o = list(set(pred_text_o))
        target_obj_idx = []
        target_obj_txt = []
        for t in targets:
            if self.args.dataset_file == 'hico':
                label = t["labels"].tolist()
            elif self.args.dataset_file == 'vcoco':
                label = t["obj_labels"].tolist()
            temp = []
            to_remove = [0, 80]
            if 0 or 80 in label:
                # Remove All "Person"
                label = [x for x in label if x not in to_remove]
            # if 80 in label:
            #     # Remove All "Background"
            #     label = [x for x in label if x not in to_remove]
            for i in range(len(label)):
                temp.append(obj_classes_dict[label[i]])
            target_obj_idx.append(label)
            target_obj_txt.append(temp)

        # Find object pos and neg index and there text --------------------------------------------------------------------------------------------------------------
        pred_o_pos_text = []
        pred_o_neg_text = []
        for i in range(len(pred_text_o)):
            temp_pos = []
            temp_neg = []
            for j in range(len(pred_text_o[i])):
                if pred_text_o[i][j] in target_obj_txt[i]:
                    temp_pos.append(pred_text_o[i][j].replace('_', ' '))
                else:
                    temp_neg.append(pred_text_o[i][j].replace('_', ' '))
            pred_o_pos_text.append(list(set(temp_pos)))
            pred_o_neg_text.append(list(set(temp_neg)))



        # --------------------------------------------------------------------------------------------------------------------------------
        # Parse the GT action classes to text format -------------------------------------------------------------------------------------------
        
        gt_verb = target_classes.cpu().numpy()
        gt_idx = []
        max_gt_idx = []
        if self.args.dataset_file == 'vcoco':
            for t in targets:
                label = t["obj_labels"].tolist()
                act = t["verb_labels"].tolist()
                temp = []
                for i, v in enumerate(label):
                    obj_act = act[i]
                    for j in range(len(obj_act)):
                        if obj_act[j] == 1:
                            temp.append(target_classes_dict[j].split(' ')[0])
                gt_idx.append(temp)

        elif self.args.dataset_file == 'hico':
            for i in range(gt_verb.shape[0]):
                _, g_idx = np.where(gt_verb[i] == 1)
                max_gt_idx.append(g_idx)
                temp = []
                for j in g_idx:
                    if j == 28:
                        continue
                    else:
                        temp.append(target_classes_dict[j].split(' ')[0])
                gt_idx.append(temp)
        # gt_idx = np.array(gt_idx)
        prefix = "person "
        suffix = " something"
        gt_idx = [[prefix + x for x in gt_idx[y]] for y in range(len(gt_idx))]

        # Parse positive index ----------------------------------------------------------------------------------------------------------
        pred_pos_text = []
        pred_neg_text = []
        for i in range(len(pred_text)):
            temp_pos = []
            temp_neg = []
            for j in range(len(pred_text[i])):
                if pred_text[i][j] in gt_idx[i]:
                    temp_pos.append(pred_text[i][j].replace('_', ' '))
                else:
                    temp_neg.append(pred_text[i][j].replace('_', ' '))
            pred_pos_text.append(list(set(temp_pos)))
            pred_neg_text.append(list(set(temp_neg)))

        pred_pos_sentence = [[a + " " + b for a, b in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(pred_pos_text, pred_o_pos_text)]
        pred_neg_sentence = [[a + " " + b for a, b in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(pred_neg_text, pred_o_neg_text)]
        # ITM Ranking loss ---------------------------------------------------------------------------------------------------------------
        pos_itm_score = 0
        neg_itm_score = 0

        end = time.time()
        elapsed_01 = end - start
        # print("Time taken for parsing: ", elapsed)


        start = time.time()
        if self.args.dataset_file == 'hico':
            if not pred_pos_sentence:
                pos_itm_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
            else:
                temp_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
                # inner_empty = all(not inner for inner in pred_pos_text)
                for i in range(len(pred_pos_sentence)):
                    if not pred_pos_sentence[i]:
                        temp_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
                    else:
                        # batch_img = tuple(raw_img[i] for _ in range(len(pred_pos_sentence[i])))
                        inputs = self.processor(text=pred_pos_sentence[i], images=raw_img[i], return_tensors="pt", padding=True).to("cuda", torch.float16)
                        outputs = self.itm_model(**inputs)
                        temp_score = outputs['itm_score'][...,0].mean()
                    pos_itm_score += temp_score

            if not pred_neg_sentence:
                neg_itm_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
            else:
                temp_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
                # inner_empty = all(not inner for inner in pred_neg_text)
                seq_len = min(4, len(pred_neg_sentence))
                for i in range(seq_len):
                    if not pred_neg_sentence[i]:
                        temp_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
                    else:
                        # batch_img = tuple(raw_img[i] for _ in range(len(pred_neg_sentence[i])))
                        inputs = self.processor(text=pred_neg_sentence[i], images=raw_img[i], return_tensors="pt", padding=True).to("cuda", torch.float16)
                        outputs = self.itm_model(**inputs)
                        temp_score = outputs['itm_score'][...,0].mean()
                    neg_itm_score += temp_score
        elif self.args.dataset_file == 'vcoco':
            if not pred_pos_text:
                pos_itm_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
            else:
                temp_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
                # inner_empty = all(not inner for inner in pred_pos_text)
                for i in range(len(pred_pos_text)):
                    if not pred_pos_text[i]:
                        temp_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
                    else:
                        # batch_img = tuple(raw_img[i] for _ in range(len(pred_pos_sentence[i])))
                        inputs = self.processor(text=pred_pos_text[i], images=raw_img[i], return_tensors="pt", padding=True).to("cuda", torch.float16)
                        outputs = self.itm_model(**inputs)
                        temp_score = outputs['itm_score'][...,0].mean()
                    pos_itm_score += temp_score

            if not pred_neg_text:
                neg_itm_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
            else:
                temp_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
                # inner_empty = all(not inner for inner in pred_neg_text)
                seq_len = min(4, len(pred_neg_text))
                for i in range(seq_len):
                    if not pred_neg_text[i]:
                        temp_score = torch.tensor(0, dtype=torch.float16, device=src_actions.device)
                    else:
                        # batch_img = tuple(raw_img[i] for _ in range(len(pred_neg_sentence[i])))
                        inputs = self.processor(text=pred_neg_text[i], images=raw_img[i], return_tensors="pt", padding=True).to("cuda", torch.float16)
                        outputs = self.itm_model(**inputs)
                        temp_score = outputs['itm_score'][...,0].mean()
                    neg_itm_score += temp_score

        end = time.time()
        elapsed_02 = end - start

        margin = 1
        itm_loss = max(0,  neg_itm_score - pos_itm_score - margin) / self.args.batch_size

        losses = {'loss_itm': itm_loss}

        return losses

    def _neg_loss(self, pred, gt, alpha=0.25):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        loss = 0

        pos_loss = alpha * torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = (1 - alpha) * torch.log(1 - pred) * torch.pow(pred, 2) * neg_inds

        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num, raw_img, **kwargs):
        loss_map = {
            'obj_labels': self.loss_obj_labels,
            'obj_cardinality': self.loss_obj_cardinality,
            'verb_labels': self.loss_verb_labels,
            'sub_obj_boxes': self.loss_sub_obj_boxes,
            'itm': self.loss_itm,
            'matching_labels': self.loss_matching_labels
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num, raw_img, **kwargs)

    def forward(self, outputs, targets, raw_img=None):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

        num_interactions = sum(len(t['obj_labels']) for t in targets)
        num_interactions = torch.as_tensor([num_interactions], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_interactions)
        num_interactions = torch.clamp(num_interactions / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_interactions, raw_img))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'obj_labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_interactions, raw_img, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
        if 'cross_outputs' in outputs:
            indices = self.matcher(outputs_without_aux, targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'obj_labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, outputs['cross_outputs'][-1], targets, indices, num_interactions, raw_img, **kwargs)
                l_dict = {k + f'_cross':  self.args.cl_w * v for k, v in l_dict.items()}
                losses.update(l_dict)

        
        return losses


class PostProcessHOI(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.subject_category_id = args.subject_category_id
        self.use_matching = args.use_matching

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_obj_logits = outputs['pred_obj_logits']
        out_verb_logits = outputs['pred_verb_logits']
        out_sub_boxes = outputs['pred_sub_boxes']
        out_obj_boxes = outputs['pred_obj_boxes']

        assert len(out_obj_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        obj_prob = F.softmax(out_obj_logits, -1)
        obj_scores, obj_labels = obj_prob[..., :-1].max(-1)

        verb_scores = out_verb_logits.sigmoid()

        if self.use_matching:
            out_matching_logits = outputs['pred_matching_logits']
            matching_scores = F.softmax(out_matching_logits, -1)[..., 1]

        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device)
        sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)
        sub_boxes = sub_boxes * scale_fct[:, None, :]
        obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)
        obj_boxes = obj_boxes * scale_fct[:, None, :]

        results = []
        for index in range(len(obj_scores)):
            os, ol, vs, sb, ob =  obj_scores[index], obj_labels[index], verb_scores[index], sub_boxes[index], obj_boxes[index]
            sl = torch.full_like(ol, self.subject_category_id)
            l = torch.cat((sl, ol))
            b = torch.cat((sb, ob))
            results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

            vs = vs * os.unsqueeze(1)

            if self.use_matching:
                ms = matching_scores[index]
                vs = vs * ms.unsqueeze(1)

            ids = torch.arange(b.shape[0])

            results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2],
                                'obj_ids': ids[ids.shape[0] // 2:],'obj_scores':os.to('cpu')})

        return results
def build(args):
    device = torch.device(args.device)
    backbone = build_backbone(args)
    MUREN = build_muren(args)

    model = MURENHOI(
        backbone,
        MUREN,
        num_obj_classes=args.num_obj_classes,
        num_verb_classes=args.num_verb_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        args=args
    )

    matcher = build_matcher(args)
    weight_dict = {}
    weight_dict['loss_obj_ce'] = args.obj_loss_coef
    weight_dict['loss_verb_ce'] = args.verb_loss_coef
    weight_dict['loss_sub_bbox'] = args.bbox_loss_coef
    weight_dict['loss_obj_bbox'] = args.bbox_loss_coef
    weight_dict['loss_sub_giou'] = args.giou_loss_coef
    weight_dict['loss_obj_giou'] = args.giou_loss_coef
    weight_dict['loss_itm'] = args.itm_loss_coef
    if args.use_matching:
        weight_dict['loss_matching'] = args.matching_loss_coef

    if args.aux_loss:
        min_dec_layers_num = args.dec_layers
        aux_weight_dict = {}
        for i in range(min_dec_layers_num - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    losses = ['obj_labels', 'verb_labels', 'sub_obj_boxes', 'obj_cardinality', 'itm']
    if args.use_matching:
        losses.append('matching_labels')


    criterion = SetCriterionHOI(args.num_obj_classes, args.num_queries, args.num_verb_classes, matcher=matcher,
                            weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses,
                            args=args)

    criterion.to(device)
    postprocessors = {'hoi': PostProcessHOI(args)}

    return model, criterion, postprocessors
