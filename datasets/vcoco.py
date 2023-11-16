from pathlib import Path
from PIL import Image
import json
import numpy as np

import torch
import torch.utils.data
import torchvision
from datasets import meta_data

import datasets.transforms as T

class VCOCO(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, anno_sub, transforms, num_queries):
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries

        self._valid_obj_ids = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
                               14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
                               24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
                               37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
                               48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
                               58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
                               72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
                               82, 84, 85, 86, 87, 88, 89, 90)
        self._valid_verb_ids = range(29)

        self.vcoco_all = self.load_vcoco(anno_sub)
        self.save_action_name()

    def __len__(self):
        return len(self.annotations)
    
    def get_actions(self):
        return self.act_list

    def get_objects(self):
        meta = meta_data._get_coco_instances_meta()
        self.COCO_CLASSES = meta['coco_classes']

        self.valid_obj_names = []

        # self.valid_obj_names.append('None')
        for i in self._valid_obj_ids:
            self.valid_obj_names.append(self.COCO_CLASSES[i])

        self.valid_obj_names.append('Background')
        # self.valid_obj_names = [self.COCO_CLASSES[i] for i in self._valid_obj_ids]

        return self.valid_obj_names
    
    def save_action_name(self):
        self.inst_act_list = list()
        self.act_list = list()

        # add instance action human classes
        self.num_subject_act = 0
        for vcoco in self.vcoco_all:
            self.inst_act_list.append('person ' + vcoco['action_name'])
            self.num_subject_act += 1

        # add instance action object classes
        for vcoco in self.vcoco_all:
            if len(vcoco['role_name']) == 3:
                self.inst_act_list.append('object ' + vcoco['action_name']+'_'+vcoco['role_name'][1])
                self.inst_act_list.append('object ' + vcoco['action_name']+'_'+vcoco['role_name'][2])
            elif len(vcoco['role_name']) < 2:
                continue
            else:
                self.inst_act_list.append('object ' + vcoco['action_name']+'_'+vcoco['role_name'][-1]) # when only two roles

        # add action classes
        for vcoco in self.vcoco_all:
            if len(vcoco['role_name']) == 3:
                self.act_list.append(vcoco['action_name']+' '+vcoco['role_name'][1])
                self.act_list.append(vcoco['action_name']+' '+vcoco['role_name'][2])
            else:
                self.act_list.append(vcoco['action_name']+' '+vcoco['role_name'][-1])

        # self.act_list.append('none')

    def load_vcoco(self, dir_name=None):
        with open(dir_name, 'rt') as f:
            vsrl_data = json.load(f)

        for i in range(len(vsrl_data)):
            vsrl_data[i]['role_object_id'] = np.array(vsrl_data[i]['role_object_id']).reshape((len(vsrl_data[i]['role_name']),-1)).T
            for j in ['ann_id', 'label', 'image_id']:
                vsrl_data[i][j] = np.array(vsrl_data[i][j]).reshape((-1,1))

        return vsrl_data
    
    def __getitem__(self, idx):
        img_anno = self.annotations[idx]

        image = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = image.size

        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)

        target = {}
        target['filename'] = img_anno['file_name']
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(image, target)

            kept_box_indices = [label[0] for label in target['labels']]

            target['labels'] = target['labels'][:, 1] # Remove Person class

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or \
                   (hoi['object_id'] != -1 and hoi['object_id'] not in kept_box_indices):
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    if hoi['object_id'] == -1:
                        obj_labels.append(torch.tensor(len(self._valid_obj_ids)))
                    else:
                        obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    if hoi['object_id'] == -1:
                        obj_box = torch.zeros((4,), dtype=torch.float32)
                    else:
                        obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    verb_labels.append(verb_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)

            target['filename'] = img_anno['file_name']
            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['matching_labels'] = torch.zeros((0,), dtype=torch.int64)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
                target['matching_labels'] = torch.ones_like(target['obj_labels'])
        else:
            target['filename'] = img_anno['file_name']
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx
            target['img_id'] = int(img_anno['file_name'].rstrip('.jpg').split('_')[2])

            if self._transforms is not None:
                img, _ = self._transforms(image, None)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        return img, target, image

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_vcoco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.hoi_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'images' / 'train2014', root / 'annotations' / 'trainval_vcoco.json'),
        'val': (root / 'images' / 'val2014', root / 'annotations' / 'test_vcoco.json')
    }
    CORRECT_MAT_PATH = root / 'annotations' / 'corre_vcoco.npy'

    img_folder, anno_file = PATHS[image_set]

    ANNO_PATHS = {
        "train": (root / "coco/images/train2014/", root / "data/vcoco" / 'vcoco_trainval.json'),
        "val": (root / "coco/images/val2014", root / "data/vcoco" / 'vcoco_test.json'),
        "test": (root / "coco/images/val2014", root / "data/vcoco" / 'vcoco_test.json'),
    }

    _, sub_anno = ANNO_PATHS[image_set]
    dataset = VCOCO(image_set, img_folder, anno_file, sub_anno, transforms=make_vcoco_transforms(image_set),
                    num_queries=args.num_queries)
    if image_set == 'val':
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset
