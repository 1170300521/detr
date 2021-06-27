from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import functional as F
import torchvision.transforms as T
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import re
import PIL
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import pickle
import ast
import logging
from torchvision import transforms
import spacy
import cv2
import random
from util.misc import nested_tensor_from_tensor_list, tlbr2cthw
# from extended_config import cfg as conf


nlp = spacy.load('en_core_web_lg')
words_dict = nlp.vocab.vectors.key2row
# print(len(words_dict))
vocab = nlp.vocab.strings
# nlp = spacy.load('en_core_web_md')


class NewDistributedSampler(DistributedSampler):
    """
    Same as default distributed sampler of pytorch
    Just has another argument for shuffle
    Allows distributed in validation/testing as well
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)


class ImgQuDataset(Dataset):
    """
    Any Grounding dataset.
    Args:
        train_file (string): CSV file with annotations
        The format should be: img_file, bbox, queries
        Can have same img_file on multiple lines
    """

    def __init__(self, cfg, csv_file, ds_name, split_type='train'):
        self.cfg = cfg
        self.ann_file = csv_file
        self.ds_name = ds_name
        self.split_type = split_type

        # self.image_data = pd.read_csv(csv_file)
        self.image_data = self._read_annotations(csv_file)
        # self.image_data = self.image_data.iloc[:200]
        self.img_dir = Path(self.cfg.ds_info[self.ds_name]['img_dir'])
        # self.phrase_len = cfg.phrase_len
        self.phrase_len = cfg.num_queries  # keep the same as detr
        self.item_getter = getattr(self, 'simple_item_getter')
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        return self.item_getter(idx)

    def simple_item_getter(self, idx):
        img_file, annot, q_chosen = self.load_annotations(idx)
        img = PIL.Image.open(img_file).convert('RGB')
        # img_ = np.array(img)
        h, w = img.height, img.width

        q_chosen = q_chosen.strip()
        sents = q_chosen
        qtmp = nlp(str(q_chosen))
        if len(qtmp) == 0:
            # logger.error('Empty string provided')
            raise NotImplementedError
        qlen = min(len(qtmp), self.phrase_len) + 1
        # q_chosen = q_chosen + ' PD'*(self.phrase_len - qlen)
        q_chosen = 'ANS ' + q_chosen
        q_chosen_emb = nlp(q_chosen)
        if not len(q_chosen_emb) == self.phrase_len:
            q_chosen_emb = q_chosen_emb[:self.phrase_len]

        q_chosen_emb_vecs = np.array([q.vector for q in q_chosen_emb])
        # qlen = len(q_chosen_emb_vecs)
        # Annot is in x1y1x2y2 format
        target = np.array(annot)
        center = (target[..., :2] + target[..., 2:])/2
        sizes = target[..., 2:] - target[..., :2]
        cthw = np.concatenate([center, sizes], axis=-1)
        cthw[0::2] /= w
        cthw[1::2] /= h
        img = self.transform(img)

        out = {
            'img': img,
            'idxs': torch.tensor(idx).long(),
            'qvec': torch.from_numpy(q_chosen_emb_vecs).float(),
            'qlens': torch.tensor(qlen),
            'boxes': torch.from_numpy(target).float(),
            'cthw': torch.from_numpy(cthw).float(),
            'labels': torch.tensor([0], dtype=torch.long),
            'orig_size': torch.tensor([h, w]),
            'size': torch.tensor([h, w]),
            'sents': sents,
        }

        return out

    def load_annotations(self, idx):
        annotation_list = self.image_data.iloc[idx]
        img_file, x1, y1, x2, y2, queries = annotation_list
        img_file = self.img_dir / f'{img_file}'
        if isinstance(queries, list):
            query_chosen = np.random.choice(queries)
        else:
            assert isinstance(queries, str)
            query_chosen = queries
        if '_' in query_chosen:
            query_chosen = query_chosen.replace('_', ' ')
        # annotations = np.array([y1, x1, y2, x2])
        annotations = np.array([x1, y1, x2, y2])
        return img_file, annotations, query_chosen

    def _read_annotations(self, trn_file):
        trn_data = pd.read_csv(trn_file)
        trn_data['bbox'] = trn_data.bbox.apply(
            lambda x: ast.literal_eval(x))
        sample = trn_data['query'].iloc[0]
        if sample[0] == '[':
            trn_data['query'] = trn_data['query'].apply(
                lambda x: ast.literal_eval(x))

        trn_data['x1'] = trn_data.bbox.apply(lambda x: x[0])
        trn_data['y1'] = trn_data.bbox.apply(lambda x: x[1])
        trn_data['x2'] = trn_data.bbox.apply(lambda x: x[2])
        trn_data['y2'] = trn_data.bbox.apply(lambda x: x[3])
        if self.ds_name == 'flickr30k':
            trn_data = trn_data.assign(
                image_fpath=trn_data.img_id.apply(lambda x: f'{x}.jpg'))
            trn_df = trn_data[['image_fpath',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        elif self.ds_name == 'refclef':
            trn_df = trn_data[['img_id',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        elif 'flickr30k_c' in self.ds_name:
            trn_data = trn_data.assign(
                image_fpath=trn_data.img_id.apply(lambda x: x))
            trn_df = trn_data[['image_fpath',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        elif self.ds_name in ['refcoco', 'refcoco+', 'refcocog']:
            trn_df = trn_data[['img_id',
                                'x1', 'y1', 'x2', 'y2', 'query']]
        else :
            raise RuntimeError("No dataset named {}".format(self.ds_name))
        return trn_df


class PretrainDataset(Dataset):
    """
    Pretraining dataset.
    Args:
        train_file (string): CSV file with annotations
        The format should be: img_file, queries
        Can NOT have same img_file on multiple lines
    """

    def __init__(self, cfg, csv_file, ds_name, split_type='train'):
        self.cfg = cfg
        self.ann_file = csv_file
        self.ds_name = ds_name
        self.split_type = split_type

        # self.image_data = pd.read_csv(csv_file)
        self.image_data = self._read_annotations(csv_file)
        # self.image_data = self.image_data.iloc[:200]
        self.img_dir = Path(self.cfg.ds_info[self.ds_name]['img_dir'])
        # self.phrase_len = cfg.phrase_len
        self.phrase_len = cfg.num_queries  # keep the same as detr
        self.item_getter = getattr(self, 'simple_item_getter')
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        return self.item_getter(idx)

    def simple_item_getter(self, idx):
        img_file, q_chosen = self.load_annotations(idx)
        is_match = 0
        if random.random() > 0.5:
            _, q_chosen = self.load_annotations(np.random.randint(self.__len__()))
            is_match = 1
        img = PIL.Image.open(img_file).convert('RGB')
        # img_ = np.array(img)
        h, w = img.height, img.width
        q_chosen = q_chosen.strip()
        sents = q_chosen
        q_chosen = 'ANS ' + q_chosen
        # query_words = q_chosen.split()
        mlm_labels = [-1]
        qtmp = nlp(q_chosen)
        query_words = [qw.text for qw in qtmp]
        # print(q_chosen)
        for i in range(1, len(query_words)-1):
            prob = random.random()
            if prob < 0.15:
                prob /= 0.15
                # 80% randomly change token to mask token
                if prob < 0.8:
                    query_words[i] = 'MASK'
                # 10% randomly change token to random token
                elif prob < 0.9: 
                    word = np.random.choice(vocab)
                    # remove compound word
                    word = nlp(str(word))[0].text
                    query_words[i] = word
                token = qtmp[i]
                if token.has_vector:
                    mlm_labels.append(words_dict[token.norm])
                else:
                    mlm_labels.append(-1)
            else:
                mlm_labels.append(-1)
        mlm_labels.append(-1)
        q_chosen = ''
        for words in query_words:
            q_chosen = q_chosen + ' ' + words
        q_chosen_emb = nlp(q_chosen.strip())
        if not len(q_chosen_emb) == self.phrase_len:
            q_chosen_emb = q_chosen_emb[:self.phrase_len]

        q_chosen_emb_vecs = np.array([q.vector for q in q_chosen_emb])
        
        img = self.transform(img)
        out = {
            'img': img,
            'qvec': torch.from_numpy(q_chosen_emb_vecs).float(),
            'text_labels': torch.tensor(mlm_labels).long(),
            'is_match': torch.tensor(is_match).long(),
            'orig_size': torch.tensor([h, w]),
            'size': torch.tensor([h, w]),
            'sents': sents,
        }

        return out

    def load_annotations(self, idx):
        annotation_list = self.image_data.iloc[idx]
        img_file, query_chosen = annotation_list
        img_file = self.img_dir / f'{img_file}'
        if '_' in query_chosen:
            query_chosen = query_chosen.replace('_', ' ')
        return img_file, query_chosen

    def _read_annotations(self, trn_file):
        trn_data = pd.read_csv(trn_file)
        trn_df = trn_data[['img_id', 'query']]
        
        return trn_df


def collater(batch):
    # qlens = torch.Tensor([i['qlens'] for i in batch])
    # max_qlen = int(qlens.max().item())
    # query_vecs = [torch.Tensor(i['query'][:max_qlen]) for i in batch]
    out_dict = {}
    for k in batch[0]:
        if k in ['sents', 'img', 'qvec', 'text_labels']:
            out_dict[k] = [b[k] for b in batch]
        else:
            out_dict[k] = torch.stack([b[k] for b in batch])
    out_dict['img'] = nested_tensor_from_tensor_list(out_dict['img'])
    out_dict['qvec'] = nested_tensor_from_tensor_list(out_dict['qvec'])
    if 'text_labels' in batch[0].keys():
        max_len = max([len(l) for l in out_dict['text_labels']])
        text_labels_pad = torch.zeros(len(batch), max_len).long() - 1
        for i in range(len(batch)):
            labels_i = out_dict['text_labels'][i]
            text_labels_pad[i][0:len(labels_i)] = labels_i
        out_dict['text_labels'] = text_labels_pad
    return out_dict


def get_data(cfg, ds_info):
    # Get which dataset to use
    ds_name = cfg.ds_name
    trn_csv_file = ds_info[ds_name]['trn_csv_file']
    val_csv_file = ds_info[ds_name]['val_csv_file']
    if ds_name == 'pretrain':
        trn_ds = PretrainDataset(cfg=cfg, csv_file=trn_csv_file,
                        ds_name=ds_name, split_type='train')
        val_ds = PretrainDataset(cfg=cfg, csv_file=val_csv_file,
                          ds_name=ds_name, split_type='valid')
        test_ds = val_ds
    else:
        trn_ds = ImgQuDataset(cfg=cfg, csv_file=trn_csv_file,
                            ds_name=ds_name, split_type='train')
        val_ds = ImgQuDataset(cfg=cfg, csv_file=val_csv_file,
                            ds_name=ds_name, split_type='valid')
        if ds_name == 'refcoco' or ds_name == 'refcoco+':
            test_csv_filea = ds_info[ds_name]['test_csv_fileA']
            test_dsa = ImgQuDataset(cfg=cfg, csv_file=test_csv_filea,
                                ds_name=ds_name, split_type='valid')
            #test_dla = get_dataloader(cfg, test_dsa, is_train=False)
            test_csv_fileb = ds_info[ds_name]['test_csv_fileB']
            test_dsb = ImgQuDataset(cfg=cfg, csv_file=test_csv_fileb,
                                ds_name=ds_name, split_type='valid')
            test_ds = {'testA': test_dsa, 'testB': test_dsb}
        else :
            test_csv_file = ds_info[ds_name]['test_csv_file']
            test_ds = ImgQuDataset(cfg=cfg, csv_file=test_csv_file,
                                ds_name=ds_name, split_type='valid')
            test_ds = {'test0': test_ds}

    return {
        "train": trn_ds,
        "val": val_ds,
        "test": test_ds
    }


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return NewDistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


# def collater(batch):
#     # qlens = torch.Tensor([i['qlens'] for i in batch])
#     # max_qlen = int(qlens.max().item())
#     # query_vecs = [torch.Tensor(i['query'][:max_qlen]) for i in batch]
#     out_dict = {}
#     for k in batch[0]:
#         if k in ['sents', 'img', 'qvec']:
#             out_dict[k] = [b[k] for b in batch]
#         else:
#             out_dict[k] = torch.stack([b[k] for b in batch])
#     out_dict['img'] = nested_tensor_from_tensor_list(out_dict['img'])
#     out_dict['qvec'] = nested_tensor_from_tensor_list(out_dict['qvec'])
#     # out_dict['qvec'] = out_dict['qvec'][:, :max_qlen]

#     return out_dict


# def get_data(cfg, ds_info):
#     # Get which dataset to use
#     ds_name = cfg.ds_name

#     # Training file
#     trn_csv_file = ds_info[ds_name]['trn_csv_file']
#     trn_ds = ImgQuDataset(cfg=cfg, csv_file=trn_csv_file,
#                           ds_name=ds_name, split_type='train')
#     #trn_dl = get_dataloader(cfg, trn_ds, is_train=True)

#     # Validation file
#     val_csv_file = ds_info[ds_name]['val_csv_file']
#     val_ds = ImgQuDataset(cfg=cfg, csv_file=val_csv_file,
#                           ds_name=ds_name, split_type='valid')
#     #val_dl = get_dataloader(cfg, val_ds, is_train=False)

#     if ds_name == 'refcoco' or ds_name == 'refcoco+':
#         test_csv_filea = ds_info[ds_name]['test_csv_fileA']
#         test_dsa = ImgQuDataset(cfg=cfg, csv_file=test_csv_filea,
#                             ds_name=ds_name, split_type='valid')
#         #test_dla = get_dataloader(cfg, test_dsa, is_train=False)
#         test_csv_fileb = ds_info[ds_name]['test_csv_fileB']
#         test_dsb = ImgQuDataset(cfg=cfg, csv_file=test_csv_fileb,
#                             ds_name=ds_name, split_type='valid')
#         #test_dlb = get_dataloader(cfg, test_dsb, is_train=False)
#         #test_dl={'testA': test_dla, 'testB': test_dlb}
#         test_ds = {'testA': test_dsa, 'testB': test_dsb}
#     else :
#         test_csv_file = ds_info[ds_name]['test_csv_file']
#         test_ds = ImgQuDataset(cfg=cfg, csv_file=test_csv_file,
#                             ds_name=ds_name, split_type='valid')
#         #test_dl = get_dataloader(cfg, test_ds, is_train=False)
#         #test_dl = {'test0': test_dl}
#         test_ds = {'test0': test_ds}

# #    data = DataWrap(path=cfg.tmp_path, train_dl=trn_dl, valid_dl=val_dl,
# #                    test_dl=test_dl)
#     return {
#         "train": trn_ds,
#         "val": val_ds,
#         "test": test_ds
#     }


if __name__ == '__main__':
    cfg = conf
    data = get_data(cfg, ds_name='refclef')
