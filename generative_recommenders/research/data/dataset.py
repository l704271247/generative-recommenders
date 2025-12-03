# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import csv
import linecache

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from generative_recommenders.research.data.item_features import ItemFeatures


class DatasetV2(torch.utils.data.Dataset):
    """In reverse chronological order."""

    def __init__(
        self,
        ratings_file: str,
        padding_length: int,
        ignore_last_n: int,  # used for creating train/valid/test sets
        shift_id_by: int = 0,
        chronological: bool = False,
        sample_ratio: float = 1.0,
        item_fea_len: int = 0,
        feature_conf = None
    ) -> None:
        """
        Args:
            csv_file (string): Path to the csv file.
        """
        super().__init__()

        self.ratings_frame: pd.DataFrame = pd.read_csv(
            ratings_file,
            sep="\t",
            # iterator=True,
        )
        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache: Dict[int, Dict[str, torch.Tensor]] = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio
        self._item_fea_len: int = item_fea_len
        self._feature_conf = feature_conf

    def __len__(self) -> int:
        return len(self.ratings_frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self._cache.keys():
            return self._cache[idx]
        data = self.ratings_frame.iloc[idx]
        sample = self.load_item(data)
        self._cache[idx] = sample
        return sample

    def str2dtype(self, x):
        if x.lower() == 'int':
            return int
        elif x.lower() == 'float':
            return float
        else:
            return str

    def conf2torchdtype(self, conf):
        if conf.get('need_code', False):
            return torch.int64
        elif conf.get('dtype', 'int').lower() == 'int':
            return torch.int64
        elif conf.get('dtype', 'int').lower() == 'float':
            return torch.float32
        else:
            return torch.int64

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        ufea = {}
        ufea_conf = self._feature_conf['user_fea']
        for fea in ufea_conf:
            ufea[fea] = torch.tensor(data[fea], dtype=self.conf2torchdtype(ufea_conf[fea])).view([-1, ufea_conf[fea].get('fea_len', 1)])

        ifea = {}
        ifea_conf = self._feature_conf['item_fea']
        for fea in ifea_conf:
            ifea[fea] = data[fea]

        def eval_as_list(x: str, ignore_last_n: int, fea_len: int=1) -> List[List[int]]:
            y = eval(x)
            y_list = [y] if type(y) == int else list(y)
            y_list = [y_list[i:i+fea_len] for i in range(0, len(y_list), fea_len)]
            if ignore_last_n > 0:
                # for training data creation
                y_list = y_list[:-ignore_last_n]
            return y_list

        def eval_int_list(
            x: str,
            fea_len: int,
            ignore_last_n: int,
            shift_id_by: int,
            sampling_kept_mask: Optional[List[bool]],
        ) -> Tuple[List[List[int]], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n, fea_len=fea_len)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y.reverse()
            y_len = len(y)
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        if self._sample_ratio < 1.0:
            raw_length = len(eval_as_list(data['sid'], self._ignore_last_n))
            sampling_kept_mask = (
                torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio
            ).tolist()
        else:
            sampling_kept_mask = None

        sampled_ifea = {}
        ifea_lens = {}
        for k,v in ifea.items():
            seq, seq_len = eval_int_list(
                x=v, 
                fea_len=ifea_conf[k].get('fea_len', 1), 
                ignore_last_n=self._ignore_last_n,
                shift_id_by=self._shift_id_by,
                sampling_kept_mask=sampling_kept_mask,
            )
            sampled_ifea[k] = seq
            ifea_lens[k] = seq_len

        for k,v in ifea_lens.items():
            assert v == ifea_lens['sid'], f"feature {k} len {v} differs from sid len {ifea_lens['sid']}."

        def _truncate_or_pad_seq(
            y: List[List[int]], target_len: int, fea_len: int, chronological: bool
        ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [[0] * fea_len] * (target_len - y_len)
            else:
                if not chronological:
                    y = y[:target_len]
                else:
                    y = y[-target_len:]
            assert len(y) == (target_len)
            y = [item for sublist in y for item in sublist]
            return y


        max_seq_len = self._padding_length - 1
        history_length = min(ifea_lens['sid'] - 1, max_seq_len)
        historical_ifea = {}
        target_ifea = {}
        for k,v in sampled_ifea.items():
            target_ifea[k] = v[0]
            tmp_historical_ifea = v[:0:-1] if self._chronological else v[1:]
            historical_ifea[k] = _truncate_or_pad_seq(
            tmp_historical_ifea,
            max_seq_len,
            ifea_conf[k].get('fea_len', 1),
            self._chronological,
        )
        # moved to features.py
        # if self._chronological:
        #     historical_ids.append(0)
        #     historical_ratings.append(0)
        #     historical_timestamps.append(0)
        # print(historical_ids, historical_ratings, historical_timestamps, target_ids, target_ratings, target_timestamps)
        ret = {}
        ret.update(ufea)
        for k,v in historical_ifea.items():
            ret["historical_" + k] = torch.tensor(v, dtype=self.conf2torchdtype(ifea_conf[k])).view(-1, ifea_conf[k].get('fea_len', 1))
        for k,v in target_ifea.items():
            ret["target_" + k] = torch.tensor(v, dtype=self.conf2torchdtype(ifea_conf[k])).view(-1, ifea_conf[k].get('fea_len', 1))
        ret['historical_lengths'] = history_length

        return ret
