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
        user_fea_id_base: Optional[Dict[str, int]] = None,
        item_features: Optional[ItemFeatures] = None,
    ) -> None:
        """
        Args:
            csv_file (string): Path to the csv file.
        """
        super().__init__()

        self.ratings_frame: pd.DataFrame = pd.read_csv(
            ratings_file,
            delimiter=",",
            # iterator=True,
        )
        self._padding_length: int = padding_length
        self._ignore_last_n: int = ignore_last_n
        self._cache: Dict[int, Dict[str, torch.Tensor]] = dict()
        self._shift_id_by: int = shift_id_by
        self._chronological: bool = chronological
        self._sample_ratio: float = sample_ratio
        self._user_fea_id_base = user_fea_id_base
        self._item_features = item_features

    def __len__(self) -> int:
        return len(self.ratings_frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self._cache.keys():
            return self._cache[idx]
        data = self.ratings_frame.iloc[idx]
        sample = self.load_item(data)
        self._cache[idx] = sample
        return sample

    def load_item(self, data) -> Dict[str, torch.Tensor]:
        user_id = data.user_id.astype(int)
        sex = data.sex.astype(int) + self._user_fea_id_base['sex']
        age_group = data.age_group.astype(int) + self._user_fea_id_base['age_group']
        occupation = data.occupation.astype(int) + self._user_fea_id_base['occupation']
        zip_code = data.zip_code.astype(int) + self._user_fea_id_base['zip_code']

        def eval_as_list(x: str, ignore_last_n: int) -> List[int]:
            y = eval(x)
            y_list = [y] if type(y) == int else list(y)
            if ignore_last_n > 0:
                # for training data creation
                y_list = y_list[:-ignore_last_n]
            return y_list

        def eval_int_list(
            x: str,
            target_len: int,
            ignore_last_n: int,
            shift_id_by: int,
            sampling_kept_mask: Optional[List[bool]],
        ) -> Tuple[List[int], int]:
            y = eval_as_list(x, ignore_last_n=ignore_last_n)
            if sampling_kept_mask is not None:
                y = [x for x, kept in zip(y, sampling_kept_mask) if kept]
            y_len = len(y)
            y.reverse()
            if shift_id_by > 0:
                y = [x + shift_id_by for x in y]
            return y, y_len

        if self._sample_ratio < 1.0:
            raw_length = len(eval_as_list(data.sequence_item_ids, self._ignore_last_n))
            sampling_kept_mask = (
                torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio
            ).tolist()
        else:
            sampling_kept_mask = None

        movie_history, movie_history_len = eval_int_list(
            data.sequence_item_ids,
            self._padding_length,
            self._ignore_last_n,
            shift_id_by=self._shift_id_by,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_history_ratings, ratings_len = eval_int_list(
            data.sequence_ratings,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_timestamps, timestamps_len = eval_int_list(
            data.sequence_timestamps,
            self._padding_length,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        assert (
            movie_history_len == timestamps_len
        ), f"history len {movie_history_len} differs from timestamp len {timestamps_len}."
        assert (
            movie_history_len == ratings_len
        ), f"history len {movie_history_len} differs from ratings len {ratings_len}."

        def _truncate_or_pad_seq(
            y: List[int], target_len: int, chronological: bool
        ) -> List[int]:
            y_len = len(y)
            if y_len < target_len:
                y = y + [0] * (target_len - y_len)
            else:
                if not chronological:
                    y = y[:target_len]
                else:
                    y = y[-target_len:]
            assert len(y) == target_len
            return y

        historical_ids = movie_history[1:]
        historical_ratings = movie_history_ratings[1:]
        historical_timestamps = movie_timestamps[1:]
        target_ids = movie_history[0]
        target_ratings = movie_history_ratings[0]
        target_timestamps = movie_timestamps[0]
        if self._chronological:
            historical_ids.reverse()
            historical_ratings.reverse()
            historical_timestamps.reverse()    

        max_seq_len = self._padding_length - 1
        history_length = min(len(historical_ids), max_seq_len-4) + 4
        historical_ids = _truncate_or_pad_seq(
            historical_ids,
            max_seq_len-4,
            self._chronological,
        )
        historical_ids = [sex, age_group, occupation, zip_code] + historical_ids
        historical_ratings = _truncate_or_pad_seq(
            historical_ratings,
            max_seq_len-4,
            self._chronological,
        )
        historical_ratings =  [0] * 4 + historical_ratings

        historical_timestamps = _truncate_or_pad_seq(
            historical_timestamps,
            max_seq_len-4,
            self._chronological,
        )
        historical_timestamps = [0] * 4 + historical_timestamps
        # moved to features.py
        # if self._chronological:
        #     historical_ids.append(0)
        #     historical_ratings.append(0)
        #     historical_timestamps.append(0)
        # print(historical_ids, historical_ratings, historical_timestamps, target_ids, target_ratings, target_timestamps)
        
        # process item features
        history_item_fea_ids = []
        target_item_fea_ids = torch.zeros(
            self._item_features.max_jagged_dimension * 3,
            dtype=torch.int64,
        )
        if self._item_features is not None:
            for item in historical_ids:
                if item < len(self._item_features.values[0]):
                    item_fea = torch.cat(
                        (
                            self._item_features.values[0][item,...],
                            self._item_features.values[1][item,...],
                            self._item_features.values[2][item,...],
                        ),
                        dim=0,
                    )
                else:
                    item_fea = torch.zeros(
                        self._item_features.max_jagged_dimension * 3,
                        dtype=torch.int64,
                    )
                history_item_fea_ids.append(item_fea)
            
            if target_ids < len(self._item_features.values[0]):
                target_item_fea_ids = torch.cat(
                    (
                        self._item_features.values[0][target_ids,...],
                        self._item_features.values[1][target_ids,...],
                        self._item_features.values[2][target_ids,...],
                    ),
                    dim=0,
                )

        history_item_fea_ids = torch.stack(history_item_fea_ids, dim=0)
    
        ret = {
            "user_id": user_id,
            "sex": sex,
            "age_group": age_group,
            "occupation": occupation,
            "zip_code": zip_code,
            "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
            "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
            "historical_timestamps": torch.tensor(
                historical_timestamps, dtype=torch.int64
            ),
            "history_item_fea_ids": history_item_fea_ids,
            "history_lengths": history_length,
            "target_ids": target_ids,
            "target_ratings": target_ratings,
            "target_timestamps": target_timestamps,
            "target_item_fea_ids": target_item_fea_ids,
        }
        return ret
