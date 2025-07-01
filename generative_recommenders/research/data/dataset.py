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
        self._item_fea_len: int = item_fea_len

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
        sex = data.sex.astype(int)
        age_group = data.age_group.astype(int)
        occupation = data.occupation.astype(int)
        zip_code = data.zip_code.astype(int)

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
            raw_length = len(eval_as_list(data.sequence_item_ids, self._ignore_last_n))
            sampling_kept_mask = (
                torch.rand((raw_length,), dtype=torch.float32) < self._sample_ratio
            ).tolist()
        else:
            sampling_kept_mask = None

        movie_history, movie_history_len = eval_int_list(
            data.sequence_item_ids,
            1,
            self._ignore_last_n,
            shift_id_by=self._shift_id_by,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_history_ratings, ratings_len = eval_int_list(
            data.sequence_ratings,
            1,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_timestamps, timestamps_len = eval_int_list(
            data.sequence_timestamps,
            1,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_genres, genres_len = eval_int_list(
            data.sequence_hash_genres,
            self._item_fea_len,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_title, title_len = eval_int_list(
            data.sequence_hash_title,
            self._item_fea_len,
            self._ignore_last_n,
            0,
            sampling_kept_mask=sampling_kept_mask,
        )
        movie_year, year_len = eval_int_list(
            data.sequence_hash_year,
            1,
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
        assert (
            movie_history_len  == genres_len
        ), f"history len {movie_history_len} differs from genres len {genres_len}."
        assert (
            movie_history_len  == title_len
        ), f"history len {movie_history_len} differs from title len {title_len}."
        assert (
            movie_history_len == year_len
        ), f"history len {movie_history_len} differs from year len {year_len}."

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

        historical_ids = movie_history[1:]
        historical_ratings = movie_history_ratings[1:]
        historical_timestamps = movie_timestamps[1:]
        historical_genres = movie_genres[1:]
        historical_title = movie_title[1:]
        historical_year = movie_year[1:]
        target_ids = movie_history[0]
        target_ratings = movie_history_ratings[0]
        target_timestamps = movie_timestamps[0]
        target_genres = movie_genres[0]
        target_title = movie_title[0]
        target_year = movie_year[0]
        if self._chronological:
            historical_ids.reverse()
            historical_ratings.reverse()
            historical_timestamps.reverse()
            historical_genres.reverse()
            historical_title.reverse()
            historical_year.reverse() 

        max_seq_len = self._padding_length - 1
        history_length = min(len(historical_ids), max_seq_len)
        historical_ids = _truncate_or_pad_seq(
            historical_ids,
            max_seq_len,
            self._item_fea_len,
            self._chronological,
        )
        historical_ratings = _truncate_or_pad_seq(
            historical_ratings,
            max_seq_len,
            self._item_fea_len,
            self._chronological,
        )
        historical_timestamps = _truncate_or_pad_seq(
            historical_timestamps,
            max_seq_len,
            self._item_fea_len,
            self._chronological,
        )
        historical_genres = _truncate_or_pad_seq(
            historical_genres,
            max_seq_len,
            self._item_fea_len,
            self._chronological,
        )
        historical_title = _truncate_or_pad_seq(
            historical_title,
            max_seq_len,
            self._item_fea_len,
            self._chronological,
        )
        historical_year = _truncate_or_pad_seq(
            historical_year,
            max_seq_len,
            self._item_fea_len,
            self._chronological,
        )
        # moved to features.py
        # if self._chronological:
        #     historical_ids.append(0)
        #     historical_ratings.append(0)
        #     historical_timestamps.append(0)
        # print(historical_ids, historical_ratings, historical_timestamps, target_ids, target_ratings, target_timestamps)
    
        ret = {
            "user_id": user_id,
            "sex": sex,
            "age_group": age_group,
            "occupation": occupation,
            "zip_code": zip_code,
            "historical_ids": torch.tensor(historical_ids, dtype=torch.int64),
            "historical_ratings": torch.tensor(historical_ratings, dtype=torch.int64),
            "historical_timestamps": torch.tensor(historical_timestamps, dtype=torch.int64),
            "historical_genres": torch.tensor(historical_genres, dtype=torch.int64).view(-1, self._item_fea_len),
            "historical_title": torch.tensor(historical_title, dtype=torch.int64).view(-1, self._item_fea_len), # view(-1, self._item_fea_len) for mult
            "historical_year": torch.tensor(historical_year, dtype=torch.int64),
            "history_lengths": history_length,
            "target_ids": target_ids,
            "target_ratings": target_ratings,
            "target_timestamps": target_timestamps,
            "target_genres": torch.tensor(target_genres, dtype=torch.int64),
            "target_title": torch.tensor(target_title, dtype=torch.int64),
            "target_year": torch.tensor(target_year, dtype=torch.int64),
        }
        return ret
