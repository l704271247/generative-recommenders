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

from dataclasses import dataclass
from typing import List

from generative_recommenders.research.data import item_features
import pandas as pd

import torch

from generative_recommenders.research.data.dataset import DatasetV2 # , MultiFileDatasetV2
from generative_recommenders.research.data.item_features import ItemFeatures
from generative_recommenders.research.data.preprocessor import get_common_preprocessors


@dataclass
class RecoDataset:
    max_sequence_length: int
    num_unique_items: int
    max_item_id: int
    all_item_ids: List[int]
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset
    max_id: int


def get_reco_dataset(
    dataset_name: str,
    max_sequence_length: int,
    chronological: bool,
    positional_sampling_ratio: float = 1.0,
) -> RecoDataset:
    dp = get_common_preprocessors()[dataset_name]
    max_item_id = dp.expected_max_item_id()
    items = pd.read_csv(dp.processed_item_csv(), delimiter=",")
    all_item_ids = []
    for df_index, row in items.iterrows():
        # print(f"index {df_index}: {row}")
        movie_id = int(row["movie_id"])
        all_item_ids.append(movie_id)
    for x in all_item_ids:
        assert x > 0, "x in all_item_ids should be positive"

    if dataset_name == "ml-1m":
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
            sample_ratio=positional_sampling_ratio,
            item_fea_len = dp.max_jagged_dimension()
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
            sample_ratio=1.0,
            item_fea_len = dp.max_jagged_dimension()
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=dp.expected_num_unique_items(),  # pyre-ignore [6]
        max_item_id=max_item_id,  # pyre-ignore [6]
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_id=max_item_id
    )
