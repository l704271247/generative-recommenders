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
    item_features: ItemFeatures
    user_fea_id_base: dict[str, int]
    max_id: int


def get_reco_dataset(
    dataset_name: str,
    max_sequence_length: int,
    chronological: bool,
    positional_sampling_ratio: float = 1.0,
) -> RecoDataset:

    user_fea_id_base = {
        "sex": max_item_id,  # 枚举值2
        "age_group": max_item_id + 2, # 枚举值7
        "occupation": max_item_id + 2 + 7, # 枚举值21
        "zip_code": max_item_id + 2 + 7 + 21, # 枚举值3438,  
    }
    item_fea_id_base = {
        "genres": max_item_id + 2 + 7 + 21 + 3438, # 枚举值63
        "titles": max_item_id + 2 + 7 + 21 + 3438 + 63, # 枚举值16383
        "years": max_item_id + 2 + 7 + 21 + 3438 + 63 + 16383, # 枚举值511
    }
    dp = get_common_preprocessors()[dataset_name]

    items = pd.read_csv(dp.processed_item_csv(), delimiter=",")
    max_jagged_dimension = 16
    expected_max_item_id = dp.expected_max_item_id()
    assert expected_max_item_id is not None
    item_features: ItemFeatures = ItemFeatures(
        max_ind_range=[63, 16383, 511],
        item_fea_id_base=item_fea_id_base,
        num_items=expected_max_item_id + 1,
        max_jagged_dimension=max_jagged_dimension,
        lengths=[
            torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
            torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
            torch.zeros((expected_max_item_id + 1,), dtype=torch.int64),
        ],
        values=[
            torch.zeros(
                (expected_max_item_id + 1, max_jagged_dimension),
                dtype=torch.int64,
            ),
            torch.zeros(
                (expected_max_item_id + 1, max_jagged_dimension),
                dtype=torch.int64,
            ),
            torch.zeros(
                (expected_max_item_id + 1, max_jagged_dimension),
                dtype=torch.int64,
            ),
        ],
    )
    all_item_ids = []
    for df_index, row in items.iterrows():
        # print(f"index {df_index}: {row}")
        movie_id = int(row["movie_id"])
        genres = row["genres"].split("|")
        titles = row["cleaned_title"].split(" ")
        # print(f"{index}: genres{genres}, title{titles}")
        genres_vector = [hash(x) % item_features.max_ind_range[0] + item_fea_id_base['genres'] for x in genres]
        titles_vector = [hash(x) % item_features.max_ind_range[1] + item_fea_id_base['titles'] for x in titles]
        years_vector = [hash(row["year"]) % item_features.max_ind_range[2] + item_fea_id_base['year']]
        item_features.lengths[0][movie_id] = min(
            len(genres_vector), max_jagged_dimension
        )
        item_features.lengths[1][movie_id] = min(
            len(titles_vector), max_jagged_dimension
        )
        item_features.lengths[2][movie_id] = min(
            len(years_vector), max_jagged_dimension
        )
        for f, f_values in enumerate([genres_vector, titles_vector, years_vector]):
            for j in range(min(len(f_values), max_jagged_dimension)):
                item_features.values[f][movie_id][j] = f_values[j]
        all_item_ids.append(movie_id)
    max_item_id = dp.expected_max_item_id()
    for x in all_item_ids:
        assert x > 0, "x in all_item_ids should be positive"

    if dataset_name == "ml-1m":
        train_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=1,
            chronological=chronological,
            sample_ratio=positional_sampling_ratio,
            user_fea_id_base=user_fea_id_base,
            item_features=item_features,
        )
        eval_dataset = DatasetV2(
            ratings_file=dp.output_format_csv(),
            padding_length=max_sequence_length + 1,  # target
            ignore_last_n=0,
            chronological=chronological,
            sample_ratio=1.0,  # do not sample
            user_fea_id_base=user_fea_id_base,
            item_features=item_features,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset_name}")

    print("test item_features:", item_features)

    return RecoDataset(
        max_sequence_length=max_sequence_length,
        num_unique_items=dp.expected_num_unique_items(),  # pyre-ignore [6]
        max_item_id=max_item_id,  # pyre-ignore [6]
        all_item_ids=all_item_ids,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        item_features=item_features,
        user_fea_id_base=user_fea_id_base,
        max_id=max_item_id + 2 + 7 + 21 + 3439 + 63 + 16383 + 511
    )
