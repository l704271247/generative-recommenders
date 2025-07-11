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

from typing import Dict, NamedTuple, Optional, Tuple

import torch


class SequentialFeatures(NamedTuple):
    # (B,) x int64. Requires past_lengths[i] > 0 \forall i.
    past_lengths: torch.Tensor
    # (B, N,) x int64. 0 denotes valid ids.
    past_ids: torch.Tensor
    # (B, N, D) x float.
    past_embeddings: Optional[torch.Tensor]
    # Implementation-specific payloads.
    # e.g., past timestamps, past event_types (e.g., clicks, likes), etc.
    past_payloads: Dict[str, torch.Tensor]


def movielens_seq_features_from_row(
    row: Dict[str, torch.Tensor],
    device: int,
    max_output_length: int,
) -> Tuple[SequentialFeatures, torch.Tensor, torch.Tensor]:
    historical_lengths = row["history_lengths"].to(device)  # [B]
    historical_ids = row["historical_ids"].to(device)  # [B, N]
    historical_ratings = row["historical_ratings"].to(device)  # [B, N]
    historical_timestamps = row["historical_timestamps"].to(device)  # [B, N]
    historical_genres = row["historical_genres"].to(device)  # [B, N, 16]
    historical_title = row["historical_title"].to(device)  # [B, N, 16]
    historical_year = row["historical_year"].to(device)  # [B, N]
    
    target_ids = row["target_ids"].to(device).unsqueeze(1)  # [B, 1]
    target_ratings = row["target_ratings"].to(device).unsqueeze(1)  # [B, 1]
    target_timestamps = row["target_timestamps"].to(device).unsqueeze(1)  # [B, 1]
    target_genres = row["target_genres"].to(device).unsqueeze(1)  # [B, 1, 16]
    target_title = row["target_title"].to(device).unsqueeze(1)  # [B, 1, 16]
    target_year = row["target_year"].to(device).unsqueeze(1)  # [B, 1]

    sex = row["sex"].to(device).unsqueeze(1)  # [B, 1]
    age = row["age"].to(device).unsqueeze(1)  # [B, 1]
    occupation = row["occupation"].to(device).unsqueeze(1)  # [B, 1]
    zip_code = row["zip_code"].to(device).unsqueeze(1)  # [B, 1]

    if max_output_length > 0:
        B = historical_lengths.size(0)
        historical_ids = torch.cat(
            [
                historical_ids,
                torch.zeros(
                    (B, max_output_length), dtype=historical_ids.dtype, device=device
                ),
            ],
            dim=1,
        )
        historical_ratings = torch.cat(
            [
                historical_ratings,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_ratings.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_timestamps = torch.cat(
            [
                historical_timestamps,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_timestamps.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_timestamps.scatter_(
            dim=1,
            index=historical_lengths.view(-1, 1),
            src=target_timestamps.view(-1, 1),
        )
        historical_genres = torch.cat(
            [
                historical_genres,
                torch.zeros(
                    (B, max_output_length, historical_genres.size(2)),
                    dtype=historical_genres.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_title = torch.cat(
            [
                historical_genres,
                torch.zeros(
                    (B, max_output_length, historical_title.size(2)),
                    dtype=historical_title.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        historical_year = torch.cat(
            [
                historical_year,
                torch.zeros(
                    (B, max_output_length),
                    dtype=historical_year.dtype,
                    device=device,
                ),
            ],
            dim=1,
        )
        # print(f"historical_ids.size()={historical_ids.size()}, historical_timestamps.size()={historical_timestamps.size()}")
    features = SequentialFeatures(
        past_lengths=historical_lengths,
        past_ids=historical_ids,
        past_embeddings=None,
        past_payloads={
            "timestamps": historical_timestamps,
            "ratings": historical_ratings,
            "genres": historical_genres,
            "title": historical_title,
            "year": historical_year,
            "sex": sex,
            "age": age,
            "occupation": occupation,
            "zip_code": zip_code,
            "target_genres": target_genres,
            "target_title": target_title,
            "target_year": target_year,
        },
    )
    return features, target_ids, target_ratings
