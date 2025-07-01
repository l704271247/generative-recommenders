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

import abc
import logging
import os
import sys
import tarfile
from typing import Dict, Optional, Union

from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np

import pandas as pd


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


class DataProcessor:
    """
    This preprocessor does not remap item_ids. This is intended so that we can easily join other
    side-information based on item_ids later.
    """

    def __init__(
        self,
        prefix: str,
        expected_num_unique_items: Optional[int],
        expected_max_item_id: Optional[int],
    ) -> None:
        self._prefix: str = prefix
        self._expected_num_unique_items = expected_num_unique_items
        self._expected_max_item_id = expected_max_item_id

    @abc.abstractmethod
    def expected_num_unique_items(self) -> Optional[int]:
        return self._expected_num_unique_items

    @abc.abstractmethod
    def expected_max_item_id(self) -> Optional[int]:
        return self._expected_max_item_id

    @abc.abstractmethod
    def processed_item_csv(self) -> str:
        pass

    def output_format_csv(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format.csv"

    def to_seq_data(
        self,
        ratings_data: pd.DataFrame,
        user_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if user_data is not None:
            ratings_data_transformed = ratings_data.join(
                user_data.set_index("user_id"), on="user_id"
            )
        else:
            ratings_data_transformed = ratings_data
        ratings_data_transformed.item_ids = ratings_data_transformed.item_ids.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.timestamps = ratings_data_transformed.timestamps.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.hash_genres = ratings_data_transformed.hash_genres.apply(
            lambda x: ",".join([",".join([str(i) for i in v]) for v in x])
        )
        ratings_data_transformed.hash_title = ratings_data_transformed.hash_title.apply(
            lambda x: ",".join([",".join([str(i) for i in v]) for v in x])
        )
        ratings_data_transformed.hash_year = ratings_data_transformed.hash_year.apply(
            lambda x: ",".join([str(v) for v in x])
        )
        ratings_data_transformed.rename(
            columns={
                "item_ids": "sequence_item_ids",
                "ratings": "sequence_ratings",
                "timestamps": "sequence_timestamps",
                "hash_genres": "sequence_hash_genres",
                "hash_title": "sequence_hash_title",
                "hash_year": "sequence_hash_year",
            },
            inplace=True,
        )
        return ratings_data_transformed

    def file_exists(self, name: str) -> bool:
        return os.path.isfile("%s/%s" % (os.getcwd(), name))


class MovielensDataProcessor(DataProcessor):
    def __init__(
        self,
        download_path: str,
        saved_name: str,
        prefix: str,
        convert_timestamp: bool,
        expected_num_unique_items: Optional[int] = None,
        expected_max_item_id: Optional[int] = None,
    ) -> None:
        super().__init__(prefix, expected_num_unique_items, expected_max_item_id)
        self._download_path = download_path
        self._saved_name = saved_name
        self._convert_timestamp: bool = convert_timestamp
        self._max_jagged_dimension = 16
        self._max_ind_range=[63, 16383, 511]

    def download(self) -> None:
        if not self.file_exists(self._saved_name):
            urlretrieve(self._download_path, self._saved_name)
        if self._saved_name[-4:] == ".zip":
            ZipFile(self._saved_name, "r").extractall(path="tmp/")
        else:
            with tarfile.open(self._saved_name, "r:*") as tar_ref:
                tar_ref.extractall("tmp/")

    def processed_item_csv(self) -> str:
        return f"tmp/processed/{self._prefix}/movies.csv"

    def sasrec_format_csv_by_user_train(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format_by_user_train.csv"

    def sasrec_format_csv_by_user_test(self) -> str:
        return f"tmp/{self._prefix}/sasrec_format_by_user_test.csv"
    
    def max_jagged_dimension(self) -> int:
        return self._max_jagged_dimension

    def preprocess_rating(self) -> int:
        self.download()

        users = pd.read_csv(
            f"tmp/{self._prefix}/users.dat",
            sep="::",
            names=["user_id", "sex", "age_group", "occupation", "zip_code"],
        )
        ratings = pd.read_csv(
            f"tmp/{self._prefix}/ratings.dat",
            sep="::",
            names=["user_id", "movie_id", "rating", "unix_timestamp"],
        )
        movies = pd.read_csv(
            f"tmp/{self._prefix}/movies.dat",
            sep="::",
            names=["movie_id", "title", "genres"],
            encoding="iso-8859-1",
        )

        if movies is not None:
            # ML-1M and ML-20M only
            movies["year"] = movies["title"].apply(lambda x: x[-5:-1])
            movies["cleaned_title"] = movies["title"].apply(lambda x: x[:-7])
            # movies.year = pd.Categorical(movies.year)
            # movies["year"] = movies.year.cat.codes

        if users is not None:
            ## Users (ml-1m only)
            users.sex = pd.Categorical(users.sex)
            users["sex"] = users.sex.cat.codes

            users.age_group = pd.Categorical(users.age_group)
            users["age_group"] = users.age_group.cat.codes

            users.occupation = pd.Categorical(users.occupation)
            users["occupation"] = users.occupation.cat.codes

            users.zip_code = pd.Categorical(users.zip_code)
            users["zip_code"] = users.zip_code.cat.codes
        
        
        # Normalize movie ids to speed up training
        print(
            f"{self._prefix} #item before normalize: {len(set(ratings['movie_id'].values))}"
        )
        print(
            f"{self._prefix} max item id before normalize: {max(set(ratings['movie_id'].values))}"
        )

        if self._convert_timestamp:
            ratings["unix_timestamp"] = pd.to_datetime(
                ratings["unix_timestamp"], unit="s"
            )

        
        processed_movies = {"movie_id":[], "title":[], "genres":[], "year":[], "cleaned_title":[], 
                            "hash_genres":[], "hash_year":[], "hash_title":[]}
        for df_index, row in movies.iterrows():
            movie_id = row["movie_id"]
            titles = row["title"].split(" ")
            genres = row["genres"].split("|")
            year = row["year"]
            cleaned_title = row["cleaned_title"].split(" ")
            
            genres_vector = [(hash(x) % self._max_ind_range[0]) + 1 for x in genres]
            titles_vector = [(hash(x) % self._max_ind_range[1]) + 1 for x in cleaned_title]
            years_vector = (hash(year) % self._max_ind_range[2]) + 1

            genres_vector = (genres_vector + [0] * self._max_jagged_dimension)[:self._max_jagged_dimension]
            titles_vector = (titles_vector + [0] * self._max_jagged_dimension)[:self._max_jagged_dimension]

            processed_movies["movie_id"].append(movie_id)
            processed_movies["title"].append(titles)
            processed_movies["genres"].append(genres)
            processed_movies["year"].append(year)
            processed_movies["cleaned_title"].append(cleaned_title)
            processed_movies["hash_genres"].append(genres_vector)
            processed_movies["hash_year"].append(years_vector)
            processed_movies["hash_title"].append(titles_vector)

        processed_movies = pd.DataFrame(processed_movies)
        ratings_join = ratings.join(
            processed_movies.set_index("movie_id"), on="movie_id"
        )
        # Save primary csv's
        if not os.path.exists(f"tmp/processed/{self._prefix}"):
            os.makedirs(f"tmp/processed/{self._prefix}")
        if users is not None:
            users.to_csv(f"tmp/processed/{self._prefix}/users.csv", index=False)
        if processed_movies is not None:
            processed_movies.to_csv(f"tmp/processed/{self._prefix}/movies.csv", index=False)
        ratings_join.to_csv(f"tmp/processed/{self._prefix}/ratings.csv", index=False)

        num_unique_users = len(set(ratings["user_id"].values))
        num_unique_items = len(set(ratings["movie_id"].values))

        # SASRec version
        ratings_group = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")
        seq_ratings_data = pd.DataFrame(
            data={
                "user_id": list(ratings_group.groups.keys()),
                "item_ids": list(ratings_group.movie_id.apply(list)),
                "ratings": list(ratings_group.rating.apply(list)),
                "timestamps": list(ratings_group.unix_timestamp.apply(list)),
                "hash_genres": list(ratings_group.hash_genres.apply(list)),
                "hash_year": list(ratings_group.hash_year.apply(list)),
                "hash_title": list(ratings_group.hash_title.apply(list)),
            }
        )

        result = pd.DataFrame([[]])
        for col in ["item_ids"]:
            result[col + "_mean"] = seq_ratings_data[col].apply(len).mean()
            result[col + "_min"] = seq_ratings_data[col].apply(len).min()
            result[col + "_max"] = seq_ratings_data[col].apply(len).max()
        print(self._prefix)
        print(result)

        seq_ratings_data = self.to_seq_data(seq_ratings_data, users)
        seq_ratings_data.sample(frac=1).reset_index().to_csv(
            self.output_format_csv(), index=False, sep=","
        )

        # Split by user ids (not tested yet)
        user_id_split = int(num_unique_users * 0.9)
        seq_ratings_data_train = seq_ratings_data[
            seq_ratings_data["user_id"] <= user_id_split
        ]
        seq_ratings_data_train.sample(frac=1).reset_index().to_csv(
            self.sasrec_format_csv_by_user_train(),
            index=False,
            sep=",",
        )
        seq_ratings_data_test = seq_ratings_data[
            seq_ratings_data["user_id"] > user_id_split
        ]
        seq_ratings_data_test.sample(frac=1).reset_index().to_csv(
            self.sasrec_format_csv_by_user_test(), index=False, sep=","
        )
        print(
            f"{self._prefix}: train num user: {len(set(seq_ratings_data_train['user_id'].values))}"
        )
        print(
            f"{self._prefix}: test num user: {len(set(seq_ratings_data_test['user_id'].values))}"
        )

        # print(seq_ratings_data)
        if self.expected_num_unique_items() is not None:
            assert (
                self.expected_num_unique_items() == num_unique_items
            ), f"Expected items: {self.expected_num_unique_items()}, got: {num_unique_items}"

        return num_unique_items

def get_common_preprocessors() -> (
    Dict[
        str,
        Union[
             MovielensDataProcessor
        ],
    ]
):
    ml_1m_dp = MovielensDataProcessor(  # pyre-ignore [45]
        "http://files.grouplens.org/datasets/movielens/ml-1m.zip",
        "tmp/movielens1m.zip",
        prefix="ml-1m",
        convert_timestamp=False,
        expected_num_unique_items=3706,
        expected_max_item_id=3952,
    )
    return {
        "ml-1m": ml_1m_dp
    }
