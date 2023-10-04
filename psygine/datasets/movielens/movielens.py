# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/10/02
# License: MIT License
"""MovieLens Datasets.
"""
import os.path as op
from pathlib import Path
from collections import OrderedDict

from .base import BaseMovieLensDataset
from ..utils.network import get_data_path

import zipfile
import pandas as pd

ML_URLS = {
    "ml-100k": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-10m": "https://files.grouplens.org/datasets/movielens/ml-10m.zip",
    "ml-20m": "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
}


class MovieLensDataset_100K(BaseMovieLensDataset):
    """MovieLensDataset 100K.

    Available tables include: data, item, and user.

    """

    def __init__(self):
        super().__init__(
            "ml-100k",
            ["data", "item", "user"],
        )

    def data_path(self, local_path=None, force_update=False, proxies=None):
        url = None
        try:
            url = ML_URLS[self.uid]
        except KeyError as error:
            raise NotImplementedError(
                "Current dataset {:s} is not implemented.".format(self.uid)
            ) from error

        file_dest = get_data_path(
            url,
            "movielens",
            path=local_path,
            proxies=proxies,
            force_update=force_update,
        )

        parent_dir = op.join(Path(file_dest).parent, self.uid)
        if not op.exists(op.join(parent_dir, "u.data")):
            with zipfile.ZipFile(file_dest, "r") as archive:
                archive.extractall(path=Path(file_dest).parent)

        files = OrderedDict()
        for table in self.tables:
            files[table] = op.join(parent_dir, "u.{:s}".format(table))
        dests = [files]
        return dests

    def get_data(self, tables=None):
        dests = self.data_path()

        column_names = {
            "data": ["user id", "item id", "rating", "timestamp"],
            "item": [
                "movie id",
                "movie title",
                "release date",
                "video release date",
                "imdb url",
                "unknown",
                "action",
                "adventure",
                "animation",
                "children's",
                "comedy",
                "crime",
                "documentary",
                "drama",
                "fantasy",
                "film-noir",
                "horror",
                "musical",
                "mystery",
                "romance",
                "sci-fi",
                "thriller",
                "war",
                "western",
            ],
            "user": ["user id", "age", "gender", "occupation", "zip code"],
        }

        if tables is None:
            tables = self.tables

        rawdata = OrderedDict()
        for table in tables:
            if table == "data":
                rawdata[table] = pd.read_csv(
                    dests[0][table],
                    sep="\t",
                    names=column_names[table],
                    encoding="utf_8",
                )
            elif table == "item":
                rawdata[table] = pd.read_csv(
                    dests[0][table],
                    sep="|",
                    names=column_names[table],
                    encoding="latin_1",
                )
            else:
                rawdata[table] = pd.read_csv(
                    dests[0][table],
                    sep="|",
                    names=column_names[table],
                    encoding="utf_8",
                )
        return rawdata


class MovieLensDataset_1M(BaseMovieLensDataset):
    """MovieLensDataset 1M.

    Available tables include: ratings, movies, and users.

    These files contain 1,000,209 anonymous ratings of approximately 3,900 moviesmade
    by 6,040 MovieLens users who joined MovieLens in 2000.

    RATINGS FILE DESCRIPTION
    ================================================================================

    All ratings are contained in the file "ratings.dat" and are in the
    following format:

    UserID::MovieID::Rating::Timestamp

    - UserIDs range between 1 and 6040
    - MovieIDs range between 1 and 3952
    - Ratings are made on a 5-star scale (whole-star ratings only)
    - Timestamp is represented in seconds since the epoch as returned by time(2)
    - Each user has at least 20 ratings

    USERS FILE DESCRIPTION
    ================================================================================

    User information is in the file "users.dat" and is in the following
    format:

    UserID::Gender::Age::Occupation::Zip-code

    All demographic information is provided voluntarily by the users and is
    not checked for accuracy.  Only users who have provided some demographic
    information are included in this data set.

    - Gender is denoted by a "M" for male and "F" for female
    - Age is chosen from the following ranges:

            *  1:  "Under 18"
            * 18:  "18-24"
            * 25:  "25-34"
            * 35:  "35-44"
            * 45:  "45-49"
            * 50:  "50-55"
            * 56:  "56+"

    - Occupation is chosen from the following choices:

            *  0:  "other" or not specified
            *  1:  "academic/educator"
            *  2:  "artist"
            *  3:  "clerical/admin"
            *  4:  "college/grad student"
            *  5:  "customer service"
            *  6:  "doctor/health care"
            *  7:  "executive/managerial"
            *  8:  "farmer"
            *  9:  "homemaker"
            * 10:  "K-12 student"
            * 11:  "lawyer"
            * 12:  "programmer"
            * 13:  "retired"
            * 14:  "sales/marketing"
            * 15:  "scientist"
            * 16:  "self-employed"
            * 17:  "technician/engineer"
            * 18:  "tradesman/craftsman"
            * 19:  "unemployed"
            * 20:  "writer"

    MOVIES FILE DESCRIPTION
    ================================================================================

    Movie information is in the file "movies.dat" and is in the following
    format:

    MovieID::Title::Genres

    - Titles are identical to titles provided by the IMDB (including
    year of release)
    - Genres are pipe-separated and are selected from the following genres:

            * Action
            * Adventure
            * Animation
            * Children's
            * Comedy
            * Crime
            * Documentary
            * Drama
            * Fantasy
            * Film-Noir
            * Horror
            * Musical
            * Mystery
            * Romance
            * Sci-Fi
            * Thriller
            * War
            * Western

    - Some MovieIDs do not correspond to a movie due to accidental duplicate
    entries and/or test entries
    - Movies are mostly entered by hand, so errors and inconsistencies may exist
    """

    def __init__(self):
        super().__init__("ml-1m", ["ratings", "movies", "users"])

    def data_path(self, local_path=None, force_update=False, proxies=None):
        url = None
        try:
            url = ML_URLS[self.uid]
        except KeyError as error:
            raise NotImplementedError(
                "Current dataset {:s} is not implemented.".format(self.uid)
            ) from error

        file_dest = get_data_path(
            url,
            "movielens",
            path=local_path,
            proxies=proxies,
            force_update=force_update,
        )

        parent_dir = op.join(Path(file_dest).parent, self.uid)
        if not op.exists(op.join(parent_dir, "ratings.dat")):
            with zipfile.ZipFile(file_dest, "r") as archive:
                archive.extractall(path=Path(file_dest).parent)

        files = OrderedDict()
        for table in self.tables:
            files[table] = op.join(parent_dir, "{:s}.dat".format(table))
        dests = [files]
        return dests

    def get_data(self, tables=None):
        dests = self.data_path()

        column_names = {
            "ratings": ["UserID", "MovieID", "Rating", "Timestamp"],
            "movies": ["MovieID", "Title", "Genres"],
            "users": ["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        }

        if tables is None:
            tables = self.tables

        rawdata = OrderedDict()
        for table in tables:
            rawdata[table] = pd.read_csv(
                dests[0][table],
                sep="::",
                names=column_names[table],
                encoding="latin_1" if table == "movies" else "utf_8",
            )
        return rawdata


class MovieLensDataset_10M(BaseMovieLensDataset):
    """MovieLensDataset 10M.

    Available tables include: ratings, movies, and tags.

    """

    def __init__(self):
        super().__init__("ml-10m", ["ratings", "movies", "tags"])

    def data_path(self, local_path=None, force_update=False, proxies=None):
        url = None
        try:
            url = ML_URLS[self.uid]
        except KeyError as error:
            raise NotImplementedError(
                "Current dataset {:s} is not implemented.".format(self.uid)
            ) from error

        file_dest = get_data_path(
            url,
            "movielens",
            path=local_path,
            proxies=proxies,
            force_update=force_update,
        )

        parent_dir = op.join(Path(file_dest).parent, "ml-10M100K")
        if not op.exists(op.join(parent_dir, "ratings.dat")):
            with zipfile.ZipFile(file_dest, "r") as archive:
                archive.extractall(path=Path(file_dest).parent)

        files = OrderedDict()
        for table in self.tables:
            files[table] = op.join(parent_dir, "{:s}.dat".format(table))
        dests = [files]
        return dests

    def get_data(self, tables=None):
        dests = self.data_path()

        column_names = {
            "ratings": ["UserID", "MovieID", "Rating", "Timestamp"],
            "movies": ["MovieID", "Title", "Genres"],
            "tags": ["UserID", "MovieID", "Tag", "Timestamp"],
        }

        if tables is None:
            tables = self.tables

        rawdata = OrderedDict()
        for table in tables:
            rawdata[table] = pd.read_csv(
                dests[0][table],
                sep="::",
                names=column_names[table],
                encoding="utf_8",
            )
        return rawdata


class MovieLensDataset_20M(BaseMovieLensDataset):
    """MovieLensDataset 20M.

    Available tables include: ratings, movies, tags, links, genome-scores, and genome-tags.

    """

    def __init__(self):
        super().__init__(
            "ml-20m",
            ["ratings", "movies", "tags", "links", "genome-scores", "genome-tags"],
        )

    def data_path(self, local_path=None, force_update=False, proxies=None):
        url = None
        try:
            url = ML_URLS[self.uid]
        except KeyError as error:
            raise NotImplementedError(
                "Current dataset {:s} is not implemented.".format(self.uid)
            ) from error

        file_dest = get_data_path(
            url,
            "movielens",
            path=local_path,
            proxies=proxies,
            force_update=force_update,
        )

        parent_dir = op.join(Path(file_dest).parent, self.uid)
        if not op.exists(op.join(parent_dir, "ratings.csv")):
            with zipfile.ZipFile(file_dest, "r") as archive:
                archive.extractall(path=Path(file_dest).parent)

        files = OrderedDict()
        for table in self.tables:
            files[table] = op.join(parent_dir, "{:s}.csv".format(table))
        dests = [files]
        return dests

    def get_data(self, tables=None):
        dests = self.data_path()

        column_names = {
            "ratings": ["userId", "movieId", "rating", "timestamp"],
            "movies": ["movieId", "title", "genres"],
            "tags": ["userId", "movieId", "tag", "timestamp"],
            "links": ["movieId", "imdbId", "tmdbId"],
            "genome-scores": ["movieId", "tagId", "relevance"],
            "genome-tags": ["tagId", "tag"],
        }

        if tables is None:
            tables = self.tables

        rawdata = OrderedDict()
        for table in tables:
            rawdata[table] = pd.read_csv(
                dests[0][table],
                sep=",",
                header=0,
                names=column_names[table],
                encoding="utf_8",
            )
        return rawdata
