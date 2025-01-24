# -*- coding: utf-8 -*-
#
# Authors: Swolf <swolfforever@gmail.com>
# Date: 2023/10/02
# License: MIT License
"""MovieLens Datasets.

See https://grouplens.org/datasets/movielens/
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

    Here are brief descriptions of the data.

    ml-data.tar.gz   -- Compressed tar file.  To rebuild the u data files do this:
                    gunzip ml-data.tar.gz
                    tar xvf ml-data.tar
                    mku.sh

    u.data     -- The full u data set, 100000 ratings by 943 users on 1682 items.
                  Each user has rated at least 20 movies.  Users and items are
                  numbered consecutively from 1.  The data is randomly
                  ordered. This is a tab separated list of
                     user id | item id | rating | timestamp.
                  The time stamps are unix seconds since 1/1/1970 UTC

    u.info     -- The number of users, items, and ratings in the u data set.

    u.item     -- Information about the items (movies); this is a tab separated
                  list of
                  movie id | movie title | release date | video release date |
                  IMDb URL | unknown | Action | Adventure | Animation |
                  Children's | Comedy | Crime | Documentary | Drama | Fantasy |
                  Film-Noir | Horror | Musical | Mystery | Romance | Sci-Fi |
                  Thriller | War | Western |
                  The last 19 fields are the genres, a 1 indicates the movie
                  is of that genre, a 0 indicates it is not; movies can be in
                  several genres at once.
                  The movie ids are the ones used in the u.data data set.

    u.genre    -- A list of the genres.

    u.user     -- Demographic information about the users; this is a tab
                  separated list of
                  user id | age | gender | occupation | zip code
                  The user ids are the ones used in the u.data data set.

    u.occupation -- A list of the occupations.

    u1.base    -- The data sets u1.base and u1.test through u5.base and u5.test
    u1.test       are 80%/20% splits of the u data into training and test data.
    u2.base       Each of u1, ..., u5 have disjoint test sets; this if for
    u2.test       5 fold cross validation (where you repeat your experiment
    u3.base       with each training and test set and average the results).
    u3.test       These data sets can be generated from u.data by mku.sh.
    u4.base
    u4.test
    u5.base
    u5.test

    ua.base    -- The data sets ua.base, ua.test, ub.base, and ub.test
    ua.test       split the u data into a training set and a test set with
    ub.base       exactly 10 ratings per user in the test set.  The sets
    ub.test       ua.test and ub.test are disjoint.  These data sets can
                  be generated from u.data by mku.sh.

    allbut.pl  -- The script that generates training and test sets where
                  all but n of a users ratings are in the training data.

    mku.sh     -- A shell script to generate all the u data sets from u.data.
    """

    __TABLE_COLS = {
        "data": ["user id", "movie id", "rating"],
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

    def __init__(self, local_path=None):
        super().__init__("ml-100k", self.__TABLE_COLS, local_path=local_path)

    def __len__(self):
        return 100000

    def _data_path(self, local_path=None, force_update=False, proxies=None):
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

        dests = OrderedDict()
        for table in self.get_tables():
            dests[table] = op.join(parent_dir, "u.{:s}".format(table))
        return dests

    def _get_rawdata(self, dests):
        tables = self.get_tables()

        rawdata = OrderedDict()
        for table in tables:
            if table == "data":
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep="\t",
                    names=["user id", "item id", "rating", "timestamp"],
                    encoding="utf_8",
                )
                rawdata[table].rename(columns={"item id": "movie id"}, inplace=True)
                rawdata[table].drop("timestamp", axis=1, inplace=True)
            elif table == "item":
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep="|",
                    names=self.get_columns(table),
                    encoding="latin_1",
                )
            else:
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep="|",
                    names=self.get_columns(table),
                    encoding="utf_8",
                )
        return rawdata

    def _get_merged_tables(self, raw_tables):
        data = raw_tables["data"]
        item = raw_tables["item"]
        user = raw_tables["user"]

        merged = pd.merge(data, user, how="outer", on="user id")
        merged = pd.merge(merged, item, how="outer", on="movie id")
        return merged


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

    __TABLE_COLS = {
        "ratings": ["UserID", "MovieID", "Rating"],
        "movies": ["MovieID", "Title", "Genres"],
        "users": ["UserID", "Gender", "Age", "Occupation", "Zip-code"],
    }

    def __init__(self, local_path=None):
        super().__init__("ml-1m", self.__TABLE_COLS, local_path=local_path)

    def __len__(self):
        return 1000386

    def _data_path(self, local_path=None, force_update=False, proxies=None):
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

        dests = OrderedDict()
        for table in self.get_tables():
            dests[table] = op.join(parent_dir, "{:s}.dat".format(table))
        return dests

    def _get_rawdata(self, dests):
        tables = self.get_tables()
        rawdata = OrderedDict()
        for table in tables:
            if table == "ratings":
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep="::",
                    names=["UserID", "MovieID", "Rating", "Timestamp"],
                    encoding="latin_1" if table == "movies" else "utf_8",
                )
                rawdata[table].drop("Timestamp", axis=1, inplace=True)
            else:
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep="::",
                    names=self.get_columns(table),
                    encoding="latin_1" if table == "movies" else "utf_8",
                )
        return rawdata

    def _get_merged_tables(self, raw_tables):
        ratings = raw_tables["ratings"]
        movies = raw_tables["movies"]
        users = raw_tables["users"]

        merged = pd.merge(ratings, users, how="outer", on="UserID")
        merged = pd.merge(merged, movies, how="outer", on="MovieID")
        return merged


class MovieLensDataset_10M(BaseMovieLensDataset):
    """MovieLensDataset 10M.

    Available tables include: ratings, movies, and tags.

    User Ids

    Movielens users were selected at random for inclusion. Their ids have been anonymized.

    Users were selected separately for inclusion in the ratings and tags data sets, which implies that user ids may appear in one set but not the other.

    The anonymized values are consistent between the ratings and tags data files. That is, user id n, if it appears in both files, refers to the same real MovieLens user.
    Ratings Data File Structure

    All ratings are contained in the file ratings.dat. Each line of this file represents one rating of one movie by one user, and has the following format:

        UserID::MovieID::Rating::Timestamp

    The lines within this file are ordered first by UserID, then, within user, by MovieID.

    Ratings are made on a 5-star scale, with half-star increments.

    Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.
    Tags Data File Structure

    All tags are contained in the file tags.dat. Each line of this file represents one tag applied to one movie by one user, and has the following format:

        UserID::MovieID::Tag::Timestamp

    The lines within this file are ordered first by UserID, then, within user, by MovieID.

    Tags are user generated metadata about movies. Each tag is typically a single word, or short phrase. The meaning, value and purpose of a particular tag is determined by each user.

    Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.
    Movies Data File Structure

    Movie information is contained in the file movies.dat. Each line of this file represents one movie, and has the following format:

        MovieID::Title::Genres

    MovieID is the real MovieLens id.

    Movie titles, by policy, should be entered identically to those found in IMDB, including year of release. However, they are entered manually, so errors and inconsistencies may exist.

    Genres are a pipe-separated list, and are selected from the following:

        Action
        Adventure
        Animation
        Children's
        Comedy
        Crime
        Documentary
        Drama
        Fantasy
        Film-Noir
        Horror
        Musical
        Mystery
        Romance
        Sci-Fi
        Thriller
        War
        Western


    """

    __TABLE_COLS = {
        "ratings": ["UserID", "MovieID", "Rating"],
        "movies": ["MovieID", "Title", "Genres"],
        "tags": ["UserID", "MovieID", "Tag"],
    }

    def __init__(self, local_path=None):
        super().__init__("ml-10m", self.__TABLE_COLS, local_path=local_path)

    def __len__(self):
        return 10000058

    def _data_path(self, local_path=None, force_update=False, proxies=None):
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

        dests = OrderedDict()
        for table in self.get_tables():
            dests[table] = op.join(parent_dir, "{:s}.dat".format(table))
        return dests

    def _get_rawdata(self, dests):
        tables = self.get_tables()

        rawdata = OrderedDict()
        for table in tables:
            if table == "ratings":
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep="::",
                    names=["UserID", "MovieID", "Rating", "Timestamp"],
                    encoding="utf_8",
                )
                rawdata[table].drop("Timestamp", axis=1, inplace=True)
            elif table == "tags":
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep="::",
                    names=["UserID", "MovieID", "Tag", "Timestamp"],
                    encoding="utf_8",
                )
                rawdata[table].drop("Timestamp", axis=1, inplace=True)
            else:
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep="::",
                    names=self.get_columns(table),
                    encoding="utf_8",
                )
        return rawdata

    def _get_merged_tables(self, raw_tables):
        ratings = raw_tables["ratings"]
        movies = raw_tables["movies"]
        # tags = raw_tables["tags"]

        merged = pd.merge(ratings, movies, how="outer", on="MovieID")
        # merged = pd.merge(merged, tags, how="outer", on=["UserID", "MovieID"])
        return merged


class MovieLensDataset_20M(BaseMovieLensDataset):
    """MovieLensDataset 20M.

    Available tables include: ratings, movies, tags, links, genome-scores, and genome-tags.

    The dataset files are written as [comma-separated values](http://en.wikipedia.org/wiki/Comma-separated_values) files with a single header row. Columns that contain commas (`,`) are escaped using double-quotes (`"`). These files are encoded as UTF-8. If accented characters in movie titles or tag values (e.g. Misérables, Les (1995)) display incorrectly, make sure that any program reading the data, such as a text editor, terminal, or script, is configured for UTF-8.

    User Ids
    --------

    MovieLens users were selected at random for inclusion. Their ids have been anonymized. User ids are consistent between `ratings.csv` and `tags.csv` (i.e., the same id refers to the same user across the two files).

    Movie Ids
    ---------

    Only movies with at least one rating or tag are included in the dataset. These movie ids are consistent with those used on the MovieLens web site (e.g., id `1` corresponds to the URL <https://movielens.org/movies/1>). Movie ids are consistent between `ratings.csv`, `tags.csv`, `movies.csv`, and `links.csv` (i.e., the same id refers to the same movie across these four data files).


    Ratings Data File Structure (ratings.csv)
    -----------------------------------------

    All ratings are contained in the file `ratings.csv`. Each line of this file after the header row represents one rating of one movie by one user, and has the following format:

        userId,movieId,rating,timestamp

    The lines within this file are ordered first by userId, then, within user, by movieId.

    Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).

    Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

    Tags Data File Structure (tags.csv)
    -----------------------------------

    All tags are contained in the file `tags.csv`. Each line of this file after the header row represents one tag applied to one movie by one user, and has the following format:

        userId,movieId,tag,timestamp

    The lines within this file are ordered first by userId, then, within user, by movieId.

    Tags are user-generated metadata about movies. Each tag is typically a single word or short phrase. The meaning, value, and purpose of a particular tag is determined by each user.

    Timestamps represent seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970.

    Movies Data File Structure (movies.csv)
    ---------------------------------------

    Movie information is contained in the file `movies.csv`. Each line of this file after the header row represents one movie, and has the following format:

        movieId,title,genres

    Movie titles are entered manually or imported from <https://www.themoviedb.org/>, and include the year of release in parentheses. Errors and inconsistencies may exist in these titles.

    Genres are a pipe-separated list, and are selected from the following:

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
    * (no genres listed)

    Links Data File Structure (links.csv)
    ---------------------------------------

    Identifiers that can be used to link to other sources of movie data are contained in the file `links.csv`. Each line of this file after the header row represents one movie, and has the following format:

        movieId,imdbId,tmdbId

    movieId is an identifier for movies used by <https://movielens.org>. E.g., the movie Toy Story has the link <https://movielens.org/movies/1>.

    imdbId is an identifier for movies used by <http://www.imdb.com>. E.g., the movie Toy Story has the link <http://www.imdb.com/title/tt0114709/>.

    tmdbId is an identifier for movies used by <https://www.themoviedb.org>. E.g., the movie Toy Story has the link <https://www.themoviedb.org/movie/862>.

    Use of the resources listed above is subject to the terms of each provider.

    Tag Genome (genome-scores.csv and genome-tags.csv)
    -------------------------------------------------

    This data set includes a current copy of the Tag Genome.

    [genome-paper]: http://files.grouplens.org/papers/tag_genome.pdf

    The tag genome is a data structure that contains tag relevance scores for movies.  The structure is a dense matrix: each movie in the genome has a value for *every* tag in the genome.

    As described in [this article][genome-paper], the tag genome encodes how strongly movies exhibit particular properties represented by tags (atmospheric, thought-provoking, realistic, etc.). The tag genome was computed using a machine learning algorithm on user-contributed content including tags, ratings, and textual reviews.

    The genome is split into two files.  The file `genome-scores.csv` contains movie-tag relevance data in the following format:

        movieId,tagId,relevance

    The second file, `genome-tags.csv`, provides the tag descriptions for the tag IDs in the genome file, in the following format:

        tagId,tag

    The `tagId` values are generated when the data set is exported, so they may vary from version to version of the MovieLens data sets.
    """

    __TABLE_COLS = {
        "ratings": ["userId", "movieId", "rating"],
        "movies": ["movieId", "title", "genres"],
        "tags": ["userId", "movieId", "tag"],
        "links": ["movieId", "imdbId", "tmdbId"],
        "genome-scores": ["movieId", "tagId", "relevance"],
        "genome-tags": ["tagId", "tag"],
    }

    def __init__(self, local_path=None):
        super().__init__("ml-20m", self.__TABLE_COLS, local_path=local_path)

    def __len__(self):
        return 20000797

    def _data_path(self, local_path=None, force_update=False, proxies=None):
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

        dests = OrderedDict()
        for table in self.get_tables():
            dests[table] = op.join(parent_dir, "{:s}.csv".format(table))
        return dests

    def _get_rawdata(self, dests):
        tables = self.get_tables()

        rawdata = OrderedDict()
        for table in tables:
            if table == "ratings":
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep=",",
                    header=0,
                    names=["userId", "movieId", "rating", "timestamp"],
                    encoding="utf_8",
                )
                rawdata[table].drop("timestamp", axis=1, inplace=True)
            elif table == "tags":
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep=",",
                    header=0,
                    names=["userId", "movieId", "tag", "timestamp"],
                    encoding="utf_8",
                )
                rawdata[table].drop("timestamp", axis=1, inplace=True)
            else:
                rawdata[table] = pd.read_csv(
                    dests[table],
                    sep=",",
                    header=0,
                    names=self.get_columns(table),
                    encoding="utf_8",
                )
        return rawdata

    def _get_merged_tables(self, raw_tables):
        ratings = raw_tables["ratings"]
        movies = raw_tables["movies"]
        # tags = raw_tables["tags"]
        # links = raw_tables["links"]
        # genome_scores = raw_tables["genome-scores"]
        # genome_tags = raw_tables["genome-tags"]

        merged = pd.merge(ratings, movies, how="outer", on="movieId")
        # merged = pd.merge(merged, tags, how="outer", on=["userId", "movieId"])
        return merged
