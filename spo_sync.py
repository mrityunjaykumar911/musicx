#
import pandas as pd
import numpy as np
import os

pickle_path = "data/dummy2.pkl"

import sys

from spotipy import util
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from pprint import pprint

count = 0
total = 0
spotify_search_done = 0
none_count = 0
done = 0
if len(sys.argv) > 1:
    search_str = sys.argv[1]
else:
    search_str = 'Radiohead'

SPOTIPY_CLIENT_ID = '46cde2c012e444fbbdd451b5d6adfad4'
SPOTIPY_CLIENT_SECRET = '87d073dcb76b43778dcc39c9192751d0'

client_id = SPOTIPY_CLIENT_ID
client_secret = SPOTIPY_CLIENT_SECRET
df_tracks = None
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

import signal

run = True


def handler_stop_signals(signum, frame):
    global run
    run = False
    save()


signal.signal(signal.SIGINT, handler_stop_signals)
signal.signal(signal.SIGTERM, handler_stop_signals)


def search_sp(row_input):
    if run is False:
        exit(0)

    global count, done, total, spotify_search_done, none_count
    print("Done {} Misses {} Total {} Spotify API Counter {} Nones {}".format(done, count, total, spotify_search_done,
                                                                              none_count))
    if row_input["spotify"] is None:
        none_count += 1

    if row_input["spotify_ret"] is True:
        done += 1
        return row_input
    else:
        track_name = row_input["tracktitle"]
        try:
            result = sp.search(track_name, limit=1)
            ans = result["tracks"]["items"]
            if len(ans) != 0:
                xxx = {k: v for k, v in result["tracks"]["items"][0].items() if
                       k in ["name", "external_urls", "uri", "images","artists"]}
                try:
                    aid = xxx["artists"][0]['id']
                    artist_info = sp.artist(artist_id=aid)
                    xxx.update({"genre": artist_info['genres']})
                except:
                    xxx.update({"genre": ""})

                row_input["spotify"] = xxx
                row_input["spotify_ret"] = True

                done += 1
                spotify_search_done += 1
            else:
                row_input["spotify"] = None
                row_input["spotify_ret"] = True
                done += 1
                spotify_search_done += 1
        except:
            row_input["spotify_ret"] = False
            count += 1
            spotify_search_done -= 1
        return row_input


def savecounter():
    save()


import atexit

atexit.register(savecounter)


def save():
    global df_tracks
    if df_tracks is None:
        return
    df_tracks.to_pickle(pickle_path)
    print("exit done")


def load():
    return pd.read_pickle(pickle_path)


def run2():
    global df_tracks, total

    if os.path.exists(pickle_path):
        df_tracks = load()
        # df_tracks = df_tracks.sample(5)
        total = df_tracks.trackid.count().tolist()
    else:
        df_tracks = pd.read_csv("data/tracks.csv")
        df_tracks["spotify"] = ""
        df_tracks["spotify_ret"] = False
        exit(0)

    df_tracks = df_tracks.apply(search_sp, axis=1)


def run3():
    df_tracks = load()
    df_tracks.to_csv("data/tracks22.csv")


if __name__ == '__main__':
    # run2()
    run3()
