import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from dotenv import load_dotenv, dotenv_values
import os
import time
import streamlit as st
from pandas.io.json import json_normalize


load_dotenv()


CLIENT_ID = os.getenv("client_id")
SECRET = os.getenv("secret")
URI = os.getenv("uri")
SCOPE = os.getenv("scope")

spotify = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=CLIENT_ID,client_secret=SECRET,redirect_uri=URI,scope=SCOPE,username="stefa2211"))


def time_songs(convertion):
    recently_played = spotify.current_user_recently_played()
    total_time_spent = 0
    for song in recently_played['items']:
        total_time_spent += song['track']['duration_ms']
        min_time = total_time_spent / convertion
    return round(min_time, 2)

def tracks_dataframe():
    top_tracks =  spotify.current_user_top_tracks(time_range='medium_term', limit=100)
    tracks = top_tracks['items']
    tracks_ids = []
    tracks_names = []
    features = []
    artist_names = []

    for track in tracks:
        tracks_id = track['id']
        tracks_name = track['name']
        artist_name = track['artists'][0]['name']
        feature = spotify.audio_features(tracks_id)

        tracks_ids.append(tracks_id)
        artist_names.append(artist_name)
        tracks_names.append(tracks_name)
        features.append(feature[0])
    
    top_df = pd.DataFrame(features)
    top_df['Tracks Names'] = tracks_names
    top_df['Artist Names'] = artist_names
    top_df = top_df.reindex(columns=['Tracks Names','Artist Names',"id", "acousticness", "danceability", 
            "duration_ms", "energy", "instrumentalness",  "key", "liveness", 
            "loudness", "mode", "speechiness", "tempo", "valence"])


   
    return top_df

def count_artist():
    df = tracks_dataframe()
    artist_counts = df['Artist Names'].value_counts().head(6)
    count_artists = pd.DataFrame(artist_counts)
    count_artists = count_artists.reset_index()
    return count_artists

def count_tracks():
    df = tracks_dataframe()
    tracks_counts = df['Tracks Names'].value_counts().head(6)
    count_track = pd.DataFrame(tracks_counts)
    count_track = count_track.reset_index()
    return count_track

def get_top_tracks():
    '''Obtener listado de pistas más escuchadas recientemente'''
    top_tracks = spotify.current_user_top_tracks(time_range='medium_term', limit=5)
    return top_tracks

def create_tracks_dataframe(top_tracks):
    '''Obtener "audio features" de las pistas más escuchadas por el usuario'''
    tracks = top_tracks['items']
    tracks_ids = [track['id'] for track in tracks]
    audio_features = spotify.audio_features(tracks_ids)
    top_tracks_df = pd.DataFrame(audio_features)
    top_tracks_df = top_tracks_df[["id", "acousticness", "danceability", 
            "duration_ms", "energy", "instrumentalness",  "key", "liveness", 
            "loudness", "mode", "speechiness", "tempo", "valence"]]

    return top_tracks_df

def get_artists_ids(top_tracks):
        '''Obtener ids de los artistas en "top_tracks"'''
        ids_artists = []

        for item in top_tracks['items']:
            artist_id = item['artists'][0]['id']
            artist_name = item['artists'][0]['name']
            ids_artists.append(artist_id)

        # Depurar lista para evitar repeticiones
        ids_artists = list(set(ids_artists))

        return ids_artists

def get_similar_artists_ids(ids_artists):
        '''Expandir el listado de "ids_artists" con artistas similares'''
        ids_similar_artists = []
        for artist_id in ids_artists:
            artists = spotify.artist_related_artists(artist_id)['artists']
            for item in artists:
                artist_id = item['id']
                artist_name = item['name']
                ids_similar_artists.append(artist_id)

        ids_artists.extend(ids_similar_artists)

        # Depurar lista para evitar repeticiones
        ids_artists = list(set(ids_artists))

        return ids_artists

def get_new_releases_artists_ids(ids_artists):
        '''Expandir el listado de "ids_artists" con artistas con nuevos lanzamientos'''

        new_releases = spotify.new_releases(limit=10)['albums']
        for item in new_releases['items']:
            artist_id = item['artists'][0]['id']
            ids_artists.append(artist_id)

        # Depurar lista para evitar repeticiones
        ids_artists = list(set(ids_artists))

        return ids_artists

def get_albums_ids(ids_artists):
        '''Obtener listado de albums para cada artista en "ids_artists"'''
        ids_albums = []
        for id_artist in ids_artists:
            album = spotify.artist_albums(id_artist, limit=1)['items'][0]
            ids_albums.append(album['id'])

        return ids_albums

def get_albums_tracks(ids_albums):
        '''Extraer 3 tracks para cada album en "ids_albums"'''
        ids_tracks = []
        for id_album in ids_albums:
            album_tracks = spotify.album_tracks(id_album, limit=1)['items']
            for track in album_tracks:
                ids_tracks.append(track['id'])

        return ids_tracks

def get_tracks_features(ids_tracks):
        '''Extraer audio features de cada track en "ids_tracks" y almacenar resultado
        en un dataframe de Pandas'''

        ntracks = len(ids_tracks)

        if ntracks > 100:
            # Crear lotes de 100 tracks (limitacion de audio_features)
            m = ntracks//100
            n = ntracks%100
            lotes = [None]*(m+1)
            for i in range(m):
                lotes[i] = ids_tracks[i*100:i*100+100]

            if n != 0:
                lotes[i+1] = ids_tracks[(i+1)*100:]
        else:
            lotes = [ids_tracks]


        # Iterar sobre "lotes" y agregar audio features
        audio_features = []
        for lote in lotes:
            features = spotify.audio_features(lote)
            audio_features.append(features)

        audio_features = [item for sublist in audio_features for item in sublist]

        # Crear dataframe
        candidates_df = pd.DataFrame(audio_features)
        candidates_df = candidates_df[["id", "acousticness", "danceability", "duration_ms",
            "energy", "instrumentalness",  "key", "liveness", "loudness", "mode", 
            "speechiness", "tempo", "valence"]]

        return candidates_df

def compute_cossim(top_tracks_df, candidates_df):
        '''Calcula la similitud del coseno entre cada top_track y cada pista
        candidata en candidates_df. Retorna matriz de n_top_tracks x n_candidates_df'''
        top_tracks_mtx = top_tracks_df.iloc[:,1:].values
        candidates_mtx = candidates_df.iloc[:,1:].values

        # Estandarizar cada columna de features: mu = 0, sigma = 1
        scaler = StandardScaler()
        top_tracks_scaled = scaler.fit_transform(top_tracks_mtx)
        can_scaled = scaler.fit_transform(candidates_mtx)

        # Normalizar cada vector de características (magnitud resultante = 1)
        top_tracks_norm = np.sqrt((top_tracks_scaled*top_tracks_scaled).sum(axis=1))
        can_norm = np.sqrt((can_scaled*can_scaled).sum(axis=1))

        n_top_tracks = top_tracks_scaled.shape[0]
        n_candidates = can_scaled.shape[0]
        top_tracks = top_tracks_scaled/top_tracks_norm.reshape(n_top_tracks,1)
        candidates = can_scaled/can_norm.reshape(n_candidates,1)

        # Calcular similitudes del coseno
        cos_sim = linear_kernel(top_tracks,candidates)

        return cos_sim

def content_based_filtering(pos, cos_sim, ncands, umbral = 0.8):
        '''Dada una pista de top_tracks (pos = 0, 1, ...) extraer "ncands" candidatos,
        usando "cos_sim" y siempre y cuando superen un umbral de similitud'''

        # Obtener todas las pistas candidatas por encima del umbral
        idx = np.where(cos_sim[pos,:]>=umbral)[0] # ejm. idx: [27, 82, 135]

        # Y organizarlas de forma descendente (por similitudes de mayor a menor)
        idx = idx[np.argsort(cos_sim[pos,idx])[::-1]]

        # Si hay más de "ncands", retornar únicamente un total de "ncands"
        if len(idx) >= ncands:
            cands = idx[0:ncands]
        else:
            cands = idx

        return cands

def create_recommended_playlist():
        '''Crear la lista de recomendaciones en Spotify. Ejecuta todos los métodos
        anteriores'''



        # Obtener candidatos y compararlos (distancias coseno) con las pistas
        # del playlist original
        top_tracks = get_top_tracks()
        top_tracks_df = create_tracks_dataframe(top_tracks)
        ids_artists = get_artists_ids(top_tracks)
        ids_artists = get_similar_artists_ids(ids_artists)
        ids_artists = get_new_releases_artists_ids(ids_artists)
        ids_albums = get_albums_ids(ids_artists)
        ids_tracks = get_albums_tracks(ids_albums)
        candidates_df = get_tracks_features(ids_tracks)
        cos_sim = compute_cossim(top_tracks_df, candidates_df)

        # Crear listado de ids con las recomendaciones
        ids_top_tracks = []
        ids_playlist = []

        for i in range(top_tracks_df.shape[0]):
            ids_top_tracks.append(top_tracks_df['id'][i])

            # Obtener listado de candidatos (5) para esta pista
            cands = content_based_filtering(pos=i, cos_sim=cos_sim, ncands=1, umbral=0.8)

            # Si hay pistas relacionadas obtener los ids correspondientes
            if len(cands)==0:
                continue
            else:
                for j in cands:
                    id_cand = candidates_df['id'][j]
                    ids_playlist.append(id_cand)

        # Eliminar candidatos que ya están en top-tracks
        ids_playlist_dep = [x for x in ids_playlist if x not in ids_top_tracks]

        # Y eliminar posibles repeticiones
        ids_playlist_dep = list(set(ids_playlist_dep))

        song = []
        for id in ids_playlist_dep:
            song_info = spotify.track(id)
            name_song = song_info['name']
            song.append(name_song)

        return song
        

