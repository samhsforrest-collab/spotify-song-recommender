# Python Environments
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Tensor Flow Environments
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D, concatenate
from tensorflow.keras.utils import plot_model


# define the genre categories with dictionary
genre_categories2 = {
    'pop-mainstream': [
        'pop', 'pop-film', 'power-pop', 'k-pop', 'j-pop', 'mandopop', 
        'cantopop', 'indie-pop', 'synth-pop', 'j-idol'
    ],
    'rock': [
        'rock', 'alt-rock', 'grunge', 'punk', 'punk-rock', 'indie', 
        'psych-rock', 'garage', 'rock-n-roll', 'rockabilly', 'hard-rock'
    ],
    'electronic': [
        'house', 'techno', 'trance', 'dubstep', 'edm', 'electro', 'electronic',
        'drum-and-bass', 'deep-house', 'progressive-house', 'chicago-house',
        'detroit-techno', 'hardstyle', 'minimal-techno', 'idm'
    ],
    'hiphop-rnb': [
        'hip-hop', 'r-n-b','trip-hop'
    ],
    'metal': [
        'metal', 'heavy-metal', 'death-metal', 'black-metal', 'metalcore',
        'grindcore', 'hardcore'
    ],
    'country-folk': [
        'country', 'folk', 'honky-tonk', 'singer-songwriter', 'songwriter'
    ],
    'jazz-blues': [
        'jazz', 'blues', 'soul'
    ],
    'world-regional': [
        'latin', 'latino', 'afrobeat', 'brazil', 'forro', 'salsa', 'samba',
        'sertanejo', 'pagode', 'mpb', 'french', 'spanish', 'german', 'swedish',
        'indian', 'iranian', 'malay', 'turkish', 'j-dance', 'j-rock', 'tango'
    ],
    'dance-club': [
        'dance', 'dancehall', 'disco', 'club', 'reggaeton', 'reggae', 'dub'
    ],
    'classical': [
        'classical', 'opera', 'new-age'
    ],
    'niche-mood': [
        'acoustic', 'alternative', 'ambient', 'anime', 'bluegrass', 'breakbeat', 'british',
        'children', 'chill', 'comedy', 'disney', 'emo', 'funk', 'gospel', 
        'goth', 'guitar', 'groove', 'happy', 'industrial', 'kids', 'party',
        'piano', 'romance', 'sad', 'show-tunes', 'ska', 'sleep', 'study', 'world-music'
    ]
}
def process_genres(df: pd.DataFrame) -> pd.DataFrame:
    X = df.copy()
    genre_to_cat = {}
    for cat, genres in genre_categories2.items():
        for genre in genres:
            genre_to_cat[genre] = cat

    X['genre_subcategory'] = X['track_genre'].map(genre_to_cat).fillna('uncategorized')

    unique_track_genres = pd.Series(X['track_genre'].unique())
    unique_subcats = pd.Series(X['genre_subcategory'].unique())

    track_genre_to_index = {g: i for i, g in enumerate(unique_track_genres)}
    subcat_to_index = {g: i for i, g in enumerate(unique_subcats)}

    X['track_genre_id'] = X['track_genre'].map(track_genre_to_index)
    X['genre_subcat_id'] = X['genre_subcategory'].map(subcat_to_index)

    return X

def prepare_numeric_features(df, numeric_features=None):
    if numeric_features is None:
        numeric_features = [
            'popularity','danceability','duration_ms','energy','key','loudness',
            'speechiness','acousticness','instrumentalness','liveness',
            'valence','tempo'
        ]
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(df[numeric_features])
    return X_numeric.astype('float32'), scaler


def build_get_embeddings(multi_genre_df2, X_numeric_sound_profile_input, song_ids):
    # build arrays for each position
    X_track = multi_genre_df2['track_genre_id'].astype('int32').values
    X_subcat = multi_genre_df2['genre_subcat_id'].astype('int32').values

    # vocab sizes for separate embeddings
    num_track_genres = int(X_track.max()) + 1
    num_subcats = int(X_subcat.max()) + 1

    # data arrays
    X_numeric = X_numeric_sound_profile_input.astype('float32')

    embedding_dim = 32

    # model inputs
    track_input = Input(shape=(1,), dtype='int32', name='track_input')
    subcat_input = Input(shape=(1,), dtype='int32', name='subcat_input')
    numeric_input = Input(shape=(X_numeric.shape[1],), name='numeric_input')

    # numeric branch (song profile) NN pipeline
    x_numeric = BatchNormalization()(numeric_input)
    x_numeric = Dense(64, activation='relu')(x_numeric)
    x_numeric = Dropout(0.2)(x_numeric)

    # embeddings (separate)
    track_emb = Embedding(input_dim=num_track_genres, output_dim=24, name='track_embedding')(track_input)
    track_emb = Flatten()(track_emb)

    subcat_emb = Embedding(input_dim=num_subcats, output_dim=8, name='subcat_embedding')(subcat_input)
    subcat_emb = Flatten()(subcat_emb)

    # combine
    x = concatenate([track_emb, subcat_emb, x_numeric])
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    song_embedding = Dense(64, activation=None, name='song_embedding')(x)

    model = Model(inputs=[track_input, subcat_input, numeric_input], outputs=song_embedding)

    # prepare inputs (ensure int32 for indices, float32 for numeric)
    X_track_in = X_track.astype('int32')
    X_subcat_in = X_subcat.astype('int32')
    X_numeric_in = X_numeric

    # Extract embeddings for all songs
    all_embeddings = model.predict([X_track_in, X_subcat_in, X_numeric_in], batch_size=1024)

    # Normalize for cosine similarity (avoid division by zero)
    norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    all_embeddings = all_embeddings / norms

    id2embedding = {sid: emb for sid, emb in zip(song_ids, all_embeddings)}
    return model, all_embeddings, id2embedding

def recommend_by_song(df, song_id, k=None, id2index=None, index2id=None, all_embeddings=None):
    """
    Return top-k most similar songs to song_id based on cosine similarity over normalized embeddings.
    Returns list of dicts: [{'track_id': ..., 'track_name': ..., 'artists': ..., 'popularity': ..., 
    'danceability': ..., 'energy': ..., 'score': ...}, ...]
    """
    if song_id not in id2index:
        raise KeyError(f"song_id {song_id} not found")
    
    q_idx = id2index[song_id]
    q_emb = all_embeddings[q_idx].reshape(1, -1)  # (1, D)
    
    # Cosine similarity with normalized vectors = dot product
    sims = all_embeddings.dot(q_emb.T).ravel()  # shape (n_songs,)
    
    # Exclude the query itself
    sims[q_idx] = -1.0
    
    # Get top-k indices
    topk_idx = np.argpartition(-sims, range(k))[:k]
    topk_idx = topk_idx[np.argsort(-sims[topk_idx])]  # sort top-k by score desc

    results = []
    for idx in topk_idx:
        sid = index2id[idx]
        row = df.loc[df['track_id'] == sid].iloc[0]  # Get the corresponding row
        results.append({
            'track_id': sid,
            'track_name': row.get('track_name', None),
            'artists': row.get('artists', None),
            'popularity': row.get('popularity', None),  # Include popularity
            'danceability': row.get('danceability', None),  # Include other features
            'energy': row.get('energy', None),
            'acousticness': row.get('acousticness', None),
            'instrumentalness': row.get('instrumentalness', None),
            'liveness': row.get('liveness', None),
            'valence': row.get('valence', None),
            'score': f"{float(sims[idx]) * 100:.0f}%"  # Format score as percentage
        })
    return results
