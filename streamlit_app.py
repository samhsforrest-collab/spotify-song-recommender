# Python Environments
import numpy as np
import pandas as pd
import ast
import plotly.graph_objects as go

# Streamlit Environments
import streamlit as st

# Tensor Flow Environments
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Flatten, Dropout, BatchNormalization, GlobalAveragePooling1D, concatenate
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from neural_network_functions import process_genres, prepare_numeric_features, build_get_embeddings, recommend_by_song

st.title("Music is the Answer. To your problems...")

# Load logo
image_path = "assets/spoofify_logo.png"  # Ensure this path points to your logo image
st.sidebar.image(image_path, width=200)  # Display the logo in the sidebar

# Create a two-column layout for the first row
col1, col2  = st.columns (2) # Adjust the widths if desired

data = pd.read_csv("datasets/sam_df_clean.csv")

# create new df with additional genre column and adding integer columns for genre ids
genre_df = process_genres(data)

# Create the dropdown in the sidebar
st.sidebar.title("Put your Soul in your Playlist")
# Create a list of song names combined with their artists, sorted alphabetically
song_names_with_artists = genre_df.apply(lambda row: f"{row['track_name']} | {row['artists']}", axis=1).tolist()
song_names_with_artists.sort()  # Sort the list alphabetically

# Create a dropdown select box with options for playlist lengths
playlist_length = st.sidebar.selectbox("Choose playlist length:", [5, 10, 25, 50, 100])

# Dropdown select for song names with artists
selected_song = st.sidebar.selectbox("Start typing to select your song:", song_names_with_artists)

# Extract the track name and artist from the selected option
track_name, artists = selected_song.split(" | ")

st.write(f"You selected: {track_name} by {artists}")  # Display the selected song name

# Save mapping from song Id to row index (needed for recommendations later)
song_ids = genre_df['track_id'].values
id2index = {sid: idx for idx, sid in enumerate(song_ids)}
index2id = {idx: sid for sid, idx in id2index.items()}

# prepare numeric features
X_numeric_sound_profile_input, scaler = prepare_numeric_features(genre_df)

# build model and get id->embedding mapping
model, all_embeddings, id2embedding = build_get_embeddings(genre_df, X_numeric_sound_profile_input, song_ids)

# Get recommendations
song_id = genre_df.loc[genre_df['track_name'] == track_name, 'track_id'].values[0]

recommendations = recommend_by_song(genre_df,song_id, k=playlist_length, id2index=id2index, index2id=index2id, all_embeddings=all_embeddings)
recs_df = pd.DataFrame(recommendations)
recs_df.index = range(1, len(recs_df) + 1)  # Adjusting the index to start at 1
recs_display_df = recs_df[['track_name', 'artists', 'score']]

with col1:
    # Assuming you have genre_df defined with the necessary columns including song_id
    selected_track = genre_df.loc[genre_df['track_id'] == song_id]

    if not selected_track.empty:
        numeric_features = ['popularity', 'danceability', 'energy', 
                            'acousticness', 'instrumentalness', 
                            'liveness', 'valence']

        # Scale 'popularity' to range [0, 1]
        popularity_scaled = selected_track['popularity'].values[0] / 100.0  # Assuming original range is [0, 100]

        # Create a feature array replacing the original popularity with the scaled value
        selected_features = selected_track[numeric_features].values.flatten()
        selected_features[0] = popularity_scaled  # Replace popularity with scaled value

        # Create radar chart
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=selected_features,
            theta=numeric_features,
            fill='toself',
            name=f'Song: {selected_track["track_name"].values[0]} by {selected_track["artists"].values[0]}',
            fillcolor='rgba(0, 255, 0, 0.6)',  # Neon green fill with transparency
            line=dict(color='lime', width=2)   # Lime green outline
        ))

        # Update layout of the radar chart
        fig.update_layout(
            width=500,
            height=350,
            polar=dict(
                radialaxis=dict(
                    showticklabels=False,  # Hide radial axis labels
                    range=[0, 1]  # This ensures the range of the radar chart
                )
            ),
            title=f"Song profile: {selected_track['track_name'].values[0]}",
            template="plotly_dark"
        )

        # Display the radar chart in the Streamlit app
        st.plotly_chart(fig)
    else:
        st.write("No data found for the selected song.")

# Second Radar Chart for Recommendations
with col2:
    if not recs_df.empty:
        # Create the radar chart
        fig = go.Figure()

        # List of distinct colors to use for layering
        colors = ['rgba(0, 255, 0, 0.6)',  # Neon Green
                  'rgba(255, 0, 255, 0.6)',  # Neon Purple
                  'rgba(255, 165, 0, 0.6)',  # Neon Orange
                  'rgba(255, 0, 0, 0.6)',    # Neon Red
                  'rgba(0, 0, 255, 0.6)',    # Neon Blue
                  'rgba(255, 20, 147, 0.6)', # Deep Pink
                  'rgba(75, 0, 130, 0.6)']   # Indigo

        # Create radar charts for each recommended song
        for idx in range(len(recs_df)):
            selected_recommendation = recs_df.iloc[idx]
            selected_features = np.array([
                selected_recommendation['popularity'] / 100.0,  # Scale popularity to [0, 1]
                selected_recommendation['danceability'],
                selected_recommendation['energy'],
                selected_recommendation['acousticness'],
                selected_recommendation['instrumentalness'],
                selected_recommendation['liveness'],
                selected_recommendation['valence']
            ])

            # Add each recommended song as a layer in the radar chart with different colors
            fig.add_trace(go.Scatterpolar(
                r=selected_features,
                theta=numeric_features,
                fill='toself',
                name=str(idx + 1),  # Use index number as the label
                fillcolor=colors[idx % len(colors)],  # Cycle through the color list
                line=dict(color=colors[idx % len(colors)], width=2)  # Use the same color for the line
            ))

        # Update layout of the radar chart
        fig.update_layout(
            width=500,
            height=350,
            polar=dict(
                radialaxis=dict(
                    showticklabels=False,  # Hide radial axis labels
                    range=[0, 1]  # Ensure this matches your scaled values
                )
            ),
            title=f"Recommended song profiles",
            template="plotly_dark"
        )

        # Display the radar chart in the Streamlit app
        st.plotly_chart(fig)
    else:
        st.write("No recommendations found.")

# Display the recommendations as a table in the Streamlit app
st.title('Song Recommendations')
st.table(recs_display_df)