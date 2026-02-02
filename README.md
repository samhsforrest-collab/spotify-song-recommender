# Spotify Music Recommendation System
A project by Team 3 for Code Institute's Hackathon 2
 
<br>

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Understanding the Dataset](#2-understanding-the-dataset)
3. [Aim & Objectives](#3-aim--objectives)
4. [Research Questions](#4-research-questions)
5. [Tools & Technologies](#5-tools--technologies)
6. [How to Run the Project Locally](#6-how-to-run-the-project-locally)
7. [Team Members & Roles](#7-team-members--roles)

<br>

## 1. Project Overview

Music streaming platforms rely heavily on data-driven insights to understand listener preferences, track performance, and relationships between musical attributes. With the growth of large-scale music datasets, there is increasing opportunity to analyse how measurable audio features relate to engagement outcomes such as popularity and musical characteristics (like valence and energy).

This project explores a large Spotify track dataset to investigate whether quantifiable audio features and genre classifications can act as indicators of popularity and musical characteristics. In addition, the project explores how these insights can support the foundations of a simple recommendation system.

The analysis combines exploratory data analysis, statistical investigation, and neural network techniques to derive interpretable insights from music streaming data.

<br>

## 2. Understanding the Dataset

The dataset used in this project is the [Spotify tracks dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) sourced from Kaggle. The original dataset contains **114,000 entries,** and **21 columns**. The tracks span across **125 distinct genres**, with each track associated with a range of metadata and audio features.

<br>

**Column Specifications**

| Column             | Type              | Description |
|--------------------|-------------------|-------------|
| track_id          | String           | Unique Spotify ID for the track (e.g., "spotify:track:2takcwOaAZWiXQijPHIx7B"). |
| artists           | String           | Names of performing artists, separated by semicolons for multiples. |
| album_name        | String           | Name of the album containing the track. |
| track_name        | String           | Title of the track. |
| popularity        | Integer (0-100)  | Score where higher values indicate greater popularity based on plays and recency. |
| duration_ms       | Integer          | Track length in milliseconds. |
| explicit          | Boolean          | Indicates if the track contains explicit lyrics. |
| danceability      | Float (0.0-1.0)  | Suitability for dancing based on tempo, rhythm stability, beat strength, and regularity (0.0 least, 1.0 most). |
| energy            | Float (0.0-1.0)  | Perceived intensity and activity (0.0 calm, 1.0 high-energy). |
| key               | Integer (-1 to 11)| Numeric key (0=C, 1=C♯/D♭); -1 if no key detected. |
| loudness          | Float            | Overall loudness in decibels (typically -60 to 0). |
| mode              | Integer (0-1)    | Modality (0=major, 1=minor). |
| speechiness       | Float (0.0-1.0)  | Presence of spoken words (0.0=music, 1.0=spoken like podcast). |
| acousticness      | Float (0.0-1.0)  | Confidence that the track is acoustic (0.0=electric, 1.0=acoustic). |
| instrumentalness  | Float (0.0-1.0)  | Likelihood of no vocals (0.0=vocalic, 1.0=instrumental). |
| liveness          | Float (0.0-1.0)  | Detection of live audience (higher for live recordings). |
| valence           | Float (0.0-1.0)  | Musical positiveness (0.0=sad/angry, 1.0=happy/cheerful). |
| tempo             | Float            | Estimated beats per minute (BPM). |
| time_signature    | Integer          | An estimated overall time signature (e.g., 4/4 as 4). |
| track_genre       | String           | Assigned genre label (e.g., from datasets covering 125 genres). |

<br>

## 3. Aim & Objectives

**Aim**

The primary aim of this project is to analyse whether audio features and genre classifications can be used to explain or predict track popularity and musical characteristics, and to explore how these features can support basic music recommendation logic.

**Objectives**

- To clean and prepare a large-scale music dataset for reliable analysis
- To explore relationships between genre and engagement metrics such as popularity
- To examine whether specific audio features act as indicators of other musical attributes (e.g. danceability)
- To assess correlations between emotional tone (valence), tempo, loudness, and danceability
- To experiment with regression and machine learning techniques for modelling musical features
- To lay groundwork for a simple recommendation approach based on audio similarity

<br>

## 4. Research Questions

Based on initial exploration of the dataset, the following research questions guide the project:

1. Is musical genre an indicator of track popularity on Spotify?
    
    **Hypothesis:** Tracks belonging to certain genres have significantly higher average popularity scores than others.
2. Do certain audio features (e.g. loudness, energy, tempo) show strong relationships with musical characteristics (e.g. valence and danceability)?

    **Hypothesis**: Tempo is positively correlated with danceability.
3. Do collaborative tracks (tracks with multiple artists) differ in popularity or behaviour compared to solo tracks?

    **Hypothesis:** Tracks involving multiple artists demonstrate different popularity patterns compared to solo-artist tracks.

**NOTE:**

For the sake of clarity and consistency within this analysis, some of the variables derived from the dataset are grouped into two categories: **audio features** and **musical characteristics**.

**Audio features** - Variables that describe the technical and signal-level properties of a track. Things that are less subjective. In this project, they are the following variables:

- Energy
- Loudness
- Tempo
- Speechiness
- Instrumentalness

**Musical characteristics** - Variables that describe the perceptual qualities of a track, particularly in relation to emotional tone and movement. In this project, they are the following variables:

- Danceability
- Valence
- Acousticness

<br>

## 5. Tools & Technologies

The project uses the following tools and technologies:

- **Python** for data analysis and modelling
- **Pandas & NumPy** for data cleaning, manipulation, and feature engineering
- **Matplotlib & Seaborn** for visualisation (histograms, box plots, bar charts, heatmaps)
- **Scikit-learn** for regression modelling and evaluation
- **TensorFlow / Keras** for exploratory neural network modelling and embeddings
- **Jupyter Notebooks** for reproducible analysis and documentation
- **Tableau/ Power BI** for dashboarding
- **Streamlit** for dashboarding and presentation

<br>

## 6. How to Run the Project Locally
### Clone the Repository

```bash
git clone [https://github.com/PearlisSad/CI--Music--Streaming-and-Recommendation-System]

cd CI--Music--Streaming-and-Recommendation-System
```

### Install Dependencies

You will need the ```requirements.txt``` file listing pandas, matplotlib, streamlit, etc.
Open your terminal or a Jupyter cell and run:

```bash
pip install -r requirements.txt
```

### Streamlit Dashboard

Run the streamlit ```streamlit.py``` app and it will open automatically in your browser, displaying the plots.

```bash
streamlit run streamlit.py
```

### Power BI/Tableau Dasboard

An interactive Power BI/Tableau Dasboard can be accessed [__here.__](https://public.tableau.com/app/profile/kanyinsola.fakehinde/viz/DecipheringtheDNAofPopularity/Dashboard1?publish=yes)


### Run the Notebook

All the notebooks can be run sequentially. They will automatically download the data, run the ETL pipeline, and generate all visualizations.

<br>

## 7. Team Members & Roles

*   **Alona – Documentation & Presentation**
    
    *   Prepared project documentation and written deliverables
                
*   **Rapheal – Data Architecture & Preparation**
    
    *   Cleaned, transformed, and structured the raw dataset
                
*   **Tish – Data Analysis & Modelling**
    
    *   Performed exploratory data analysis using Pandas and NumPy
        
    *   Built visualisations and regression/classification models in notebooks
        
*   **Sam – Machine Learning & Application Development**
    
    *   Developed machine learning models to identify key predictors
        
    *   Built a Streamlit recommendation app showcasing the model and outputs
        
*   **Kanyin – Dashboarding & Visual Analytics**
    
    *   Designed and developed the Tableau dashboard
                
*   **Project Management (Shared Role)**
    
    *   Coordinated task planning, scheduling, and progress tracking
        
    *   Maintained team communication and updates via Discord
 
## 8. Credits

The below referecnes were used to support tensorflow NNP learning  and code development:

[Friend Spotify Recommender](https://www.kaggle.com/code/phrogmother/friend-spotify-recommender)

[Machine Learning models for song popularity](https://pcontreras97.medium.com/predicting-a-songs-success-using-machine-learning-376ae01b27de)

[Tensor Flow Explanations](https://www.kaggle.com/code/ramakrushnamohapatra/music-generation-using-tensorflow)


