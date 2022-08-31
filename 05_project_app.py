## imports
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# from ml_reco import recoMovieModelKNN_train
# from ml_reco import recoMovieModelKNN

## import de la database de films
link = r"https://raw.githubusercontent.com/deconiak/streamlit_proj/main/movie_df.csv"
movie_df = pd.read_csv(link)

## fonction pour machine learning
def recoMovieModelKNN_train(dataframe):
    # definition de X
    X = dataframe.select_dtypes("number")

    # standardisation de X
    scaler = StandardScaler().fit(X)
    scaled_data = pd.DataFrame(scaler.transform(X), index=X.index, columns=X.columns)
    
    # definition et entrainement du modèle
    model = NearestNeighbors().fit(scaled_data)
    return model, scaled_data


def recoMovieModelKNN(dataframe, model, scaled_data, index_film, k):
    
    # valeur de X pour film concerné
    film_concerne_scaled = scaled_data.loc[index_film].to_frame().T
    
    neigh_dist, neigh_idx = model.kneighbors(film_concerne_scaled, n_neighbors=k)
    film_ressem = neigh_idx[0][1:]
    
    # affichage des recommandations
    return dataframe.loc[film_ressem]

## entrainement du modèle
modelKNN, X_scaled = recoMovieModelKNN_train(movie_df)


## Page configuration
st.set_page_config(
     page_title="Sceno-Film App",
     page_icon="random",
     menu_items={
         'Get help': "https://discuss.streamlit.io/",
         'Report a bug': "https://github.com/deconiak/streamlit_proj/issues",
         'About': "This is an app realized by Camille, Emeline, José and Noura as part of a project with Wild Code School. Datasets used can be found at https://datasets.imdbws.com/ and documentation at https://www.imdb.com/interfaces/ "
     }
 )


## mise en page de l'appli
st.header("Système de recommandation de films")


# Sélecteur de page dans la variable select
select = st.selectbox('Selectionnez une rubrique', 
             ("Introduction","Sélection des films","KPI et Viz","Moteur de recommandations"))
## 

## fonction pour les pages 

def intro():
    st.markdown("_Il était une fois un cinéma de la Creuse._")
    st.write("""En perte de vitesse, il a décidé de passer le cap du digital en créant un site internet taillé pour ses clients locaux. Il souhaite proposer sur son site internet un moteur de recommandations de films à l'instar de grands noms du streaming.""")
    st.write("""Notre équipe Scéno-film est contactée. Le client n'ayant renseigné aucune de ses préférences, nous sommes dans une situation de cold-start. Nous récupérons un dataset de films depuis le site IMDB et procédons en trois étapes. """)

    st.write("1) Récupération d'une sélection comportant 50 000 à 100 000 films")
    st.write("2) Analyse de la sélection de films")
    st.write("3) Moteur de recommandation")

    st.write("""Sélectionnez une rubrique dans le menu pour découvrir les différentes étapes du projet.""")

    st.image(
            "https://upload.wikimedia.org/wikipedia/commons/9/93/Paris_arthouse_cinema_seats.jpg",
            caption='Kotivalo, CC BY-SA 4.0, via Wikimedia Commons',
            width=400
        )


def selec():
    return "Selec de film bâtard"

def viz(): # Partie 2 du notebook 2 KPI


    st.header("2. Informations clés à partir des données de notre sélection")
    st.subheader("2.1 Notation : Top 10 des films les mieux notés")

    # création de la condition de filtrage : tous les films ayant un nombre de votes supérieur à la moyenne
    cond_moy = movie_df["numVotes"] > movie_df["numVotes"].mean()
    #nouveau df des films les plus votés
    film_pop = movie_df[cond_moy]
    #Top 10 des films les mieux notés
    st.dataframe(film_pop.sort_values(by=["averageRating"], ascending = False).head(10).iloc[:,:6])

    st.subheader("2.2 Nombre de votes en fonction de l'année de sortie des films")

    # Visualisation graphique du nombre de votes, en fonction de l'année de réalisation du film

    test = movie_df.plot.scatter(x='releaseYear',y='numVotes',figsize=(15,8))
    plt.ticklabel_format(style='plain')
    plt.title("Nombre de votes par années")
    plt.xlabel('Année de sortie') 
    plt.ylabel('Nombre de votes') 
    plt.show()
    st.pyplot(test.figure,clear_figure=True)


    st.write("""* Le nombre de votes augmente progressivement entre les films les plus anciens et les plus récents. 

    * Cela s'interprète par une plus grande popularité des films récents, les utilisateurs ont tendance à plus noter des films récemment sortis.""")

    # Test d'une viz avec plotly

    test1 = px.scatter(movie_df, x='releaseYear',y='numVotes')
    plt.ticklabel_format(style='plain')
    plt.title("Nombre de votes par années")
    plt.xlabel('Année de sortie') 
    plt.ylabel('Nombre de votes') 
    st.plotly_chart(test1)

    st.subheader("2.3 Moyenne des notes et nombre de votes")

    # Médiane et moyenne de la moyenne des notes (averageRating)
    # Création de df 
    ratings_med = movie_df['averageRating'].median()
    ratings_mean = movie_df['averageRating'].mean()

    # Médiane et moyenne du nombre de votes (numVotes) sotockée respectivement dans une dataframe
    numvotes_med = movie_df['numVotes'].median()
    numvotes_mean = movie_df['numVotes'].mean()

    # Visualisation comparative entre médiane et moyenne
    fig, ax = plt.subplots(figsize=(20,5))

    # Graphe pour la moyenne des notes
    ax1 = plt.subplot(121)
    ax1.bar(range(2), [ratings_med, ratings_mean], tick_label = ['Mediane', 'Moyenne'])
    ax1.set_title('Moyenne des notes') 
    ax1.set_ylabel('Notes') 

    # Graphe pour le nombre de votes
    ax2 = plt.subplot(122)
    ax2.bar(range(2), [numvotes_med, numvotes_mean], tick_label = ['Mediane', 'Moyenne'])
    ax2.set_title('Nombre de votes') 
    ax2.set_ylabel('Compte') 

    plt.suptitle('Comparaison moyenne et médiane')
    plt.show()

    st.pyplot(ax1.figure, clear_figure=True)

    st.write("""* La moyenne et la médiane des notes accordées aux films sont similaires. Ceci est partiellement lié, au fait que les notes sont forcément entre 0 et 10.

    * Pour le nombre de votes, l'écart entre la médiane et la moyenne est très important. 

    * La moyenne est très influencée par les films extrêmement connus qui recueillent un grand nombre de votes.""")

    # Visualisation des outliers sur un même plan
    fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(15, 10)) 
    # sur la moyenne des notes
    ax1.boxplot(movie_df["averageRating"])
    ax1.set_title("Note moyenne")
    # sur le nombre de votes
    ax2.boxplot(movie_df["numVotes"])
    ax2.set_title("Nombre de votes")

    plt.show()
    st.pyplot(ax1.figure, clear_figure=True)

    st.write("* Des outliers amplifient la moyenne de la moyenne des notes et celle du nombre de votes.")

    st.subheader("2.4 Durée des films")

    st.subheader("2.4.1. Statistiques sur la durée des films")


    st.write(movie_df["runtimeMinutes"].describe())
    st.write("""La moyenne et la médiane sont assez proches, donc le dataset assez homogène. 

    * Une durée maximum très élevée, il y a probablement quelques outliers de ce côté.

    * Un dataset avec un écart interquartile assez restreint.

    * 50% des films font entre 84 et 104 minutes.""") 

    # Vérification de la présence d'outliers
    # Par création d'un boxplot sur runtimeMinutes
    fig = plt.figure(figsize =(3, 2))
    plt.boxplot(movie_df["runtimeMinutes"])
    plt.title("Durée des films")
    plt.show()
    st.pyplot(fig.figure, clear_figure=True)

    st.write("* Confirmation de présence d'outliers sur la limite supérieure.")

    st.subheader("2.4.2. Répartition de la durée des films")

    # Proportion de films ayant une durée > 200 minutes
    pct_movie_over_200 = len(movie_df[movie_df['runtimeMinutes'] > 200]) / len(movie_df) * 100
    print(f"Il y a {round(pct_movie_over_200,1)} % de films d'une durée supérieure à 200 minutes dans notre dataframe")


    #triche

    st.write("Il y a 0.4 % de films d'une durée supérieure à 200 minutes dans notre dataframe")

    # Visualisation graphique
    viz_dur_movie = plt.figure(figsize=(10,5))
    plt.hist(movie_df['runtimeMinutes'], range=(40, 201), bins=16, ec='black')
    plt.title('Répartition de la durée des films')
    plt.xlabel('Durée en minutes')
    plt.ylabel('Nombre de films')
    plt.show()
    st.pyplot(viz_dur_movie.figure, clear_figure=True)

    #viz_dur_movie_plotly = px.bar(movie_df['runtimeMinutes']) 
    #fig.show()
    #st.plotly_chart(viz_dur_movie)

    st.write("- La durée la plus fréquente, se situe entre 90 et 100 minutes (médiane=93).")
    st.write("- Le deuxième intervalle qui apparait le plus va de 80 à 90 minutes.")
    st.write("- L'intervalle de 80 à 100 correspond bien à notre écart inter-quartile (Q3-Q1=20).") 
    st.write("- On peut également noter l'intervalle entre 100 et 110 minutes.")


    st.subheader("2.4.3. Film le plus long et film le plus court de la sélection et de l'année")

    st.write("- le plus long de la sélection")

    st.dataframe(movie_df.iloc[movie_df["runtimeMinutes"].argmax()].to_frame().transpose())

    st.image("https://www.justfocus.fr/wp-content/uploads/2021/02/a-1175173_10151675518532075_1799923040_n.jpg",width=400)

    st.write("- le plus long de l'année en cours")

    annee_en_cours=pd.datetime.now().year

    max_runtime = movie_df[movie_df["releaseYear"] == annee_en_cours]['runtimeMinutes'].max()
    st.dataframe(movie_df[(movie_df["releaseYear"] == annee_en_cours) & (movie_df['runtimeMinutes']==max_runtime)].iloc[:,:6])

    st.image("https://www.francetvinfo.fr/pictures/-pWGLGU7Ujlqtdi4-4hPaA4ui24/0x0:600x800/fit-in/720x/filters:format(webp)/2022/05/19/phpJle2ZV.jpg", width=400)

    st.write("- le plus court de la sélection")

    st.dataframe(movie_df.iloc[movie_df["runtimeMinutes"].argmin()].to_frame().transpose())

    st.image("https://media.senscritique.com/media/000019027939/300/les_nazis_attaquent.jpg", width=400)

    st.write("- le plus court de l'année en cours")

    min_runtime = movie_df[movie_df["releaseYear"] == annee_en_cours]['runtimeMinutes'].min()
    le_plus_court_an = movie_df[(movie_df["releaseYear"] == annee_en_cours) & (movie_df['runtimeMinutes']==min_runtime)]

    st.dataframe(le_plus_court_an.iloc[:,:6])

    st.image("https://m.media-amazon.com/images/M/MV5BYjA1YWRkNWUtYTgxMy00OWFmLWFiODItOTFjYWU5N2RiMTYwXkEyXkFqcGdeQXVyMjIwNjIxNjc@._V1_FMjpg_UX1000_.jpg",width=400)



def recommandations(): # page avec l'outil de recommandation

    movie_titles = movie_df['titleFR'].unique()
    user_input = st.selectbox('Choisissez un film', movie_titles)

    nb_movies = st.slider(label="Nombre de recommandations à afficher", min_value=1, max_value=10)

    if st.button('Show Recommendation'):
        user_input_idx = movie_df.loc[movie_df['titleFR'] == user_input].index.values[0]
        reco = recoMovieModelKNN(movie_df, modelKNN, X_scaled, user_input_idx, nb_movies+1) 
        # afficher la recommendation et quelques colonnes clés
        st.dataframe(reco.loc[:, ['titleFR', 'releaseYear', 'runtimeMinutes', 'averageRating',
            'numVotes', 'directorName', 'actor', 'notReleasedYet']])


# Choisir les pages
if select == "Introduction" : 
    intro()
    st.caption("""Le saviez-vous ? Le chat du "Parrain" était en fait un chat errant trouvé à l'extérieur du studio.""")

elif select == "Sélection des films" : 
    selec()
    st.caption("""Le saviez-vous ? Woody de "Toy Story" devait à l'origine être un mannequin ventriloque.""")

elif select == "KPI et Viz" :
    viz()
    st.caption("""Le saviez-vous ? Il y a une tasse Starbucks dans chaque scène de "Fight Club". """)

elif select == "Moteur de recommandations" :
    recommandations()
    st.caption("""Le saviez-vous ? Le code de "Matrix" provient de recettes de sushis.""")