from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd


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