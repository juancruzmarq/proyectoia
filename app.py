import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import cross_validate, train_test_split

# Función para cargar datos desde un archivo CSV
def load_data(file_path, use_cols=None):
    return pd.read_csv(file_path, sep=',', encoding='latin-1', usecols=use_cols, on_bad_lines='skip')

# Función para limpiar y preparar los datos de libros
def clean_books_data(df):
    required_cols = ['Title', 'authors', 'publishedDate', 'publisher']
    if set(required_cols).issubset(df.columns):
        return df[required_cols]
    else:
        print("Las columnas requeridas no están en el DataFrame 'books'.")
        return None

# Función para limpiar y preparar los datos de calificaciones
def clean_ratings_data(df):
    required_cols = ['User_id', 'Title', 'review/score']
    if set(required_cols).issubset(df.columns):
        df = df[df['review/score'] >= 1]
        return df[required_cols]
    else:
        print("Las columnas requeridas no están en el DataFrame 'ratings'.")
        return None


# Cargar datos
books = load_data('books_data.csv', ['Title', 'authors', 'publishedDate', 'publisher'])
ratings = load_data('Books_rating.csv', ['User_id', 'Title', 'review/score'])

# Limpiar datos
books = clean_books_data(books)
ratings = clean_ratings_data(ratings)

# Unir los DataFrames
data = pd.merge(ratings, books, on='Title')

# Filtrar usuarios y libros por número de calificaciones


def filter_data(df):
    user_counts = df['User_id'].value_counts()
    filtered_users = user_counts[user_counts >= 10].index
    df = df[df['User_id'].isin(filtered_users)]

    item_counts = df['Title'].value_counts()
    filtered_items = item_counts[item_counts >= 10].index
    df = df[df['Title'].isin(filtered_items)]
    return df


filtered_data = filter_data(data)

# Crear un dataset para Surprise
reader = Reader(rating_scale=(1, 5))
data_surprise = Dataset.load_from_df(filtered_data[['User_id', 'Title', 'review/score']], reader)

# Crear y evaluar el modelo SVD
svd = SVD(n_factors=150, lr_all=0.01, reg_all=0.02)
cross_validate(svd, data_surprise, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Entrenar el modelo con todo el dataset
trainset = data_surprise.build_full_trainset()
svd.fit(trainset)

# Función para recomendar libros
def recommend_books(user_id, num_recommendations=5):
    all_books = set(filtered_data['Title'].unique())
    read_books = set(filtered_data[filtered_data['User_id'] == user_id]['Title'].unique())
    books_to_predict = list(all_books - read_books)

    predictions = [svd.predict(user_id, book) for book in books_to_predict]
    predictions.sort(key=lambda x: x.est, reverse=True)

    recommended_books_details = [(pred.iid, books[books['Title'] == pred.iid]['authors'].iloc[0], pred.est)
                                 for pred in predictions[:num_recommendations]]
    return recommended_books_details


# Ejemplo de recomendación
user_id = 'A30TK6U7DNS82R'
recommendations = recommend_books(user_id)
print(recommendations)
