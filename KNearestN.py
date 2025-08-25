import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from coordExtract import extract_coord
from coordExtract import load_building


def knn_model(building_csv_path):
    # Get coords from CSV
    df = pd.read_csv(building_csv_path)

    # Check if 'Middle Coordinate' exists
    if 'Middle Coordinate' not in df.columns:
        print("Error: 'Middle Coordinate' column missing!")
        return None, None

    # Remove leading/trailing spaces and handle any misformatted coordinates
    df['Middle Coordinate'] = df['Middle Coordinate'].str.replace(' ', '')  # Remove spaces

    # Parse middle coordinate
    try:
        df[['lat', 'lon']] = df['Middle Coordinate'].str.split(',', expand=True).astype(float)
    except ValueError as e:
        print(f"Error parsing coordinates: {e}")
        return None, None

    # Drop rows with missing lat or lon
    df = df.dropna(subset=['lat', 'lon'])

    # Rename columns for clarity
    df = df.rename(columns={'Acronym': 'building'})

    return df[['lat', 'lon']].values, df['building'].values


def knn_cross_validation(building_csv_path, n_splits=5, max_neighbors=10):
    X,y = knn_model(building_csv_path)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for k in range(1, max_neighbors + 1):
        fold_accuracies = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            knn = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
            knn.fit(X_train, y_train)
            accuracy = knn.score(X_test, y_test)
            fold_accuracies.append(accuracy)

        mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
        accuracies.append({
            'k_neighbors': k,
            'mean_accuracy': mean_accuracy,
        })

    return pd.DataFrame(accuracies)

def image_knn(image_dir, building_csv_path, n_neighbors=1):
    # Finds nearest building to image using KNN
    X, y = knn_model(building_csv_path)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(X, y)

    building_bounds = load_building(building_csv_path)

    image_df = extract_coord(image_dir, building_bounds)

    results = []
    for _, row in image_df.iterrows():
        pred = knn.kneighbors([[row['lat'], row['lon']]])[0]  # Reshaping the coordinates to 2D
        distance_km = knn.kneighbors([[row['lat'], row['lon']]], return_distance=True)[0][0][0] * 6371

        results.append({
            'file_path': row['file_path'],
            'lat': row['lat'],
            'lon': row['lon'],
            'nearest_building': pred,
            'distance_km': distance_km,
        })

    return pd.DataFrame(results)

