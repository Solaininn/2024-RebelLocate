from exif import Image
import os
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image as PilImage
import numpy as np

def parse_coordinate_pair(coord_pair):
    # Parse string and turn into float
    if coord_pair is None or not isinstance(coord_pair, str):
        return None
    try:
        lat, lon = map(float, coord_pair.split(','))
        return lat, lon
    except (ValueError, TypeError):
        return None

def load_building(csv_path):
    # Convert CSV bounds to polygon
    df = pd.read_csv(csv_path)
    building_bounds = {}

    for _, row in df.iterrows():
        try:
            coords = [
                # Parse each coordinate pair
                parse_coordinate_pair(row['Top Left Coordinate']),
                parse_coordinate_pair(row['Top Right Coordinate']),
                parse_coordinate_pair(row['Bottom Right Coordinate']),
                parse_coordinate_pair(row['Bottom Left Coordinate']),
            ]

            if any(c is None for c in coords):
                print('Skipping {}'.format(row['Acronym']))
                continue

            lats = [lat for lat, lon in coords]
            lons = [lon for lat, lon in coords]

            building_bounds[row['Acronym']] = {
                'min_lat': min(lats),
                'max_lat': max(lats),
                'min_lon': min(lons),
                'max_lon': max(lons),
            }
        except Exception as e:
            print(f"Skipping {row['Acronym']}: {e}")
    return building_bounds

def convert_coordinates(coordinates, ref):
    # Convert coordinates from D/M/S notation into decimal.
    decimal = coordinates[0] + coordinates[1]/60 + coordinates[2]/3600
    return -decimal if ref in ['S', 'W'] else decimal

def find_building(lon, lat, building_bounds):
    # Checks which building boundary contains image coordinates
    for acronym, bounds in building_bounds.items():
        if (bounds['min_lat'] <= lat <= bounds['max_lat'] and
        bounds['min_lon'] <= lon <= bounds['max_lon']):
            return acronym
    return None

def extract_coord(image_dir, building_bounds):
    image_data = []
    # Extract all files from root folder
    for root, _, files in os.walk(image_dir):
        # Extract images from subdirectory.
        for file in files:
            # Images must end with .jpg/.jpeg.
            if not file.lower().endswith(('.jpg', '.jpeg')):
                continue

            # Open each image.
            file_path = os.path.join(root, file)
            with open(file_path, 'rb') as image_file:
                my_image = Image(image_file)

            # Check if EXIF data exists
            if not my_image.has_exif:
                continue

            # Extract coordinates from each image.
            try:
                # Convert coordinates from D/M/S notation into decimal.
                lat = convert_coordinates(my_image.gps_latitude, my_image.gps_latitude_ref)
                lon = convert_coordinates(my_image.gps_longitude, my_image.gps_longitude_ref)

                building = find_building(lon, lat, building_bounds)
                image_data.append({
                    'file_path': file_path,
                    'building': building,
                    'lat': lat,
                    'lon': lon
                })

            # Images didn't have coordinate metadata.
            except AttributeError:
                print(f"ERROR: {file_path} doesn't have coordinate metadata.")
                continue

    return pd.DataFrame(image_data)

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

    if X is None or y is None or len(y) < 2:
        return None

    unique, counts = np.unique(y, return_counts=True)
    min_class_samples = counts.min()

    if n_splits > min_class_samples:
        n_splits = min_class_samples

    if n_splits < 2:
        return None

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
        distances, indices = knn.kneighbors([[row['lat'], row['lon']]], return_distance=True)
        nearest_building = knn.predict([[row['lat'], row['lon']]])[0]
        distance_km = distances[0][0] * 6371

        results.append({
            'file_path': row['file_path'],
            'lat': row['lat'],
            'lon': row['lon'],
            'nearest_building': nearest_building,
            'distance_km': distance_km,
        })

    return pd.DataFrame(results)

def create_cnn(num_classes):
    # Load pre-trained ResNet
    model = models.resnet18(weights='IMAGENET1K_V1')

    # Freeze layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace final full layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Only fc params use gradients
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

def get_optimizer(model, lr=1e-3):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    return optimizer

class BuildingDataset(Dataset):
    def __init__(self, image_dir, label_df, transform=None):
        self.image_dir = image_dir
        self.label_df = label_df
        self.transform = transform
        self.image_files = []

        self.img_label_mapping = {row['img_name']: row['label'] for _, row in self.label_df.iterrows()}
        self.label_mapping = {label: idx for idx, label in enumerate(self.label_df['label'].unique())}

        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg')):
                    label = self.img_label_mapping.get(file, None)  # None if not found
                    if label is not None and label in self.label_mapping:
                        self.image_files.append(os.path.join(root, file))

        print(f"Found {len(self.image_files)} images in {image_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = PilImage.open(image_path).convert('RGB')

        # Extract the image name from the file path
        image_name = os.path.basename(image_path)  # <-- this was missing!

        # Now get the label
        label = self.img_label_mapping.get(image_name, None)

        # Skip images with invalid or missing labels
        if label is None or label not in self.label_mapping:
            return self.__getitem__((idx + 1) % len(self))  # Try the next item

        label_idx = self.label_mapping[label]

        if self.transform:
            image = self.transform(image)

        return image, label_idx

def train_cnn(image_dir,labels_csv_path, epochs=10, batch_size=8, lr=1e-3):
    # Extract image data
    label_df = pd.read_csv(labels_csv_path)

    # Build Label map
    label_mapping = {label: idx for idx, label in enumerate(label_df['label'].unique())}
    num_classes = len(label_mapping)

    # Dataset
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = BuildingDataset(image_dir, label_df, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Model
    device = torch.device('cpu')
    model = create_cnn(num_classes).to(device)
    optimizer = get_optimizer(model, lr)
    criterion = nn.CrossEntropyLoss()

    epoch_results = []

    # Training
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        accumulation_steps = 4  # Number of steps to accumulate gradients

        for step, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Accumulate gradients and update weights every `accumulation_steps` steps
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = correct / total

        epoch_results.append((epoch + 1, epoch_loss, epoch_accuracy))
    return model, label_mapping, epoch_results

def predict_image(model, image_path, label_mapping):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    image = PilImage.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image = image.to(device)
    model = model.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    idx_to_label = {idx: label for label, idx in label_mapping.items()}
    predicted_label = idx_to_label[predicted.item()]
    return predicted_label


def main():
    # Building data and Image dir
    building_csv_path = '../../buildings.csv'
    label_csv_path = '/Users/zolile/Documents/CS422/RebelLocateVirt/.venv/Zoli2/Zoli/! CSV/Label.csv'
    image_dir = '/Users/zolile/Documents/CS422/RebelLocateVirt/.venv/Zoli2/Zoli'

    # Extract coords
    print ("Loading building data...")
    building_bounds = load_building(building_csv_path)

    print ("Extracting coordinates...")
    image_data = extract_coord(image_dir, building_bounds)
    random_row = image_data.sample(1).iloc[0]
    sample_image = random_row['file_path']

    # Print extracted coords
    print ("Extracted coordinates:")
    print (image_data[['file_path', 'lat', 'lon', 'building']].head())

    #KNN Cross for Accuracy
    print("\nRunning KNN Cross-Validation...")
    knn_cv = knn_cross_validation(building_csv_path, n_splits=5, max_neighbors=10)

    if knn_cv is not None:
        best_k_row = knn_cv.loc[knn_cv['mean_accuracy'].idxmax()]
        best_k = int(best_k_row['k_neighbors'])
        best_accuracy = best_k_row['mean_accuracy']
        print(f"\nBest K: {best_k} neighbors | Cross-Validation Accuracy: {best_accuracy * 100:.2f}%")
    else:
        best_k = 1  # default to 1 neighbor
        best_accuracy = None

    # KNN for nearest buildings
    print ("Running KNN model...")
    knn_results = image_knn(image_dir, building_csv_path, n_neighbors=best_k)

    # Find prediction for sample image
    sample_row = knn_results[knn_results['file_path'] == sample_image].iloc[0]
    sample_nearest_building = sample_row['nearest_building']
    distance_km = sample_row['distance_km']

    # Print KNN
    print(f"\nImage name: {os.path.basename(sample_image)}")
    print(f"Nearest building (KNN): {sample_nearest_building}")
    print(f"Distance (km): {distance_km:.2f} km")

    # Train CNN
    print ("\nRunning CNN model...")
    model, label_mapping, epoch_results = train_cnn(image_dir, label_csv_path, epochs=10, batch_size=8, lr=1e-3)
    print ("CNN trained")

    # Print accuracy after each epoch
    for epoch, loss, accuracy in epoch_results:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy * 100:.2f}%")

    predicted_label = predict_image(model, sample_image, label_mapping)
    real_building_row = image_data[image_data['file_path'] == sample_image].iloc[0]
    real_building = real_building_row['building']

    # CNN Final Accuracy (last epoch's accuracy)
    cnn_final_epoch = epoch_results[-1]  # Get last epoch
    cnn_final_accuracy = cnn_final_epoch[2]  # (epoch, loss, accuracy

    print(f"\nFinal Results for {os.path.basename(sample_image)}:")
    print(f"Coordinates are {real_building_row['lat']}, {real_building_row['lon']}.")

    if best_accuracy is not None:
        print(f"KNN Cross-Validation Accuracy: {best_accuracy*100:.2f}%")

    print(f"CNN Final Accuracy: {cnn_final_accuracy * 100:.2f}%")
    print(f"\nPredicted building: {sample_nearest_building}")
    print(f"Predicted Label: {predicted_label}")
    print(f"Real building {real_building}")


if __name__ == '__main__':
    main()