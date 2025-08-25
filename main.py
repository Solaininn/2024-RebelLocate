from coordExtract import extract_coord, load_building
from KNearestN import image_knn
from CNN import train_cnn, predict_image

def main():
    # Building data and Image dir
    building_csv_path = '../../buildings.csv'
    label_csv_path = '/Zoli2/Zoli/! CSV/Label.csv'
    image_dir = '/Zoli2/Zoli'


    # Extract coords
    print ("Loading building data...")
    building_bounds = load_building(building_csv_path)
    print ("Extracting coordinates...")
    image_data = extract_coord(image_dir, building_bounds)

    # Print extracted coords
    print ("\nExtracted coordinates:")
    print (image_data[['file_path', 'lat', 'lon', 'building']].head())

    # KNN for nearest buildings
    print ("Running KNN model...")
    knn_results = image_knn(image_dir, building_csv_path)

    # Print KNN
    print ("\nKNN results:")
    print (knn_results[['file_path', 'nearest_building', 'distance_km']].head())

    # Train CNN
    print ("\nRunning CNN model...")
    model, label_mapping, epoch_results = train_cnn(image_dir, label_csv_path, epochs=10, batch_size=1, lr=1e-3)
    print ("CNN trained")

    # Print accuracy after each epoch
    for epoch, loss, accuracy in epoch_results:
        print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy * 100:.2f}%")

    sample_image = image_data['file_path'].iloc[0]
    predicted_building = predict_image(model, sample_image, label_mapping)
    print ("\nPredicted building: {predicted_building}".format(predicted_building=predicted_building))

if __name__ == '__main__':
    main()