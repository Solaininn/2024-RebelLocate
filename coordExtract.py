from exif import Image
import os
import pandas as pd

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