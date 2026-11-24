import os
import requests
import zipfile

# Configuration
url = "https://data.cityofnewyork.us/api/views/n6j3-6w8k/exports/download?format=zip"
download_path = os.path.expanduser("~/Downloads/lidar_data.zip")
target_dir = "/home/zia207/Github/GeoLattice/Data/08_remote_sensing/08_06_lidar/"

# Ensure the target directory exists
os.makedirs(target_dir, exist_ok=True)

try:
    # Download the file
    print("Downloading LiDAR data...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for HTTP errors

    # Save the zip file
    with open(download_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)

    # Extract the zip file
    print("Extracting LiDAR data...")
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(download_path))

    # Move LAS/LAZ files to the target directory
    print("Moving files to target directory...")
    source_dir = os.path.dirname(download_path)
    for filename in os.listdir(source_dir):
        if filename.endswith(".las") or filename.endswith(".laz"):
            src = os.path.join(source_dir, filename)
            dst = os.path.join(target_dir, filename)
            os.rename(src, dst)

    print("✅ Done. Files saved to:", target_dir)

except Exception as e:
    print("❌ Error:", str(e))
