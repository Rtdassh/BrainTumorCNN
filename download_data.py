import os
import urllib.request
import tarfile

DATA_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
DEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))

def download_and_extract():
    os.makedirs(DEST_DIR, exist_ok=True)
    tar_path = os.path.join(DEST_DIR, "Task01_BrainTumour.tar")
    if os.path.exists(tar_path):
        print(f"Tar file already exists at {tar_path}. Skipping download.")
    else:
        print(f"Downloading dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, tar_path)
        print("Download complete.")
        
    print("Extracting ...")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=DEST_DIR)
    print(f"Extraction complete. Files are in {DEST_DIR}")

if __name__ == "__main__":
    download_and_extract()

