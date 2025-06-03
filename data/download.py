import os
import requests
import tarfile

def download_and_extract(dataset_url, download_path):
    """
    Download a `.tar.gz` file from a given URL and extract it to the specified directory.
    :param dataset_url: URL of the dataset to download
    :param download_path: Directory to save and extract the dataset
    """
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Define the path for the tar file
    tar_file_path = os.path.join(download_path, "concept_image.tar.gz")

    # Download the dataset
    print(f"Downloading from {dataset_url} ...")
    try:
        # Send request to get the data
        response = requests.get(dataset_url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses

        # Write the content to the tar file
        with open(tar_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        print("Download completed!")

        # Extract the dataset
        print(f"Extracting {tar_file_path} ...")
        with tarfile.open(tar_file_path, 'r:gz') as tar:
            tar.extractall(path=download_path)

        print("Extraction completed!")

        # Remove the tar file (optional)
        os.remove(tar_file_path)
        print("Tar file removed!")

        return True

    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        return False

if __name__ == "__main__":
    dataset_url = "https://huggingface.co/datasets/Sakeoffellow001/T2i_Factualbench/resolve/main/concept_image.tar.gz"
    download_path = "./data"
    download_and_extract(dataset_url, download_path)
