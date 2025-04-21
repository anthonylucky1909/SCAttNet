import kagglehub
from pathlib import Path

def download_datasets():
    """Download all required datasets from Kaggle Hub"""
    datasets = {
        "div2k": "sharansmenon/div2k",
        "set5_14": "ll01dm/set-5-14-super-resolution-dataset",
        "manga109": "guansuo/manga109",
        "urban100": "harshraone/urban100",
        "bsd100": "asilva1691/bsd100"
    }
    
    dataset_paths = {}
    
    print("Starting dataset downloads...")
    for name, dataset_id in datasets.items():
        try:
            print(f"Downloading {name} dataset...")
            path = kagglehub.dataset_download(dataset_id)
            dataset_paths[name] = str(Path(path).resolve())  # Convert to absolute path
            print(f"Successfully downloaded {name} to: {path}")
        except Exception as e:
            print(f"Failed to download {name}: {str(e)}")
            dataset_paths[name] = None
    
    print("\nDownload summary:")
    for name, path in dataset_paths.items():
        print(f"{name.upper():<10}: {path if path else 'Download failed'}")
    
    return dataset_paths

if __name__ == "__main__":
    download_datasets()