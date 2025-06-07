import os

import requests
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor


def download_file(url, local_path, chunk_size=8192):
    """
    Downloads a file from a URL to a local path with progress indication.
    """
    print(f"Downloading {url} to {local_path}")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"Downloaded: {progress:.1f}%", end='\r')

        print(f"\nDownload completed: {local_path}")
        return True

    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return False


def download_colpali_models():
    """
    Downloads ColPali models and processors using HuggingFace transformers
    """
    print("Downloading ColPali models...")

    # Available ColPali models
    models = {
        # "colpali-v1.3": "vidore/colpali-v1.3",
        # "colqwen2-v1.0": "vidore/colqwen2-v1.0",
        "colqwen2.5-v0.2": "vidore/colqwen2.5-v0.2",
        # "colSmol-500M": "vidore/colSmol-500M",
    }

    # Default model to download
    default_model = "vidore/colpali-v1.3"

    downloaded_models = []

    for model_name, model_id in models.items():
        try:
            print(f"\nDownloading {model_name} ({model_id})...")

            # Download model
            print(f"Downloading model weights for {model_id}...")
            model = ColQwen2_5.from_pretrained(
                model_id,
                local_files_only=False,
                trust_remote_code=True
            )
            print(f"Model {model_name} downloaded successfully")

            # Download processor
            print(f"Downloading processor for {model_id}...")
            processor = ColQwen2_5_Processor.from_pretrained(
                model_id,
                local_files_only=False,
                trust_remote_code=True
            )
            print(f"Processor for {model_name} downloaded successfully")

            downloaded_models.append({
                "name": model_name,
                "model_id": model_id
            })

            # Only download the default model for now to save space
            if model_id == default_model:
                print(f"Downloaded default model: {model_name}")
                break

        except Exception as e:
            print(f"Error downloading {model_name}: {str(e)}")
            continue

    return downloaded_models


def verify_installation():
    """
    Verify that all components are installed correctly
    """
    print("\nVerifying installation...")

    try:
        # Test imports
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA devices: {torch.cuda.device_count()}")

        import transformers
        print(f"✓ Transformers version: {transformers.__version__}")

        from colpali_engine.models import ColPali, ColPaliProcessor
        print("✓ ColPali engine imported successfully")

        from transformers.utils.import_utils import is_flash_attn_2_available
        print(f"✓ Flash Attention 2 available: {is_flash_attn_2_available()}")

        import runpod
        print(f"✓ RunPod SDK imported successfully")

        print("\n✅ All components verified successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


def get_colpali_models():
    """
    Main function to download and setup ColPali models and dependencies
    """
    print("=== ColPali Model Download Script ===\n")

    # Download models
    downloaded_models = download_colpali_models()

    # Verify installation
    if verify_installation():
        print("\n🎉 ColPali setup completed successfully!")

        print("\nDownloaded models:")
        for model in downloaded_models:
            print(f"  - {model['name']}")

        print("\nYou can now run the ColPali RunPod worker!")

    else:
        print("\n❌ Setup completed with errors. Please check the output above.")

    return downloaded_models


if __name__ == "__main__":
    get_colpali_models()
