import os
import torch
from google.cloud import storage
from google.oauth2 import service_account

def save_and_upload(embeddings, local_path, gcs_bucket_name, gcs_blob_name):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    # Save locally
    torch.save(embeddings, local_path)
    print(f"✅ Saved embeddings locally at: {local_path}")

    # Authenticate using service account file
    credentials = service_account.Credentials.from_service_account_file(
        "project888-29925-3295e984d06e.json"
    )

    # Upload to GCS
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_blob_name)
    blob.upload_from_filename(local_path)
    print(f"☁️ Uploaded embeddings to GCS at: {gcs_bucket_name}/{gcs_blob_name}")

def load_embeddings(local_path, gcs_bucket_name, gcs_blob_name):
    if os.path.exists(local_path):
        print("📂 Loading embeddings from local cache...")
        return torch.load(local_path)

    print("🔄 Local file not found. Attempting GCS download...")

    # Authenticate using service account file
    credentials = service_account.Credentials.from_service_account_file(
        "project888-29925-3295e984d06e.json"
    )

    client = storage.Client(credentials=credentials)
    bucket = client.bucket(gcs_bucket_name)
    blob = bucket.blob(gcs_blob_name)
    blob.download_to_filename(local_path)

    print("✅ Download complete. Loading embeddings...")
    return torch.load(local_path)
