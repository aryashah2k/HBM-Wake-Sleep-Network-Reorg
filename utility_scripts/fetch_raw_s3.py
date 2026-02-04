import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

# Configuration for OpenNeuro's public S3 bucket
BUCKET_NAME = 'openneuro.org'
S3_PREFIX = 'ds003768/'
LOCAL_DIR = 'ds003768/'

# Initialize S3 client without credentials (No-Sign-Request)
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

def download_s3_folder(bucket, prefix, local_path):
    # List all objects in the bucket with the specified prefix
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            s3_key = obj['Key']
            file_size = obj['Size']
            
            # Create local file path
            relative_path = os.path.relpath(s3_key, prefix)
            local_file_path = os.path.join(local_path, relative_path)
            
            # Create subdirectories if they don't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Skip directories (keys ending in '/')
            if s3_key.endswith('/'):
                continue
            
            # Download file with a progress bar
            with tqdm(total=file_size, unit='B', unit_scale=True, desc=relative_path) as pbar:
                s3.download_file(
                    Bucket=bucket, 
                    Key=s3_key, 
                    Filename=local_file_path,
                    Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                )

if __name__ == "__main__":
    print(f"Starting download of {S3_PREFIX} to {LOCAL_DIR}...")
    download_s3_folder(BUCKET_NAME, S3_PREFIX, LOCAL_DIR)
    print("\nDownload complete.")
