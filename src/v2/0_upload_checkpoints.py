import os
from minio import Minio

client = Minio(os.environ['MINIO_SERVER'],
               access_key=os.environ['MINIO_ACCESS'],
               secret_key=os.environ['MINIO_SECRET'],
               cert_check=False,
               secure=False,)


VERSION = os.environ['VERSION']
bucket_name = "musicgen"

# The file to upload, change this path if needed
source_dir = "./models"

# Make the bucket if it doesn't exist.
found = client.bucket_exists(bucket_name)
if not found:
    client.make_bucket(bucket_name)
    print("Created bucket", bucket_name)
else:
    print("Bucket", bucket_name, "already exists")


filename = "theremin.pt"
client.fput_object(bucket_name, f"{VERSION}/{filename}",
                   f"./{source_dir}/{filename}")
