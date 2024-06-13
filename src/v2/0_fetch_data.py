"""
This pulls training data from minio to local
"""
import os
from minio import Minio

client = Minio(os.environ['MINIO_SERVER'],
               access_key=os.environ['MINIO_ACCESS'],
               secret_key=os.environ['MINIO_SECRET'],
               cert_check=False,
               secure=False,)

bucket_name = "musicgen"

for item in client.list_objects(bucket_name, "",
                                recursive=True):
    client.fget_object(bucket_name, item.object_name,
                       f"models/{item.object_name}")
