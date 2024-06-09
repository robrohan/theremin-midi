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

bucket_name = "robbie-v1.0.0"

# for bucket in client.list_buckets():
for item in client.list_objects(bucket_name, "robbie-v1.0.0/clean",
                                recursive=True):
    client.fget_object(bucket_name, item.object_name, item.object_name)
