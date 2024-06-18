"""
This pulls training data from minio to local (within the docker container)
"""
import os
from minio import Minio

client = Minio(os.environ['MINIO_SERVER'],
               access_key=os.environ['MINIO_ACCESS'],
               secret_key=os.environ['MINIO_SECRET'],
               cert_check=False,
               secure=False,)

VERSION = os.environ['VERSION']
bucket_name = "musicgen"

objs = client.list_objects(bucket_name, prefix=f"{VERSION}/", recursive=True)
#  print(objs)
for item in objs:
    client.fget_object(bucket_name, item.object_name,
                       f"./models/{item.object_name}")
