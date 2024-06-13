import os
from minio import Minio

client = Minio(os.environ['MINIO_SERVER'],
               access_key=os.environ['MINIO_ACCESS'],
               secret_key=os.environ['MINIO_SECRET'],
               cert_check=False,
               secure=False,)


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


# for filename in os.listdir("./output"):
# print(filename)
filename = "music_gen.pt"
client.fput_object(bucket_name, filename, f"./{source_dir}/{filename}")

# client.fput_object(
#     bucket_name, "loss.png", "./output/loss.png"
# )
# client.fput_object(
#     bucket_name, "model.json", "./output/model.json",
# )

# for filename in os.listdir(source_dir):
#     print(filename)
#     client.fput_object(bucket_name, filename, f"./{source_dir}/{filename}")
