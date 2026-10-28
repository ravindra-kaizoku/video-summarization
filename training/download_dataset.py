import kagglehub

path = kagglehub.dataset_download(
    "python16/summe-videos-and-tvsum-videos/versions/1"
)

print("Path to dataset files:", path)