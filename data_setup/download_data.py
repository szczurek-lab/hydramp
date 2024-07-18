import gdown

link = "https://drive.google.com/drive/folders/1IwJjwSKpXYWWALKXM8nghqmlSvyu3BYq"
gdown.download_folder(link, quiet=True)
