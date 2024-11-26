import gdown

link = "https://drive.google.com/drive/folders/1krim1ugqNDmgmHZCFSOvmynWxCSzyOto"
target_dir = "downloaded_data_zips"
gdown.download_folder(link, output=target_dir)
