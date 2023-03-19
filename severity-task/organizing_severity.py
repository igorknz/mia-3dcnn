import pandas as pd
import os
import shutil


train_folders = pd.read_csv('/home/lovelace/proj/proj882/givendra/MIA-challenge-2023/severity/ICASSP_severity_train_partition.csv', sep=',')
val_folders = pd.read_csv('/home/lovelace/proj/proj882/givendra/MIA-challenge-2023/severity/ICASSP_severity_validation_partition.csv', sep=',')

old_folder = '/home/lovelace/proj/proj882/givendra/MIA-challenge-2023/dataset/MIA2023_224x224x64/train/covid'
new_folder = '/home/lovelace/proj/proj882/givendra/MIA-challenge-2023/dataset/MIA_sev_224x224x64/train'

for cls in ['1','2','3','4']:
    print(cls)
    for name in train_folders[train_folders.Category==int(cls)].Name:
        old_path = os.path.join(old_folder, name)
        new_path = os.path.join(new_folder, cls, name)
        #os.rename(old_path, new_path)
        shutil.copytree(old_path, new_path)
        print(name)

old_folder = '/home/lovelace/proj/proj882/givendra/MIA-challenge-2023/dataset/MIA2023_224x224x64/val/covid'
new_folder = '/home/lovelace/proj/proj882/givendra/MIA-challenge-2023/dataset/MIA_sev_224x224x64/val'

for cls in ['1','2','3','4']:
    print(cls)
    for name in val_folders[val_folders.Category==int(cls)].Name:
        old_path = os.path.join(old_folder, name)
        new_path = os.path.join(new_folder, cls, name)
        #os.rename(old_path, new_path)
        shutil.copytree(old_path, new_path)
        print(name)
