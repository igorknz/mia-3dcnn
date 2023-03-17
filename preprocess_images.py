import os
import numpy as np
from scipy import ndimage
import cv2
import re
from tqdm import tqdm


def read_image_folder(folderpath):

    ct_names = os.listdir(folderpath)
    ct_names = [file_name.zfill(7) for file_name in ct_names]
    ct_names.sort()

    matrix = []

    for filename in ct_names:
        filepath = os.path.join(folderpath, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print('Wrong path:', filepath)
        else:
            img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
            matrix.append(img)

    matrix = np.transpose(matrix, [1,2,0])

    return np.array(matrix)


def normalize(matrix):
    return np.array(matrix/255.0, dtype=np.float32)


# method from: https://keras.io/examples/vision/3D_image_classification/
def resize_volume(img, desired_side=128, desired_depth=64):
    
    desired_width = desired_side
    desired_height = desired_side
    
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)

    return img


def preprocess_images(old_dataset_folder, new_dataset_folder, phase, label, new_side, new_depth):

    if os.path.isdir(new_dataset_folder) is not True:
        os.mkdir(new_dataset_folder)
        print(f'New folder: {new_dataset_folder}')
    if os.path.isdir(os.path.join(new_dataset_folder, phase)) is not True:
        os.mkdir(os.path.join(new_dataset_folder, phase))
        print(f'New folder: {os.path.join(new_dataset_folder, phase)}')
    if os.path.isdir(os.path.join(new_dataset_folder, phase, label)) is not True:
        os.mkdir(os.path.join(new_dataset_folder, phase, label))
        print(f'New folder: {os.path.join(new_dataset_folder, phase, label)}')

    old_label_folder = os.path.join(old_dataset_folder, phase, label)
    new_label_folder = os.path.join(new_dataset_folder, phase, label)

    list_ct_folders = os.listdir(old_label_folder)

    for ct_folder in list_ct_folders:
        if os.path.isdir(os.path.join(new_label_folder, ct_folder)) is not True:
            os.mkdir(os.path.join(new_label_folder, ct_folder))
            print(f'New folder: {os.path.join(new_label_folder, ct_folder)}')

        volume = read_image_folder(os.path.join(old_label_folder, ct_folder))
        volume = resize_volume(volume, new_side, new_depth)

        for i in range(volume.shape[2]):
            cv2.imwrite(os.path.join(new_label_folder, ct_folder, str(np.char.zfill(str(i),4))+'.jpg'), volume[:,:,i])


def change_names(dataset):
    for phase in ['train', 'val']:
        for label in ['covid', 'non-covid']:

            folder_names = os.listdir(os.path.join(dataset, phase, label))
            print(os.path.join(phase, label))
            for folder_name in tqdm(folder_names):
                names = os.listdir(os.path.join(dataset, phase, label, folder_name))
                for name in names:
                    old_name = os.path.join(dataset, phase, label, folder_name, name)
                    
                    name_list = name.split('.')
                    ext = name_list.pop(-1)

                    name_list = ''.join(name_list)
                    img_number = ''.join(re.findall('[0-9]*', name_list))
                    img_number = img_number.zfill(4)
                    img_name = '.'.join([img_number, ext])

                    new_name = os.path.join(dataset, phase, label, folder_name, img_name)
                    
                    os.replace(old_name, new_name)


def preprocess_images_test(old_dataset_folder, new_dataset_folder, new_side, new_depth):

    if os.path.isdir(new_dataset_folder) is not True:
        os.mkdir(new_dataset_folder)
        print(f'New folder: {new_dataset_folder}')

    list_ct_folders = os.listdir(old_dataset_folder)
    
    for ct_folder in list_ct_folders:
        
        if os.path.isdir(os.path.join(new_dataset_folder, ct_folder)) is not True:
            os.mkdir(os.path.join(new_dataset_folder, ct_folder))
            print(f'New folder: {os.path.join(new_dataset_folder, ct_folder)}')

        volume = read_image_folder(os.path.join(old_dataset_folder, ct_folder))
        volume = resize_volume(volume, new_side, new_depth)

        for i in range(volume.shape[2]):
            cv2.imwrite(os.path.join(new_dataset_folder, ct_folder, str(np.char.zfill(str(i),4))+'.jpg'), volume[:,:,i])


def change_names_test(dataset):

    folder_names = os.listdir(dataset)
    
    for folder_name in tqdm(folder_names):
        names = os.listdir(os.path.join(dataset, folder_name))
        
        for name in names:
            old_name = os.path.join(dataset, folder_name, name)
            
            name_list = name.split('.')
            ext = name_list.pop(-1)

            name_list = ''.join(name_list)
            img_number = ''.join(re.findall('[0-9]*', name_list))
            img_number = img_number.zfill(4)
            img_name = '.'.join([img_number, ext])

            new_name = os.path.join(dataset, folder_name, img_name)
            
            os.replace(old_name, new_name)
    

if __name__=='__main__':

    side = 224
    depth = 64

    original_folderpath = 'MIA2023_originals'
    new_folderpath = f'MIA2023_{side}x{side}x{depth}'

    original_folderpath_test = 'MIA2023_test_originals'
    new_folderpath_test = f'MIA2023_test_{side}x{side}x{depth}'

    change_names(original_folderpath)
    # change_names_test(original_folderpath_test)
    
    preprocess_images(original_folderpath, new_folderpath, 'train', 'covid', side, depth)
    preprocess_images(original_folderpath, new_folderpath, 'train', 'non-covid', side, depth)
    preprocess_images(original_folderpath, new_folderpath, 'val', 'covid', side, depth)
    preprocess_images(original_folderpath, new_folderpath, 'val', 'non-covid', side, depth)

    # preprocess_images_test(original_folderpath_test, new_folderpath_test, side, depth)
