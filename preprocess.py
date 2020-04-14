
import shutil
import os
from pathlib import Path
from tqdm import tqdm

def generate_dataset(src, dst, meta_file):

    lines= open(meta_file).readlines()
    for line in tqdm(lines):
        folder, img = line.strip().split('_')
        folder = folder.split('/')[0]
        nfolder = str(os.path.join(dst,folder))
        if not os.path.exists(nfolder):
            os.makedirs(nfolder)
        shutil.copyfile(os.path.join(src,folder,img), os.path.join(dst, nfolder, img))


if __name__=='__main__':
    generate_dataset('faces_emore/imgs', 'emore/trainset', 'lists/part0_train.list')
    print('finishing training parts')

    for i in tqdm(range(1,10)):
        generate_dataset('faces_emore/imgs', 'emore/testset/split_%d'%i, 'lists/part%s_test.list'%i)
        print('finishing testing parts: split %d'%i)

