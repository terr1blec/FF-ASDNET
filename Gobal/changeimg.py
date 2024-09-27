import cv2
import os

path1='14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_2/cam'
newpath1='14resnetmm_3fc_rgb17wos4/d0.5_n0.5_shuffle_3/cam'

for i in range(5):
    if not os.path.exists(f'{newpath1}/{i}'):
        os.makedirs(f'{newpath1}/{i}')
    medias=os.listdir(f'{path1}/{i}')
    for media in medias:
        if not os.path.exists(f'{newpath1}/{i}/{media}'):
            os.makedirs(f'{newpath1}/{i}/{media}')
        files=os.listdir(f'{path1}/{i}/{media}')
        for file in files:
            img=cv2.imread(f'{path1}/{i}/{media}/{file}')
            add_img=cv2.add(img,img)
            cv2.imwrite(f'{newpath1}/{i}/{media}/{file}',add_img)