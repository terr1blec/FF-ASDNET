import cv2
import os

path1='l_4fc_d0.5_convcon_conv512_1*1_conv256_3*3_update_1loss_3fc/d0.5_n0.5_shuffle_2/cam'
newpath1='l_4fc_d0.5_convcon_conv512_1*1_conv256_3*3_update_1loss_3fc/d0.5_n0.5_shuffle_3/cam'

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

