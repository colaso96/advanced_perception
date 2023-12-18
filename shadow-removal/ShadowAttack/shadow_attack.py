# -*- coding: utf-8 -*-

import sys, os
import cv2
import json
import torch
import pickle
import argparse
import numpy as np

sys.path.append(os.path.abspath(__file__))
from ShadowAttack.pso import PSO
import ShadowAttack.gtsrb as gtsrb
import ShadowAttack.lisa as lisa
from ShadowAttack.gtsrb import GtsrbCNN
from ShadowAttack.lisa import LisaCNN
from ShadowAttack.utils import (
    brightness,
    shadow_edge_blur,
    judge_mask_type,
    draw_shadow,
    load_mask,
    pre_process_image,
)
from collections import Counter
from torchvision import transforms

with open('ShadowAttack/params.json', 'rb') as f:
    params = json.load(f)
    class_n_gtsrb = params['GTSRB']['class_n']
    class_n_lisa = params['LISA']['class_n']
    device = params['device']
    position_list, mask_list = load_mask()


def attack(
    attack_image,
    label,
    coords,
    model,
    pre_process,
    particle_size=10,
    iter_num=10,
    targeted_attack=False,
    physical_attack=False,
    n_try=10,
    polygon=3,
    attack_type='physical',
    shadow_level=0.43,
    **parameters,
):
    r"""
    Physical-world adversarial attack by shadow.

    Args:
        attack_image: The image to be attacked.
        label: The ground-truth label of attack_image.
        coords: The coordinates of the points where mask == 1.
        targeted_attack: Targeted / Non-targeted attack.
        physical_attack: Physical / digital attack.

    Returns:
        adv_img: The generated adversarial image.
        succeed: Whether the attack is successful.
        num_query: Number of queries.
    """

    assert attack_type in ['digital', 'physical']
    if attack_type == 'digital':
        particle_size = 10
        iter_num = 10
        x_min, x_max = -16, 48
        max_speed = 1.5
    else:
        particle_size = 10
        iter_num = 20
        x_min, x_max = -112, 336
        max_speed = 10.0
        n_try = 1

    num_query = 0
    succeed = False
    global_best_solution = float('inf')
    global_best_position = None

    for attempt in range(n_try):
        if succeed:
            break

        print(f"try {attempt + 1}:", end=" ")

        pso = PSO(
            polygon * 2,
            particle_size,
            iter_num,
            x_min,
            x_max,
            max_speed,
            shadow_level,
            attack_image,
            coords,
            model,
            targeted_attack,
            physical_attack,
            label,
            pre_process,
            **parameters,
        )
        best_solution, best_pos, succeed, query = (
            pso.update_digital() if not physical_attack else pso.update_physical()
        )

        if targeted_attack:
            best_solution = 1 - best_solution
        print(f"Best solution: {best_solution} {'succeed' if succeed else 'failed'}")
        if best_solution < global_best_solution:
            global_best_solution = best_solution
            global_best_position = best_pos
        num_query += query

    adv_image, shadow_area = draw_shadow(
        global_best_position, attack_image, coords, shadow_level
    )
    adv_image = shadow_edge_blur(adv_image, shadow_area, 3)

    return adv_image, succeed, num_query


def attack_digital():
    save_dir = f'./adv_img/{attack_db}/{int(shadow_level*100)}'
    try:
        os.mkdir(save_dir)
    except FileExistsError:
        for name in os.listdir(save_dir):
            os.remove(os.path.join(save_dir, name))

    with open(f'./dataset/{attack_db}/test.pkl', 'rb') as dataset:
        test_data = pickle.load(dataset)
        images, labels = test_data['data'], test_data['labels']

    for index in range(len(images)):
        mask_type = judge_mask_type(attack_db, labels[index])
        if brightness(images[index], mask_list[mask_type]) >= 120:
            adv_img, success, num_query = attack(
                images[index], labels[index], position_list[mask_type]
            )
            cv2.imwrite(
                f"{save_dir}/{index}_{labels[index]}_{num_query}_{success}.bmp", adv_img
            )

    print("Attack finished! Success rate: ", end='')
    print(
        Counter(map(lambda x: x[:-4].split('_')[-1], os.listdir(save_dir)))['True']
        / len(os.listdir(save_dir))
    )


def attack_physical():
    global position_list

    mask_image = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_UNCHANGED), (224, 224))
    target_image = cv2.resize(cv2.imread(image_path), (224, 224))
    pos_list = np.where(mask_image.sum(axis=2) > 0)

    # EOT is included in the first stage
    adv_img, _, _ = attack(
        target_image, image_label, pos_list, physical_attack=True, transform_num=10
    )

    cv2.imwrite('./tmp/temp.bmp', adv_img)
    if attack_db == 'LISA':
        predict, failed = lisa.test_single_image(
            './tmp/temp.bmp', image_label, target_model == "robust"
        )
    else:
        predict, failed = gtsrb.test_single_image(
            './tmp/temp.bmp', image_label, target_model == "robust"
        )
    if failed:
        print('Attack failed! Try to run again.')

    # Predict stabilization
    adv_img, _, _ = attack(
        target_image,
        image_label,
        pos_list,
        targeted_attack=True,
        physical_attack=True,
        target=predict,
        transform_num=10,
    )

    cv2.imwrite('./tmp/adv_img.png', adv_img)
    if attack_db == 'LISA':
        predict, failed = lisa.test_single_image(
            './tmp/adv_img.png', image_label, target_model == "robust"
        )
    else:
        predict, failed = gtsrb.test_single_image(
            './tmp/adv_img.png', image_label, target_model == "robust"
        )
    if failed:
        print('Attack failed! Try to run again.')
    else:
        print('Attack succeed! Try to implement it in the real world.')

    cv2.imshow("Adversarial image", adv_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    attack_digital() if attack_type == 'digital' else attack_physical()
