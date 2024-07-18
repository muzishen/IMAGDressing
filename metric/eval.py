import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
import os
import argparse


def extract_clothing_keypoints(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [tuple(point) for point in keypoints]

    return keypoints, img


def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True, channel_axis=2)


def calculate_keypoint_matching(keypoints1, keypoints2):
    keypoints1 = np.array(keypoints1)
    keypoints2 = np.array(keypoints2)
    if len(keypoints2) == 0 or len(keypoints2) > 5000:
        return 0.99
    else:

        distances_array1 = np.linalg.norm(keypoints1[:, np.newaxis, :] - keypoints2[np.newaxis, :, :], axis=2)
        min_distances_array1 = np.min(distances_array1, axis=1)

        return np.mean(min_distances_array1) / (512. * np.sqrt(2))


def calculate_texture_similarity(img1, img2, P=8, R=1.0):
    lbp1 = local_binary_pattern(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), P, R)
    lbp2 = local_binary_pattern(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), P, R)
    hist1, _ = np.histogram(lbp1, bins=np.arange(0, P ** 2 + 1), density=True)
    hist2, _ = np.histogram(lbp2, bins=np.arange(0, P ** 2 + 1), density=True)
    hist2 = hist2.astype(np.float32)
    hist1 = hist1.astype(np.float32)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def calculate_cami_us(img1, img2, keypoints1, keypoints2):  # 不指定pose  face , bg  2000张
    ssim_score = calculate_ssim(img1, img2)
    keypoint_matching = calculate_keypoint_matching(keypoints1, keypoints2)

    texture_similarity = calculate_texture_similarity(img1, img2)

    cami_us_score = ssim_score + (1 - keypoint_matching) + texture_similarity

    return cami_us_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute the metrics for the generated images")
    parser.add_argument('--cloth_path', type=str, help='input cloth path')
    parser.add_argument('--cloth_mask_path', type=str, help='generate cloth mask path')
    args = parser.parse_args()

    cloth_paths = os.listdir(args.cloth_path)
    score_list = []
    for cloth in cloth_paths:
        cloth = os.path.join(args.cloth_path, cloth)
        generate_cloth = os.path.join(args.cloth_mask_path, cloth)
        if not os.path.exists(generate_cloth):
            score_list.append(0)
            continue
        reference_cloth_img = cv2.imread(cloth)
        generated_cloth_img = cv2.imread(generate_cloth)

        h, w = generated_cloth_img.shape[0], generated_cloth_img.shape[1]
        reference_cloth_img = cv2.resize(reference_cloth_img, (w, h))

        reference_keypoints, _ = extract_clothing_keypoints(cloth)
        generated_keypoints, _ = extract_clothing_keypoints(generated_cloth_img)

        cami_us_score = calculate_cami_us(reference_cloth_img, generated_cloth_img, reference_keypoints,
                                          generated_keypoints)
        score_list.append(cami_us_score)

    print('cami_us_score:', np.mean(score_list))
