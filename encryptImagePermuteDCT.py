# !/usr/bin/env python3
import numpy as np
import cv2
import traceback

redis_host = "localhost"
redis_port = 6379
redis_password = ""
image_folder = "./images/"
filename = "loki.jpg"

block_size = 8


def print_mat(arr: np.ndarray):
    """
    pretty print matrix information for small matrix
    :param arr:
    :return:
    """
    print("the shape of array: " + str(arr.shape))
    print("the dtype of array: " + str(arr.dtype))
    if arr.dtype.type is np.uint8:
        print('\n'.join(['\t\t'.join(['% .3d' % cell for cell in row]) for row in arr]))
    elif arr.dtype.type is np.float32:
        print('\n'.join(['\t\t'.join(['% .8f' % cell for cell in row]) for row in arr]))
    print('\n')


def rand_perm(n) -> [int]:
    """
        生成一个1到n之间的随机排列的P
    :param n:
    :return:
    """
    return np.random.permutation(n)


def generate_random_series(h_blocks, w_blocks):
    """

    p_table: (random_series_index, block_index) -> block_index_after_permute
    3 channel, 4 sub-block for each block, thus 12 series is needed
    block: 8 x 8
    sub-block: 4 x 4
    one block = 4 sub-block

    :param h_blocks:
    :param w_blocks:
    :return: p_table_len, p_table, destination
    """
    size_p_table = 3*4, h_blocks * w_blocks
    p_table = np.zeros(size_p_table, dtype=np.uint32)
    for i in range(0, 12):
        p_table[i] = rand_perm(h_blocks * w_blocks)

    # print_mat(p_table)
    # print(p_table.size)
    # print(p_table)

    size_dst = 2, h_blocks * w_blocks
    destination = np.zeros(size_dst, dtype=np.uint32)
    k = 0
    for i in range(h_blocks):
        for j in range(w_blocks):
            destination[0][k] = (k // w_blocks) * 8
            destination[1][k] = (k % w_blocks) * 8
            k += 1
    # print(destination)
    print()
    return p_table, destination


def encrypt(src, p_table, destination):
    print(src.shape)
    # corp to fit block size
    h, w = np.array(src.shape[:2]) // block_size * block_size
    h_blocks, w_blocks = h // block_size, w // block_size

    img_corp = src[:h_blocks * block_size, :w_blocks * block_size]
    img_split = cv2.split(img_corp)

    dst = []
    for ch in img_split:
        p = 0
        # DCT-ize original chanel
        dct_tmp = np.zeros(ch.shape, np.float32)
        permute_tmp = np.zeros(ch.shape, np.float32)
        res = np.zeros(ch.shape, np.uint8)
        for j in range(h_blocks):
            for i in range(w_blocks):
                x1 = i * 8
                y1 = j * 8
                x2 = x1 + 8
                y2 = y1 + 8
                # block = ch[y1:y2, x1:x2]
                dct_tmp[y1:y2, x1:x2] = cv2.dct(np.float32(ch[y1:y2, x1:x2]) / 255.0)
                permute_tmp[y1:y2, x1:x2] = dct_tmp[y1:y2, x1:x2]

        # permute DCT coefficients
        for sub_block_y in [0, 4]:
            for sub_block_x in [0, 4]:
                # print(sub_block_y, sub_block_x)
                block_idx = 0
                for j in range(h_blocks):
                    for i in range(w_blocks):
                        dst_block_idx = p_table[p][block_idx]
                        block_idx += 1
                        org_y1 = j * 8 + sub_block_y
                        org_x1 = i * 8 + sub_block_x
                        org_y2 = org_y1 + 4
                        org_x2 = org_x1 + 4
                        dst_y1 = destination[0][dst_block_idx] + sub_block_y
                        dst_x1 = destination[1][dst_block_idx] + sub_block_x
                        dst_y2 = dst_y1 + 4
                        dst_x2 = dst_x1 + 4
                        permute_tmp[dst_y1:dst_y2, dst_x1:dst_x2] = dct_tmp[org_y1:org_y2, org_x1:org_x2]
                p += 1
        # block_idx = 0
        # for j in range(h_blocks):
        #     for i in range(w_blocks):
        #         dst_block_idx = p_table[p][block_idx]
        #         block_idx += 1
        #         org_y1 = j * 8
        #         org_x1 = i * 8
        #         org_y2 = org_y1 + 8
        #         org_x2 = org_x1 + 8
        #         dst_y1 = destination[0][dst_block_idx]
        #         dst_x1 = destination[1][dst_block_idx]
        #         dst_y2 = dst_y1 + 8
        #         dst_x2 = dst_x1 + 8
        #         permute_tmp[dst_y1:dst_y2, dst_x1:dst_x2] = dct_tmp[org_y1:org_y2, org_x1:org_x2]

        # inverse DCT-ize back
        for j in range(h_blocks):
            for i in range(w_blocks):
                x1 = i * 8
                y1 = j * 8
                x2 = x1 + 8
                y2 = y1 + 8
                res[y1:y2, x1:x2] = np.uint8(cv2.idct(permute_tmp[y1:y2, x1:x2]) * 255 + 0.5)
                # if flag == 3:
                #     flag = 0
                #     # block = ch[y1:y2, x1:x2]
                #     # print_mat(block)
                #     # tmp = np.float32(ch[y1:y2, x1:x2]) / 255.0
                #     # res = cv2.dct(tmp)
                #     print(x1, x2, y1, y2)
                #     print_mat(res[y1:y2, x1:x2])
                #     # print_mat(org)
        dst.append(res)

    img_res = cv2.merge((dst[0], dst[1], dst[2]))
    cv2.imshow('encrypt show im sub', img_split[0])
    cv2.imshow('encrypt show im res', img_res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_res


def decrypt(src, p_table, destination):
    print(src.shape)
    # corp to fit block size
    h, w = np.array(src.shape[:2]) // block_size * block_size
    h_blocks, w_blocks = h // block_size, w // block_size

    img_corp = src[:h_blocks * block_size, :w_blocks * block_size]
    img_split = cv2.split(img_corp)

    dst = []

    for ch in img_split:
        p = 0
        # DCT-ize original chanel
        dct_tmp = np.zeros(ch.shape, np.float32)
        permute_tmp = np.zeros(ch.shape, np.float32)
        res = np.zeros(ch.shape, np.uint8)
        for j in range(h_blocks):
            for i in range(w_blocks):
                x1 = i * 8
                y1 = j * 8
                x2 = x1 + 8
                y2 = y1 + 8
                # block = ch[y1:y2, x1:x2]
                dct_tmp[y1:y2, x1:x2] = cv2.dct(np.float32(ch[y1:y2, x1:x2]) / 255.0)

        # permute DCT coefficients
        for sub_block_y in [0, 4]:
            for sub_block_x in [0, 4]:
                # print(sub_block_y, sub_block_x)
                block_idx = 0
                for j in range(h_blocks):
                    for i in range(w_blocks):
                        dst_block_idx = p_table[p][block_idx]
                        block_idx += 1
                        dst_y1 = j * 8 + sub_block_y
                        dst_x1 = i * 8 + sub_block_x
                        dst_y2 = dst_y1 + 4
                        dst_x2 = dst_x1 + 4
                        org_y1 = destination[0][dst_block_idx] + sub_block_y
                        org_x1 = destination[1][dst_block_idx] + sub_block_x
                        org_y2 = org_y1 + 4
                        org_x2 = org_x1 + 4
                        permute_tmp[dst_y1:dst_y2, dst_x1:dst_x2] = dct_tmp[org_y1:org_y2, org_x1:org_x2]
                p += 1
        # block_idx = 0
        # for j in range(h_blocks):
        #     for i in range(w_blocks):
        #         dst_block_idx = p_table[p][block_idx]
        #         block_idx += 1
        #         dst_y1 = j * 8
        #         dst_x1 = i * 8
        #         dst_y2 = dst_y1 + 8
        #         dst_x2 = dst_x1 + 8
        #         org_y1 = destination[0][dst_block_idx]
        #         org_x1 = destination[1][dst_block_idx]
        #         org_y2 = org_y1 + 8
        #         org_x2 = org_x1 + 8
        #         permute_tmp[dst_y1:dst_y2, dst_x1:dst_x2] = dct_tmp[org_y1:org_y2, org_x1:org_x2]

        # inverse DCT-ize back
        for j in range(h_blocks):
            for i in range(w_blocks):
                x1 = i * 8
                y1 = j * 8
                x2 = x1 + 8
                y2 = y1 + 8
                res[y1:y2, x1:x2] = np.uint8(cv2.idct(permute_tmp[y1:y2, x1:x2]) * 255 + 0.5)
        dst.append(res)

    img_res = cv2.merge((dst[0], dst[1], dst[2]))
    cv2.imshow('decrypt show im sub', img_split[0])
    cv2.imshow('decrypt show im res', img_res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img_res


def process():
    """
    pre process data
    :return:
    """
    try:
        pass
        img = cv2.imread(image_folder+filename, cv2.IMREAD_UNCHANGED)

        h_blocks, w_blocks = np.array(img.shape[:2]) // block_size

        p_table, destination = generate_random_series(h_blocks, w_blocks)

        tmp1 = encrypt(img, p_table, destination)

        cv2.imwrite(image_folder+"permute_dct_encrypted.jpg", tmp1)

        tmp2 = cv2.imread(image_folder+"permute_dct_encrypted.jpg", cv2.IMREAD_UNCHANGED)
        res = decrypt(tmp2, p_table, destination)

        cv2.imwrite(image_folder+"permute_dct_decrypted.jpg", res)

    except Exception as e:
        print(e)
        traceback.print_exc()


if __name__ == '__main__':
    process()
