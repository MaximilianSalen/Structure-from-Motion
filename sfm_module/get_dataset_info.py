def get_dataset_info(data_root, dataset):
    if dataset == 1:
        img_names = ["data/1/kronan1.JPG", "data/1/kronan2.JPG"]
        im_width, im_height = 1936, 1296
        focal_length_35mm = 45.0  # from the EXIF data
        init_pair = [0, 1]
        pixel_threshold = 1.0
    elif dataset == 2:
        img_names = [
            "data/2/DSC_0025.JPG",
            "data/2/DSC_0026.JPG",
            "data/2/DSC_0027.JPG",
            "data/2/DSC_0028.JPG",
            "data/2/DSC_0029.JPG",
            "data/2/DSC_0030.JPG",
            "data/2/DSC_0031.JPG",
            "data/2/DSC_0032.JPG",
            "data/2/DSC_0033.JPG",
        ]
        im_width, im_height = 1936, 1296
        focal_length_35mm = 43.0  # from the EXIF data
        init_pair = [0, 8]
        pixel_threshold = 1.0
    elif dataset == 3:
        img_names = [
            "data/3/DSC_0001.JPG",
            "data/3/DSC_0002.JPG",
            "data/3/DSC_0003.JPG",
            "data/3/DSC_0004.JPG",
            "data/3/DSC_0005.JPG",
            "data/3/DSC_0006.JPG",
            "data/3/DSC_0007.JPG",
            "data/3/DSC_0008.JPG",
            "data/3/DSC_0009.JPG",
            "data/3/DSC_0010.JPG",
            "data/3/DSC_0011.JPG",
            "data/3/DSC_0012.JPG",
        ]
        im_width, im_height = 1936, 1296
        focal_length_35mm = 43.0  # from the EXIF data
        init_pair = [4, 7]
        pixel_threshold = 1.0
    elif dataset == 4:
        img_names = [
            "data/4/DSC_0480.JPG",
            "data/4/DSC_0481.JPG",
            "data/4/DSC_0482.JPG",
            "data/4/DSC_0483.JPG",
            "data/4/DSC_0484.JPG",
            "data/4/DSC_0485.JPG",
            "data/4/DSC_0486.JPG",
            "data/4/DSC_0487.JPG",
            "data/4/DSC_0488.JPG",
            "data/4/DSC_0489.JPG",
            "data/4/DSC_0490.JPG",
            "data/4/DSC_0491.JPG",
            "data/4/DSC_0492.JPG",
            "data/4/DSC_0493.JPG",
        ]
        im_width, im_height = 1936, 1296
        focal_length_35mm = 43.0  # from the EXIF data
        init_pair = [4, 9]
        pixel_threshold = 1.0
    elif dataset == 5:
        img_names = [
            "data/5/DSC_0336.JPG",
            "data/5/DSC_0337.JPG",
            "data/5/DSC_0338.JPG",
            "data/5/DSC_0339.JPG",
            "data/5/DSC_0340.JPG",
            "data/5/DSC_0341.JPG",
            "data/5/DSC_0342.JPG",
            "data/5/DSC_0343.JPG",
            "data/5/DSC_0344.JPG",
            "data/5/DSC_0345.JPG",
        ]
        im_width, im_height = 1936, 1296
        focal_length_35mm = 45.0  # from the EXIF data
        init_pair = [4, 6]
        pixel_threshold = 1.0
    elif dataset == 6:
        img_names = [
            "data/6/DSCN2115.JPG",
            "data/6/DSCN2116.JPG",
            "data/6/DSCN2117.JPG",
            "data/6/DSCN2118.JPG",
            "data/6/DSCN2119.JPG",
            "data/6/DSCN2120.JPG",
            "data/6/DSCN2121.JPG",
            "data/6/DSCN2122.JPG",
        ]
        im_width, im_height = 2272, 1704
        focal_length_35mm = 38.0  # from the EXIF data
        init_pair = [1, 3]
        pixel_threshold = 1.0
    elif dataset == 7:
        img_names = [
            "data/7/DSCN7409.JPG",
            "data/7/DSCN7410.JPG",
            "data/7/DSCN7411.JPG",
            "data/7/DSCN7412.JPG",
            "data/7/DSCN7413.JPG",
            "data/7/DSCN7414.JPG",
            "data/7/DSCN7415.JPG",
        ]
        im_width, im_height = 2272, 1704
        focal_length_35mm = 38.0  # from the EXIF data
        init_pair = [0, 6]
        pixel_threshold = 1.0
    elif dataset == 8:
        img_names = [
            "data/8/DSCN5540.JPG",
            "data/8/DSCN5541.JPG",
            "data/8/DSCN5542.JPG",
            "data/8/DSCN5543.JPG",
            "data/8/DSCN5544.JPG",
            "data/8/DSCN5545.JPG",
            "data/8/DSCN5546.JPG",
            "data/8/DSCN5547.JPG",
            "data/8/DSCN5548.JPG",
            "data/8/DSCN5549.JPG",
            "data/8/DSCN5550.JPG",
        ]
        im_width, im_height = 2272, 1704
        focal_length_35mm = 38.0  # from the EXIF data
        init_pair = [3, 6]
        pixel_threshold = 1.0
    elif dataset == 9:
        img_names = [
            "data/9/DSCN5184.JPG",
            "data/9/DSCN5185.JPG",
            "data/9/DSCN5186.JPG",
            "data/9/DSCN5187.JPG",
            "data/9/DSCN5188.JPG",
            "data/9/DSCN5189.JPG",
            "data/9/DSCN5191.JPG",
            "data/9/DSCN5192.JPG",
            "data/9/DSCN5193.JPG",
        ]
        im_width, im_height = 2272, 1704
        focal_length_35mm = 38.0  # from the EXIF data
        init_pair = [3, 5]
        pixel_threshold = 1.0

    elif dataset == 10:
        img_names = [
            "data/10/0000.png",
            "data/10/0001.png",
            "data/10/0002.png",
            "data/10/0003.png",
            "data/10/0004.png",
            "data/10/0005.png",
            "data/10/0006.png",
            "data/10/0007.png",
            "data/10/0008.png",
            "data/10/0009.png",
            "data/10/0009.png",
            "data/10/0010.png",
        ]
        init_pair = [4, 6]
        K = [[689.87, 0, 380.17], [0, 691.04, 251.70], [0, 0, 1]]
        updated_img_names = [data_root + x for x in img_names]
        pixel_threshold = 1.0
        return K, updated_img_names, init_pair, pixel_threshold

    elif dataset == 11:
        img_names = [
            "data/11/0000.jpg",
            "data/11/0001.jpg",
            "data/11/0003.jpg",
            "data/11/0004.jpg",
            "data/11/0005.jpg",
            "data/11/0006.jpg",
            "data/11/0007.jpg",
            "data/11/0008.jpg",
            "data/11/0009.jpg",
        ]
        init_pair = [1, 3]
        K = [[2759.48, 0, 1520.69], [0, 2764.16, 1006.81], [0, 0, 1]]
        updated_img_names = [data_root + x for x in img_names]
        pixel_threshold = 1.0
        return K, updated_img_names, init_pair, pixel_threshold

    focal_length = max(im_width, im_height) * focal_length_35mm / 35.0
    K = [[focal_length, 0, im_width / 2], [0, focal_length, im_height / 2], [0, 0, 1]]
    updated_img_names = [data_root + x for x in img_names]
    return K, updated_img_names, init_pair, pixel_threshold
