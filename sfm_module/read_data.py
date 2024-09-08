import yaml


def get_data(path_to_cfg):

    with open(path_to_cfg, "r") as file:
        cfg_file = yaml.safe_load(file)
        focal_length = cfg_file["camera"]["focal_length"]
        principal_point = cfg_file["camera"]["principal_point"]
        img_paths = [path for path in cfg_file["image_file_paths"]]
        init_pair = cfg_file["initial_pair"]
    K = [
        [focal_length[0], 0, principal_point[0]],
        [0, focal_length[1], principal_point[1]],
        [0, 0, 1],
    ]
    return K, img_paths, init_pair


get_data(
    "/home/simsom/work_space/repos/Structure-from-motion/sfm_module/data/1/cfg.yml"
)
