"""
Computer Vision
EEN020
Project
2023-12-29

Performs chirality check

Authors:
        Maximilian Salén
        Axel Qvarnström
"""

import numpy as np
from auxiliary import triangulate_3D_point_DLT

def perform_chirality_check(P2s, K, x1, x2):
    xs = [x1, x2]
    P1 = np.concatenate((np.eye(3), np.zeros((3,1))), axis=1)

    triangulated_points = []
    for i, P2 in enumerate(P2s):
        DLT_points = triangulate_3D_point_DLT([P1, P2], xs)
        triangulated_points.append(DLT_points)

    best_idx, num_points_infront_of_cam = select_best_solution(triangulated_points, P1, P2s)
    best_P = P2s[best_idx]
    return best_P, num_points_infront_of_cam

def normalize_K(K, xs):
    return np.linalg.inv(K) @ xs


def count_points_in_front_of_cam(P1, P2, points_3d):
    count = 0
    for X in points_3d.T:
        if (P2 @ X)[2] > 0 and (P1 @ X)[2] > 0:
            count += 1
    return count

def select_best_solution(triangulated_points, P1, P2s):
    best_sol = None
    max_point_in_front = -1

    for i, points_3d in enumerate(triangulated_points):
        count = count_points_in_front_of_cam(P1, P2s[i], points_3d)
        
        if count > max_point_in_front:
            max_point_in_front = count
            best_idx = i
            

    return best_idx, max_point_in_front