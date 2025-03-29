import numpy as np
import cv2


LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_HIP, RIGHT_HIP = 11, 12

def calculate_overlap(label_bbox, predicted_bbox):
    """
    Calculates how much of the label box is covered by the predicted box.

    Args:
        label_bbox: List or tuple (x1, y1, x2, y2) — ground truth box
        predicted_bbox: List or tuple (x1, y1, x2, y2) — predicted box

    Returns:
        overlap_ratio: float — intersection area / label area
    """
    temp_label_bbox = label_bbox.copy()
    temp_predicted_bbox = predicted_bbox.copy()
    lx1, ly1, lx2, ly2 = temp_label_bbox
    px1, py1, px2, py2 = temp_predicted_bbox

    xi1 = max(lx1, px1)
    yi1 = max(ly1, py1)
    xi2 = min(lx2, px2)
    yi2 = min(ly2, py2)

    if xi1 >= xi2 or yi1 >= yi2:
        return 0.0

    intersection_area = (xi2 - xi1) * (yi2 - yi1)

    label_area = (lx2 - lx1) * (ly2 - ly1)
    if label_area == 0:
        return 0.0

    overlap_ratio = intersection_area / label_area

    return overlap_ratio


def calculate_euclidean_distance(bbox1, bbox2):
    temp_bbox1 = bbox1.copy()
    temp_bbox2 = bbox2.copy()
    x1, y1 = (temp_bbox1[0] + temp_bbox1[2]) / 2, (temp_bbox1[1] + temp_bbox1[3]) / 2
    x2, y2 = (temp_bbox2[0] + temp_bbox2[2]) / 2, (temp_bbox2[1] + temp_bbox2[3]) / 2

    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    
def shift_points(data):
    """ Shift points based on the bounding box and scaling factors """
    xmin, ymin, xmax, ymax, pt_x, pt_y, sscale_ratio, padding_ratio = data
    pt_x = pt_x - xmin
    pt_y = pt_y - ymin
    pt_x = pt_x * sscale_ratio
    pt_y = pt_y * sscale_ratio
    pt_x = pt_x + padding_ratio[0]
    pt_y = pt_y + padding_ratio[1]
    return pt_x, pt_y

def get_rotation_angle(keypoints):
    """ Determine the rotation angle of the keypoints """
    if keypoints.shape[1] < 2:
        # print("Keypoints không đủ dữ liệu!")
        return None, 0

    rotate = 0

    if keypoints[LEFT_HIP][0] > 0 and keypoints[RIGHT_HIP][0] > 0:
        p1, p2 = keypoints[LEFT_HIP], keypoints[RIGHT_HIP]
        rotate = 1

    elif keypoints[LEFT_SHOULDER][0] > 0 and keypoints[RIGHT_SHOULDER][0] > 0:
        p1, p2 = keypoints[LEFT_SHOULDER], keypoints[RIGHT_SHOULDER]
        rotate = 1

    elif keypoints[RIGHT_SHOULDER][0] > 0 and keypoints[RIGHT_HIP][0] > 0:
        p1, p2 = keypoints[RIGHT_SHOULDER], keypoints[RIGHT_HIP]
        rotate = 2

    elif keypoints[LEFT_SHOULDER][0] > 0 and keypoints[LEFT_HIP][0] > 0:
        p1, p2 = keypoints[LEFT_SHOULDER], keypoints[LEFT_HIP]
        rotate = 2

    elif keypoints[LEFT_SHOULDER][0] > 0 and keypoints[RIGHT_HIP][0] > 0:
        p1, p2 = keypoints[LEFT_SHOULDER], keypoints[RIGHT_HIP]
        rotate = 3
    elif keypoints[RIGHT_SHOULDER][0] > 0 and keypoints[LEFT_HIP][0] > 0:
        p1, p2 = keypoints[RIGHT_SHOULDER], keypoints[LEFT_HIP]
        rotate = 3
    else:
        # print("Không đủ keypoints để tính góc xoay!")
        return None, 0

    angle = -np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

    if rotate == 1:
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
    elif rotate == 2:
        angle += 90  
    elif rotate == 3:
        angle -= 90  

    return angle, rotate

def rotate_cropped_image(image, angle):
    h, w = image.shape[:2]

    # x, y, bw, bh = bbox
    # center = (x + bw // 2, y + bh // 2)

    center = (w // 2, h // 2)

    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    M = cv2.getRotationMatrix2D(center, -angle, 1.0)

    ones = np.ones((4, 1))
    corners_homogeneous = np.hstack([corners, ones])  
    rotated_corners = np.dot(M, corners_homogeneous.T).T  

    min_x, min_y = np.min(rotated_corners, axis=0)
    max_x, max_y = np.max(rotated_corners, axis=0)

    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))

    M[0, 2] += -min_x
    M[1, 2] += -min_y

    rotated_image = cv2.warpAffine(image, M, (new_w, new_h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return rotated_image, M

def rotate_keypoints(keypoints, M):
    """ Rotate keypoints using the rotation matrix """
    rotated_keypoints = keypoints.copy()

    for i, point in enumerate(keypoints):
        if np.any(point != [0, 0]):  
            x, y = point
            new_x, new_y = np.dot(M, np.array([x, y, 1]))  
            rotated_keypoints[i] = [new_x, new_y]

    return rotated_keypoints

def convert_ratio(rotated_image, keypoints):
    """ Normalize keypoints to the range [0, 1] """
    h, w = rotated_image.shape[:2]  
    keypoints_ratio = keypoints.copy()

    for i, (x, y) in enumerate(keypoints):
        if x > 0 and y > 0:  
            keypoints_ratio[i] = [x / w, y / h]  
        # else:
        #     if x == 0 and y == 0:
        #         print('keypioints is 0')
        #     else:
        #         print('keypoints are not valid')
        
    return keypoints_ratio

def add_padding_cropped_img(cropped_image, keypoints):
    temp_keypoint = keypoints.copy()
    pad = 25

    cropped_padded = cv2.copyMakeBorder(
        cropped_image,
        pad,  # top
        pad,  # bottom
        pad,  # left
        pad,  # right
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    temp_keypoint[:, 0] += pad
    temp_keypoint[:, 1] += pad

    return cropped_padded, temp_keypoint