import numpy as np
from math import pi, sin, cos, sqrt, acos
import copy


def slope(x1, y1, x2, y2):
    if x1 == x2:
        return None
    else:
        m = (y2 - y1) / float((x2 - x1))
        return m


def get_line_equation(x1, y1, x2, y2):
    m = slope(x1, y1, x2, y2)
    if m is None:
        m = y2 - y1
    c = -(m * x1 - y1)

    return m, c


def get_vector_representation(x1, y1, x2, y2):
    pos_vector = np.array([x1, y1])
    dir_vector = np.array([x2 - x1, y2 - y1])
    return pos_vector, dir_vector


def get_perp_intersect_point(pos_vector, dir_vector, x, y):
    # (Gerade - Punkt) * RV = 0
    p = np.array([x, y])
    point_vector = (pos_vector - p)
    res = np.dot(dir_vector, point_vector)
    r = np.dot(dir_vector, dir_vector)
    r = -res / float(r)
    return pos_vector + (r * dir_vector)


def get_distance_between_vectors(v1, v2):
    v = v1 - v2
    return get_magnitude(v)


def get_magnitude(v):
    return np.linalg.norm(v)


def is_point_on_line(x1, y1, x2, y2, x3, y3):
    dxl = x2 - x1
    dyl = y2 - y1

    if abs(dxl) >= abs(dyl):
        return x1 <= x3 <= x2 if dxl > 0 else x2 <= x3 <= x1
    else:
        return y1 <= y3 <= y2 if dyl > 0 else y2 <= y3 <= y1


def compute_line_intersection(line1, line2):
    for x1, y1, x2, y2 in line1:
        m1, n1 = get_line_equation(x1, y1, x2, y2)
    for x1, y1, x2, y2 in line2:
        m2, n2 = get_line_equation(x1, y1, x2, y2)

    new_n = n2 - n1
    new_m = m1 - m2
    intersect_x = new_n / float(new_m)
    intersect_y = m1 * intersect_x + n1

    return intersect_x, intersect_y


def compute_angle_between_points(x1, y1, x2, y2, x3, y3):
    # computes the angle at point 3
    v31 = np.array([x1 - x3, y1 - y3])
    v32 = np.array([x2 - x3, y2 - y3])

    value = np.dot(v31, v32)
    value = value / (float(get_magnitude(v31)) * float(get_magnitude(v32)))
    return np.arccos([value])[0] * 180 / pi


def normalize_vector(vector):
    return vector / np.linalg.norm(vector)


def convert_to_vector_with_magnitude(vec, magnitude):
    vector = normalize_vector(vec)
    return vector * magnitude


def move_points_to_origin(points):
    array = copy.deepcopy(points)
    x, y = array[0], array[1]
    vector = np.array([-x, -y])

    for i in range(0, len(array), 2):
        array[i] += vector[0]
        array[i + 1] += vector[1]

    return array


def compute_linear_zero(x1, y1, x2, y2):
    m, n = get_line_equation(x1, y1, x2, y2)
    x = -n / m
    return x, 0


def compute_radius(angle, p1, p2):
    length = get_distance_between_vectors(p1, p2)
    return length / (2 * sin(angle / 2))


def compute_radius_by_arc_and_angle(arc, angle):
    return abs(arc / angle)


def compute_circle_arc(angle, radius):
    return radius * angle


def compute_side_with_sws(a, beta, c):
    return sqrt((a * a) + (c * c) - (2 * a * c * cos(beta)))


def rotate_point_around_origin(point, angle):
    x = point[0]
    y = point[1]

    new_x = x * cos(angle) - y * sin(angle)
    new_y = x * sin(angle) + y * cos(angle)

    return [new_x, new_y]


def to_radians(point):
    #takes a point on unit circle and return the corrsponding radians
    x = point[0]
    y = point[1]

    radians = acos(x)
    if y < 0:
        radians = pi + (pi - radians)

    return radians
