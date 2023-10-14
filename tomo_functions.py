import numpy as np


# Takes in the x, y and z coordinates of all the points and uses a kd_tree to calculate the distance and indexes and
# the nearest neighbours for each point. If all the distances of the nearest neighbours are within the specified range
# it returns a dictonary of the vector indexes and the index of the neighbouring points

def get_close_points_indexes(np_data, kd_tree, dist_val, dist_error, neighbour):
    nearest_neighbor_distances, nearest_neighbor_indexes = kd_tree.query(np_data, k=neighbour)
    range_condition_truth_mask = ((dist_val - dist_error) <= nearest_neighbor_distances[:, 1:]) & (nearest_neighbor_distances[:, 1:] <= (dist_val + dist_error))
    good_indexes_truth_mask = np.all(range_condition_truth_mask, axis=1)
    good_indexes_numbers = np.where(good_indexes_truth_mask)[0]
    initial_indexes = {index: nearest_neighbor_indexes[index][1:].tolist() for index in good_indexes_numbers}
    return initial_indexes



# Calculates the dot product of the orientation vector for each point and the neighbouring vectors and returns a matrix
# of the dot products

def orientation_checker(selected_vectors_direction):
    selected_vectors_direction
    dot_products = np.dot(selected_vectors_direction, selected_vectors_direction.T)
    flattened_dot_products = dot_products.flatten()
    return(flattened_dot_products)


# Takes indexes which meet the distance criterea and checks the orientation of the index with its neighbours and returns
# a dictonary with the indexes of the points and their neighbours that meet the orientation critera

def get_good_orientation_points_indexes(orientation_np_data, initial_indexes, fidelity_number):
    good_orientation_indexes = {}
    for key, values in initial_indexes.items():
        direction_indexes = [key]
        direction_indexes.extend(values)
        selected_vectors_direction = orientation_np_data[direction_indexes]
        dot_products = orientation_checker(selected_vectors_direction)
        # Play with these values if this isn't working well
        true_count = [not (-0.95 <= float(value) <= 0.95) for value in dot_products].count(True)
        if true_count > fidelity_number:
            good_orientation_indexes[key] = values
    return good_orientation_indexes


# Calculates the angles between all of the nearest neighbours for a given point and returns the value

def angle_calculator(centered_vectors):
    normalized_vectors = centered_vectors / np.linalg.norm(centered_vectors, axis=1)[:, np.newaxis]
    dot_products = np.dot(normalized_vectors, normalized_vectors.T)
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angles = np.arccos(dot_products)
    angles_degrees = np.degrees(angles)
    angles_degrees_flattened = angles_degrees.flatten()
    unique_values = np.unique(angles_degrees_flattened)
    unique_values_list = unique_values.tolist()
    return(unique_values_list)


# Takes a dictionary of points and their nearest neighbours that meet the distance and orientation critera and return a list of indexes
# That meet the angle criterea. These are the indexes of the points that are kept and plotted by the cleaning and plotting function

def get_lattice_points_indexes(initial_indexes, np_data, angle_val, angle_error):
    confirmed_indexes = []
    for key, values in initial_indexes.items():
        point = np.array(np_data[key])
        neighbours = np.array(np_data[values])
        centered_vectors = neighbours - point
        angles = angle_calculator(centered_vectors)
        true_count = [(angle_val - angle_error) <= float(value) <= (angle_val + angle_error) for value in angles].count(True)
        if true_count >= 3:
            confirmed_indexes.append(key)
    return confirmed_indexes


# The user clicks three points and the x,y and z coordinates are given to this function and it returns the distance and angle between these points
# to be use as paramteres in the cleaning stage

def calculate_parameters(v1, v2, v3):
    center_point = v1
    centered_v2 = v2 - center_point
    centered_v3 = v3 - center_point
    distance_v2 = np.linalg.norm(centered_v2)
    distance_v3 = np.linalg.norm(centered_v3)
    average_distance = (distance_v2 + distance_v3) / 2
    dot_product = np.dot(centered_v2, centered_v3)
    cosine_similarity = dot_product / (distance_v2 * distance_v3)
    angle_radians = np.arccos(cosine_similarity)
    angle_degrees = np.degrees(angle_radians)
    return average_distance, angle_degrees
