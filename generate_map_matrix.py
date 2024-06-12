import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import math
from skimage.draw import line
from skimage import draw
import cv2
import pickle
import yaml
import csv
import os

shapes = []

wall_counter = 0

angles_dict = {
    'draw_triangle': [-60, 60],
    'draw_square': [-45, 45],
    'draw_pentagon': [-36, 36]
}

custom_maps_filepath: str = './worlds/custom_maps/'
map_name = "aaaaaaaaamap"

# Create and save the new Webots file
base_map_webots_filepath: str = custom_maps_filepath + 'base_map.wbt'
f = open(base_map_webots_filepath, 'r')
webots_str: str = f.read()
f.close()

map_webots_filepath: str = custom_maps_filepath + map_name + '.wbt'
f = open(map_webots_filepath, 'w')
f.write(webots_str+"\n")

resolution =  0.001
origin =  [0.000000, 0.000000, 0.000000]
occupied_thresh =  0.65
height = 10000
width = 10000

#    Add the rectangular arena
f.write('RectangleArena {\n')
f.write('  translation ' + str(origin[0] + resolution * width / 2) + ' ' + str(origin[1] + resolution * height / 2) + ' 0.0\n')
f.write('  floorSize ' + str(resolution * width) + ' ' + str(resolution * height) + "\n")
f.write('  floorTileSize 0.25 0.25\n')
f.write('  floorAppearance Parquetry {\n')
f.write('    type "light strip"\n')
f.write('  }\n')
f.write('  wallHeight 0.05\n')
f.write('}\n')


# Function to draw a square on the matrix

def expand_geometric_figure(vertices, pixel_distance):
    # Calculate the centroid
    n = len(vertices)
    G_x = sum([v[0] for v in vertices]) / n
    G_y = sum([v[1] for v in vertices]) / n

    # Scale the vertices based on the fixed pixel distance
    new_vertices = []
    for (x, y) in vertices:
        # Calculate the direction vector from the centroid
        dx = x - G_x
        dy = y - G_y

        # Calculate the distance from each vertex to the centroid
        distance_to_centroid = ((dx ** 2) + (dy ** 2)) ** 0.5

        # Calculate the scaling factor to increase the distance
        scaling_factor = (distance_to_centroid + pixel_distance) / distance_to_centroid

        # Scale the direction vector
        new_dx = dx * scaling_factor
        new_dy = dy * scaling_factor

        # Calculate the new position
        new_x = G_x + new_dx
        new_y = G_y + new_dy

        # Ensure the new positions are integers
        new_vertices.append((int(new_x), int(new_y)))

    return new_vertices

def rotate_point(center, point, angle):
    # Ensure the inputs are scalar values
    ox, oy = center
    px, py = point

    #print(f"rotate_point -> center: {center}, point: {point}, angle: {angle}")

    angle_rad = np.deg2rad(angle)
    qx = ox + np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy)
    qy = oy + np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy)
    return int(qx), int(qy)


def write_webots_world(point1, point2):
    global wall_counter
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    angle = math.atan2(y2 - y1, x2 - x1)
    M = ((x2 + x1)/2, (y2 + y1)/2)
    size = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    f.write('Wall {\n')
    f.write(f'  translation {(M[0])*resolution} {((M[1]))*resolution} 0\n')
    f.write(f'  rotation 0 0 1 {angle+(math.pi/2)} \n')
    f.write(f'  size 0.05 {size*resolution} 0.05\n')
    f.write(f'  name "solid' + str(wall_counter) + '"')
    f.write('}\n')
    wall_counter += 1

#==================================================================================================

def draw_square(matrix, mask, shapes, top_left, size, angle=0, tresh_matrix=None, threshold=300, placed=None):
    x, y = top_left

    # Define the vertices of the square
    #if x + size > matrix.shape[0] - threshold or x < threshold or y < threshold or y + size > matrix.shape[
    #    0] - threshold:
    #    matrix += 1000


    A = (x, y)
    B = (x + size, y)
    C = (x + size, y + size)
    D = (x, y + size)
    #print(A, B, C, D)
    # Calculate the centroid of the square
    vertices = np.array([A, B, C, D])
    center = tuple(np.mean(vertices, axis=0).astype(int))

    # Rotate the vertices around the centroid
    A_rot = rotate_point(center, A, angle)
    B_rot = rotate_point(center, B, angle)
    C_rot = rotate_point(center, C, angle)
    D_rot = rotate_point(center, D, angle)

    if placed:
        write_webots_world(A_rot, B_rot)
        write_webots_world(B_rot, C_rot)
        write_webots_world(C_rot, D_rot)
        write_webots_world(D_rot, A_rot)

        f_csv = open(custom_maps_filepath + map_name + '_points.csv', 'w', newline='')
        writer = csv.writer(f_csv)




    #print(A_rot, B_rot, C_rot, D_rot)
    # Draw the rotated square
    cv2.line(matrix, A_rot, B_rot, color=(255, 255, 255), thickness=10)
    cv2.line(matrix, B_rot, C_rot, color=(255, 255, 255), thickness=10)
    cv2.line(matrix, C_rot, D_rot, color=(255, 255, 255), thickness=10)
    cv2.line(matrix, D_rot, A_rot, color=(255, 255, 255), thickness=10)

    # Create and fill the polygon
    polygon = np.array([A_rot, B_rot, C_rot, D_rot], np.int32)
    #breakpoint()
    #rr, cc = draw.polygon(polygon[:, 1], polygon[:, 0], mask.shape)
    #mask[rr, cc] = 1

    cv2.fillConvexPoly(mask, polygon, 1)
    shapes.append(["square", polygon, angle, center])
    #if placed:
    #    breakpoint()

    if tresh_matrix is not None:

        A_exp, B_exp, C_exp, D_exp = expand_geometric_figure([A_rot, B_rot, C_rot, D_rot], 1000)

        if any(x < threshold or x > tresh_matrix.shape[0] - threshold or y < threshold or y > tresh_matrix.shape[1] - threshold for x, y in [A_exp, B_exp, C_exp, D_exp]):
            tresh_matrix += 1000

        #print(A_exp, B_exp, C_exp, D_exp)
        polygon_exp = np.array([A_exp, B_exp, C_exp, D_exp], np.int32)
        #print("im here")
        #breakpoint()
        #rr, cc = draw.polygon(polygon_exp[:, 1], polygon_exp[:, 0], tresh_matrix.shape)
        #tresh_matrix[rr, cc] = 1


        cv2.fillConvexPoly(tresh_matrix, polygon_exp, (255, 255, 255))
    #print("okok")
    return shapes

# Function to draw a triangle on the matrix
def draw_triangle(matrix, mask, shapes, top_left, size, angle=0, tresh_matrix = None, threshold=300, placed=None):

    x, y = top_left
    if size % 2 != 0:
        size += 1
    height = int((np.sqrt(3)/2) * size)  # Height of the equilateral triangle

    # if y-height < threshold or x+size > matrix.shape[0]-threshold or x < threshold:
        # matrix += 1000

    A = (x, y)
    B = (x + size, y)
    C = (int(x + size / 2), y + height)
    #print('#')
    #print(A, B, C)

    vertices = np.array([A, B, C])
    center = tuple(np.mean(vertices, axis=0).astype(int))

    A_rot = rotate_point(center, A, angle)
    B_rot = rotate_point(center, B, angle)
    C_rot = rotate_point(center, C, angle)

    if placed:
        write_webots_world(A_rot, B_rot)
        write_webots_world(A_rot, C_rot)
        write_webots_world(B_rot, C_rot)

    #print(A_rot, B_rot, C_rot)
    cv2.line(matrix, A_rot, B_rot, color=(255, 255, 255), thickness=10)
    cv2.line(matrix, A_rot, C_rot, color=(255, 255, 255), thickness=10)
    cv2.line(matrix, B_rot, C_rot, color=(255, 255, 255), thickness=10)

    polygon = np.array([A_rot, B_rot, C_rot], np.int32)
    cv2.fillConvexPoly(mask, polygon, 1)

    shapes.append(["triangle", polygon, angle, center])

    if tresh_matrix is not None:

        A_exp, B_exp, C_exp = expand_geometric_figure([A_rot, B_rot, C_rot], 1000)
        #print(A_exp, B_exp, C_exp)

        # test if any of the expanded points is too close to the border
        if any(x < threshold or x > tresh_matrix.shape[0] - threshold or y < threshold or y > tresh_matrix.shape[1] - threshold for x, y in [A_exp, B_exp, C_exp]):
            tresh_matrix += 1000

        polygon = np.array([A_exp, B_exp, C_exp], np.int32)
        #cv2.line(tresh_matrix, A_exp, B_exp, color=(255, 255, 255), thickness=10)
        #cv2.line(tresh_matrix, A_exp, C_exp, color=(255, 255, 255), thickness=10)
        #cv2.line(tresh_matrix, B_exp, C_exp, color=(255, 255, 255), thickness=10)
        cv2.fillConvexPoly(tresh_matrix, polygon, (255, 255, 255))

    return shapes

# Function to draw a pentagon on the matrix

def draw_pentagon(matrix, mask, shapes, center, size, angle=0, tresh_matrix=None, threshold=300, placed=None):
    x_center, y_center = center
    default_pos_angle = -90 + 180  # Align pentagon to the x-axis

    vertices = []
    for i in range(5):
        theta = np.radians(default_pos_angle + i * 360 / 5)  # Distribute vertices evenly around the center
        x = int(x_center + (size*0.75) * np.cos(theta))
        y = int(y_center + (size*0.75) * np.sin(theta))
        vertices.append((x, y))

    #print(vertices)
    #if any(x < threshold or x > matrix.shape[0] - threshold or y < threshold or y > matrix.shape[1] - threshold for x, y in vertices):
    #    matrix += 1000

    vertices_rotated = []

    for vertex in vertices:
        vertices_rotated.append(rotate_point(center, vertex, angle))

    #print(vertices_rotated)



    for i in range(5):
        x1, y1 = vertices_rotated[i]
        x2, y2 = vertices_rotated[(i + 1) % 5]
        if placed:
            write_webots_world((x1, y1), (x2, y2))
        cv2.line(matrix, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=10)

    polygon = np.array(vertices_rotated, np.int32)
    cv2.fillConvexPoly(mask, polygon, 1)
    shapes.append(["pentagon", polygon, angle, center])

    if tresh_matrix is not None:

        vertices_exp = expand_geometric_figure(vertices_rotated, 1000)
        #print(vertices_exp)

        #for i in range(5):
            #x1, y1 = vertices_exp[i]
            #x2, y2 = vertices_exp[(i + 1) % 5]
            #cv2.line(tresh_matrix, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=10)

        if any(x < threshold or x > tresh_matrix.shape[0] - threshold or y < threshold or y > tresh_matrix.shape[1] -
               threshold for x, y in vertices_exp):
            tresh_matrix += 1000

        polygon = np.array(vertices_exp, np.int32)
        cv2.fillConvexPoly(tresh_matrix, polygon, (255, 255, 255))

    return shapes



#=====================================================================

# Function to check if the shape can be placed without intersection
def can_place(matrix, mask, shape_fn, top_left, size,angle):

    if mask[top_left[0], top_left[1]] == 1:
        return False
    # temp_matrix = np.copy(matrix)
    temp_matrix = np.zeros(matrix.shape)
    temp_tresh_matrix = np.zeros(matrix.shape)
    temp_mask = np.zeros(mask.shape)
    temp_shapes = []
    temp_shapes = shape_fn(temp_matrix, temp_mask, temp_shapes, top_left, size, angle, temp_tresh_matrix)
    # breakpoint()
    return np.max(temp_tresh_matrix + matrix) <= 255
    # return np.sum(temp_matrix) == np.sum(matrix) + size * size


# Function to place a shape ensuring no intersection
def place_shape(matrix, mask, shapes, shape_fn, size,angle=0):
    placed = False
    while not placed:
        # x = random.randint(0, matrix.shape[0] - size)
        # y = random.randint(0, matrix.shape[1] - size)
        x = random.randint(0, matrix.shape[0]-1)
        y = random.randint(0, matrix.shape[1]-1)
        if can_place(matrix, mask, shape_fn, (x, y), size, angle):
            placed = True
            shapes = shape_fn(matrix, mask, shapes,(x, y), size,angle, placed=placed)
            # print the shape characteristics
            print(f'Placed {shapes[-1][0]} at {shapes[-1][3]} with a rotation of {angle}° and a size of {size}.')
            plt.imshow(matrix)
            plt.show()
    return shapes

# Main function to generate the matrix with shapes
def generate_matrix():
    # matrix = np.ones((3000, 3000, 3), dtype=np.uint8) * 0
    # mask = np.zeros((3000, 3000))
    matrix = np.ones((height, width, 3), dtype=np.uint8) * 0
    mask = np.zeros((height, width))

    shapes = []

    num_shapes = random.randint(5, 9)
    #num_shapes = 1
    print(num_shapes)
    shape_functions = [draw_square, draw_triangle, draw_pentagon]
    shape_sizes = [random.randint(800, 1200) for _ in range(num_shapes)]  # Example size range
    #breakpoint()
    for size in shape_sizes:
        shape_fn = random.choice(shape_functions)
        angle_range = angles_dict[shape_fn.__name__]
        angle = random.randint(angle_range[0], angle_range[1])
        #angle = 0
        #print(size)
        shapes = place_shape(matrix, mask, shapes,shape_fn, size,angle)

    return matrix, mask, shapes


# Function to display the matrix using matplotlib
def display_matrix(matrix):
    plt.imshow(matrix, cmap='Greys', interpolation='nearest', origin='lower')
    plt.show()

#Funtion to save the shapes to a csv file (ground truth)
def save_shapes_to_csv(shapes, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Shape", "Vertices", "Angle", "Center"])  # Write the header
        for shape in shapes:
            shape_type, polygon, angle, center = shape
            vertices_str = ";".join([f"({x},{y})" for x, y in polygon])
            writer.writerow([shape_type, vertices_str, angle, center])

if __name__ == "__main__":




    for i in range(4,5):
        i = 100



        matrix, mask, shapes = generate_matrix()
        #breakpoint()
        f.close()

        new_matrix = matrix.copy()
        new_matrix[matrix == 255] = 0
        new_matrix[matrix == 0] = 255
        display_matrix(new_matrix)

        #cv2.imshow('image', mask)
        #cv2.waitKey(0)

        plt.imsave(f'worlds/custom_maps/{map_name}_image.png', new_matrix, cmap='gray', origin='lower')

        with open(f'worlds/custom_maps/{map_name}_mask.pkl', 'wb') as f:
            pickle.dump(mask, f)

        with open(f'worlds/custom_maps/{map_name}_shapes.pkl', 'wb') as f:
            pickle.dump(shapes, f)

        save_shapes_to_csv(shapes, f'ground_truth/{map_name}_shapes.csv')
        #breakpoint()

        yaml_data = {
            'image': f'zzzmap_test_{i}_image.png',
            'resolution': 0.001,
            'origin': [0.0, 0.0, 0.0],
            'occupied_thresh': 0.65,
        }

        yaml_filename = f'worlds/custom_maps/zzzmap_test_{i}_config.yaml'
        with open(yaml_filename, 'w') as yaml_file:
            yaml.dump(yaml_data, yaml_file)

