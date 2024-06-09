import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import math
from skimage.draw import line
import cv2
import pickle

shapes = []

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


def draw_square(matrix, mask, shapes, top_left, size, tresh_matrix=None, threshold=300):
    x, y = top_left
    #if x + size < matrix.shape[0] and y + size < matrix.shape[1]:
    #    matrix[x:x + size, y:y + size] = 1
    if x+size > matrix.shape[0]-threshold or x < threshold or y < threshold or y+size > matrix.shape[0]-threshold:
        matrix += 1000

    A = (x, y)
    B = (x + size, y)
    C = (x + size, y + size)
    D = (x, y + size)
    #print(A, B, C, D)

    cv2.line(matrix, A, D, color=(255, 255, 255), thickness=10)
    cv2.line(matrix, A, B, color=(255, 255, 255), thickness=10)
    cv2.line(matrix, B, C, color=(255, 255, 255), thickness=10)
    cv2.line(matrix, D, C, color=(255, 255, 255), thickness=10)

    if tresh_matrix is not None:

        if x+size > tresh_matrix.shape[0]-threshold or x < threshold or y < threshold or y+size > tresh_matrix.shape[0]-threshold:
            tresh_matrix += 1000

        A_exp, B_exp, C_exp, D_exp = expand_geometric_figure([A, B, C, D], 300)
        #print(A_exp, B_exp, C_exp, D_exp)
        polygon = np.array([A_exp, B_exp, C_exp,D_exp], np.int32)
        #cv2.line(tresh_matrix, A_exp, D_exp, color=(255, 255, 255), thickness=10)
        #.line(tresh_matrix, A_exp, B_exp, color=(255, 255, 255), thickness=10)
        #cv2.line(tresh_matrix, B_exp, C_exp, color=(255, 255, 255), thickness=10)
        #cv2.line(tresh_matrix, D_exp, C_exp, color=(255, 255, 255), thickness=10)
        cv2.fillConvexPoly(tresh_matrix, polygon, (255, 255, 255))

    polygon = np.array([A, B, C, D], np.int32)
    cv2.fillConvexPoly(mask, polygon, 1)
    shapes.append(["square", polygon])

    return shapes

# Function to draw a triangle on the matrix
def draw_triangle(matrix, mask, shapes, top_left, size, tresh_matrix = None, threshold=300):
    x, y = top_left
    if size % 2 != 0:
        size += 1
    height = int((np.sqrt(3)/2) * size)  # Height of the equilateral triangle
    if y-height < threshold or x+size > matrix.shape[0]-threshold or x < threshold:
        matrix += 1000

    A = (x, y)
    B = (x + size, y)
    C = (int(x + size / 2), y - height)
    #print('#')
    #print(A, B, C)

    cv2.line(matrix, A, B, color=(255, 255, 255), thickness=10)
    cv2.line(matrix, A, C, color=(255, 255, 255), thickness=10)
    cv2.line(matrix, B, C, color=(255, 255, 255), thickness=10)

    polygon = np.array([A, B, C], np.int32)
    cv2.fillConvexPoly(mask, polygon, 1)

    shapes.append(["triangle", polygon])

    if tresh_matrix is not None:

        if y - height < threshold or x + size > tresh_matrix.shape[0] - threshold or x < threshold:
            tresh_matrix += 1000

        A_exp, B_exp, C_exp = expand_geometric_figure([A, B, C], 300)
        #print(A_exp, B_exp, C_exp)

        polygon = np.array([A_exp, B_exp, C_exp], np.int32)
        #cv2.line(tresh_matrix, A_exp, B_exp, color=(255, 255, 255), thickness=10)
        #cv2.line(tresh_matrix, A_exp, C_exp, color=(255, 255, 255), thickness=10)
        #cv2.line(tresh_matrix, B_exp, C_exp, color=(255, 255, 255), thickness=10)
        cv2.fillConvexPoly(tresh_matrix, polygon, (255, 255, 255))

    return shapes

# Function to draw a pentagon on the matrix

def draw_pentagon(matrix, mask, shapes, center, size, tresh_matrix=None, threshold=300):
    x_center, y_center = center
    angle = -90  # Align pentagon to the x-axis

    vertices = []
    for i in range(5):
        theta = np.radians(angle + i * 360 / 5)  # Distribute vertices evenly around the center
        x = int(x_center + (size*0.75) * np.cos(theta))
        y = int(y_center + (size*0.75) * np.sin(theta))
        vertices.append((x, y))

    #print(vertices)
    if any(x < threshold or x > matrix.shape[0] - threshold or y < threshold or y > matrix.shape[1] - threshold for x, y in vertices):
        matrix += 1000

    for i in range(5):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % 5]
        cv2.line(matrix, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=10)

    polygon = np.array(vertices, np.int32)
    cv2.fillConvexPoly(mask, polygon, 1)
    shapes.append(["pentagon", polygon])

    if tresh_matrix is not None:

        if any(x < threshold or x > tresh_matrix.shape[0] - threshold or y < threshold or y > tresh_matrix.shape[1] - threshold for
               x, y in vertices):
            tresh_matrix += 1000

        vertices_exp = expand_geometric_figure(vertices, 300)
        #print(vertices_exp)

        #for i in range(5):
            #x1, y1 = vertices_exp[i]
            #x2, y2 = vertices_exp[(i + 1) % 5]
            #cv2.line(tresh_matrix, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=10)
        polygon = np.array(vertices_exp, np.int32)
        cv2.fillConvexPoly(tresh_matrix, polygon, (255, 255, 255))

    return shapes



#=====================================================================

# Function to check if the shape can be placed without intersection
def can_place(matrix, mask, shape_fn, top_left, size):

    if mask[top_left[0], top_left[1]] == 1:
        return False
    # temp_matrix = np.copy(matrix)
    temp_matrix = np.zeros(matrix.shape)
    temp_tresh_matrix = np.zeros(matrix.shape)
    temp_mask = np.zeros(mask.shape)
    temp_shapes = []
    temp_shapes = shape_fn(temp_matrix, temp_mask, temp_shapes, top_left, size, temp_tresh_matrix)
    # breakpoint()
    return np.max(temp_tresh_matrix + matrix) <= 255
    # return np.sum(temp_matrix) == np.sum(matrix) + size * size


# Function to place a shape ensuring no intersection
def place_shape(matrix, mask, shapes, shape_fn, size):
    placed = False
    while not placed:
        x = random.randint(0, matrix.shape[0] - size)
        y = random.randint(0, matrix.shape[1] - size)
        if can_place(matrix, mask, shape_fn, (x, y), size):
            shapes = shape_fn(matrix, mask, shapes,(x, y), size)
            placed = True
    return shapes

# Main function to generate the matrix with shapes
def generate_matrix():
    matrix = np.ones((3000, 3000, 3), dtype=np.uint8) * 0
    mask = np.zeros((3000, 3000))

    shapes = []

    num_shapes = random.randint(2, 5)

    #num_shapes = 1
    print(num_shapes)
    shape_functions = [draw_square,draw_triangle, draw_pentagon]
    shape_sizes = [random.randint(400, 900) for _ in range(num_shapes)]  # Example size range
    #breakpoint()
    for size in shape_sizes:
        shape_fn = random.choice(shape_functions)
        #print(size)
        shapes = place_shape(matrix, mask, shapes,shape_fn, size)

    return matrix, mask, shapes


# Function to display the matrix using matplotlib
def display_matrix(matrix):
    plt.imshow(matrix, cmap='Greys', interpolation='nearest')
    plt.show()



if __name__ == "__main__":
    for i in range(2,3):
        matrix, mask, shapes = generate_matrix()
        #breakpoint()


        new_matrix = matrix.copy()
        new_matrix[matrix == 255] = 0
        new_matrix[matrix == 0] = 255
        display_matrix(new_matrix)

        #cv2.imshow('image', mask)
        #cv2.waitKey(0)

        plt.imsave(f'worlds/custom_maps/zzzmap_test_{i}_image.png', new_matrix, cmap='gray')

        with open(f'worlds/custom_maps/zzzmap_test_{i}_mask.pkl', 'wb') as f:
            pickle.dump(mask, f)

        with open(f'worlds/custom_maps/zzzmap_test_{i}_shapes.pkl', 'wb') as f:
            pickle.dump(shapes, f)
        #breakpoint()
    #import skimage


    #plt.imshow(image);plt.colorbar();plt.show()
    #display_matrix(image)
    #breakpoint()
    #plt.imsave('worlds/custom_maps/teste.png', new_matrix, cmap='gray')

