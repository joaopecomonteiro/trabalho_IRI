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
def draw_square(matrix, mask, shapes, top_left, size):
    x, y = top_left
    #if x + size < matrix.shape[0] and y + size < matrix.shape[1]:
    #    matrix[x:x + size, y:y + size] = 1
    if  x+size > matrix.shape[0]-100 or x < 100 or y < 100 or y+size > matrix.shape[0]-100:
        matrix += 1000
    cv2.line(matrix, (x, y), (x, y+size), color=(255, 255, 255),thickness=10)
    cv2.line(matrix, (x, y), (x+size, y), color=(255, 255, 255),thickness=10)
    cv2.line(matrix, (x+size, y), (x+size, y+size), color=(255, 255, 255),thickness=10)
    cv2.line(matrix, (x, y+size), (x+size, y+size), color=(255, 255, 255),thickness=10)

    polygon = np.array([[x, y], [x, y+size], [x+size, y+size], [x+size, y]], np.int32)
    cv2.fillConvexPoly(mask, polygon, 1)
    shapes.append(["square", polygon])

    return shapes




# Function to draw a triangle on the matrix
def draw_triangle(matrix, mask, shapes, top_left, size):
    x, y = top_left
    if size % 2 != 0:
        size += 1
    height = int((np.sqrt(3)/2) * size)  # Height of the equilateral triangle
    if y-height < 100 or x+size > matrix.shape[0]-100 or x < 100:
        matrix += 1000

    cv2.line(matrix, (x, y), (x+size, y), color=(255, 255, 255), thickness=10)
    cv2.line(matrix, (x, y), (int(x+size/2), y-height), color=(255, 255, 255), thickness=10)
    cv2.line(matrix, (x+size, y), (int(x+size/2), y-height), color=(255, 255, 255), thickness=10)

    polygon = np.array([[x, y], [x+size, y], [int(x+size/2), y-height]], np.int32)
    cv2.fillConvexPoly(mask, polygon, 1)

    shapes.append(["triangle", polygon])
    return shapes

# Function to check if the shape can be placed without intersection
def can_place(matrix, mask, shape_fn, top_left, size):
    #temp_matrix = np.copy(matrix)
    temp_matrix = np.zeros(matrix.shape)
    temp_mask = np.zeros(mask.shape)
    temp_shapes = []
    temp_shapes = shape_fn(temp_matrix, temp_mask, temp_shapes,top_left, size)
    #breakpoint()
    return np.max(temp_matrix + matrix) <= 255
    #return np.sum(temp_matrix) == np.sum(matrix) + size * size


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

    num_shapes = random.randint(3, 5)

    #num_shapes = 1
    print(num_shapes)
    shape_functions = [draw_triangle, draw_square]
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
    for i in range(10):
        matrix, mask, shapes = generate_matrix()
        #breakpoint()


        new_matrix = matrix.copy()
        new_matrix[matrix == 255] = 0
        new_matrix[matrix == 0] = 255
        display_matrix(new_matrix)


        plt.imsave(f'worlds/custom_maps/map_test_{i}_image.png', new_matrix, cmap='gray')

        with open(f'worlds/custom_maps/map_test_{i}_mask.pkl', 'wb') as f:
            pickle.dump(mask, f)

        with open(f'worlds/custom_maps/map_test_{i}_shapes.pkl', 'wb') as f:
            pickle.dump(shapes, f)
        #breakpoint()
    #import skimage


    #plt.imshow(image);plt.colorbar();plt.show()
    #display_matrix(image)
    #breakpoint()
    #plt.imsave('worlds/custom_maps/teste.png', new_matrix, cmap='gray')

