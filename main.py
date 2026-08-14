import numpy as np 
import matplotlib.pyplot as plt

def plot_points(labeled_sets: list, title: str = "Points" ):
    """
    Plot one or more sets of points and give each plotted set a label

        labeled_set:  a list of (point_array, "label") tuples,
                      where point_array is a Nx2 array holding N number of points
                      label is a string for those points

        title:        title for the entire plot

    """
    for points, label in labeled_sets:
        # Collect coordinates from array for plotting
        x_coordinates = points[:, 0]
        y_coordinates = points[:, 1]

        plt.scatter(x_coordinates, y_coordinates, label = label)

    plt.axis('equal')
    plt.grid(True)
    plt.title(title)
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.legend()
    plt.show()


def transform_points(points, angle_degrees, translation):
    """
    Simulate the OR: rotate then translate the scanned points to make
    the 'patient on the table' pose of shin.

    points:        N×2 array, each row [x, y]
    angle_degrees: rotation angle (degrees, counterclockwise)
    translation:   (tx, ty) to shift by, in mm
    returns:       N×2 array of transformed points
    """
    theta = np.radians(angle_degrees)

    # Rotation matrix transformation based on desired rotation
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    # Rotate + translate points based on inputs to get transformed matrix
    rotated = points @ R.T
    transformed = rotated + translation

    return transformed

def calc_centroid(points):
    """
    Given a point data set, return its centroid

    """
    return points.mean(axis = 0)

def recenter_matrix(points, centroid):
    """
    Given a set of points and its centroid, return its recentered data with the centroid at the origin

    """
    return points - centroid

def calc_angle(from_recentered: list, to_recentered: list):
    """
    Given two recentered point sets, return the rotation angle (in degrees)
    that best rotates the 'from' set onto the 'to' set.

    """
    fx = from_recentered[:, 0]
    fy = from_recentered[:, 1]
    tx = to_recentered[:, 0]
    ty = to_recentered[:, 1]

    cross_sum = np.sum(fx * ty - fy * tx)  # sum of the cross products of each point pair
    dot_sum = np.sum(fx * tx + fy * ty)    # sum of the dot products of each point pair

    angle = np.degrees(np.arctan2(cross_sum, dot_sum))
    
    return angle

def register(from_points: list, to_points: list):
    """
    Given two identical, but transformed data sets, the register function will return the angle transformation
    and translation to get transform from_data to to_data.

    For function purposes we will mainly be registering OR position of tool to respective SCAN position
        : register(OR, scan) --> from OR to scan

    (if one would like to know scan point in OR space, then input register(scan, OR) --> from scan to OR)
    
    """
    from_centroid = calc_centroid(from_points)
    to_centroid = calc_centroid(to_points)

    from_recenter = recenter_matrix(from_points, from_centroid)
    to_recenter = recenter_matrix(to_points, to_centroid)

    angle_diff = calc_angle(from_recenter, to_recenter) # returned angle + angle used to solve for Rotation matrix for translation

    theta = np.radians(angle_diff)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    translation = to_centroid - R @ from_centroid # translation found in isolation

    return angle_diff, translation


    




# Each row is one point [x, y], in millimeters.
# Truth Scan points of Tibia. Adult shin is around 300mm-470mm long and 40mm-95mm wide
scan_points = np.array([
    [0, 0],
    [-20, 100],
    [20, 200],
    [-30, 300],
    [30, 375],
    [0, 450],
])

or_points = transform_points(scan_points, 90, (200, 300))

plot_points([(scan_points, "scan space"), (or_points, "OR space translation + rotation")], title="Scan vs OR")

