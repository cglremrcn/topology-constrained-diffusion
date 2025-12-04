import numpy as np 
import matplotlib.pyplot as plt 
import gudhi

def generate_circle_data(n_points = 1000):

    theata = np.random.uniform(0,2*np.pi, n_points)
    x = np.cos(theata)
    y = np.sin(theata)
    return np.stack((x,y), axis = 1)

def generate_disk_data(n_points=1000):

    theata = np.random.uniform(0,2*np.pi,n_points)
    r = np.random.uniform(0,1,n_points) 
    x = r*np.cos(theata)
    y = r*np.sin(theata)
    return np.stack((x,y), axis = 1)

def get_topology(point_cloud):

    rips_complex = gudhi.RipsComplex(points = point_cloud, max_edge_length = 0.5)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = 2)
    simplex_tree.persistence()

    betti = simplex_tree.betti_numbers()
    return betti


def generate_nested_rings(n_points=200, noise=0.05):
    n_inner = n_points // 2
    n_outer = n_points - n_inner
    
    angles_in = np.random.uniform(0, 2 * np.pi, n_inner)
    x_in = (0.5 * np.cos(angles_in)) + np.random.normal(0, noise, n_inner)
    y_in = (0.5 * np.sin(angles_in)) + np.random.normal(0, noise, n_inner)
    
    angles_out = np.random.uniform(0, 2 * np.pi, n_outer)
    x_out = (1.0 * np.cos(angles_out)) + np.random.normal(0, noise, n_outer)
    y_out = (1.0 * np.sin(angles_out)) + np.random.normal(0, noise, n_outer)
    
    x = np.concatenate([x_in, x_out])
    y = np.concatenate([y_in, y_out])
    
    return np.stack([x, y], axis=1)



if __name__ == "__main__":
    circle = generate_circle_data()
    disk = generate_disk_data()
    circle_betti = get_topology(circle)
    print("Circle betti:",circle_betti)
    disk_betti = get_topology(disk)
    print("Disk betti:",disk_betti)

    plt.figure(figsize = (10,7))
    

    plt.subplot(1,2,1)
    plt.scatter(circle[:,0],circle[:,1])
    plt.title("Circle")
    plt.axis("equal")

    plt.subplot(1,2,2)
    plt.scatter(disk[:,0],disk[:,1])
    plt.title("Disk")
    plt.axis("equal")
    plt.show()