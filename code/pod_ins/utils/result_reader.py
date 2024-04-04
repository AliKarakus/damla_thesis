import numpy as np
import os
import sys
import concurrent.futures
import meshio
from tqdm import tqdm
if __name__ == '__main__':
    sys.path.append(os.path.dirname(sys.path[0]))
from geometry.mesh.rotational_mesh import RotationalStructuredMesh
from geometry.mesh.cartesian_mesh import CartesianStructuredMesh


def load_file(file_path, idx):
    namespace = {}
    contents = open(file_path, 'r').read()
    exec(contents, namespace)
    return np.array(namespace['DGNu']), np.array(namespace['DGNv'])


def read_from_dir(directory):
    number_of_files = len(os.listdir(directory))
    TimeStep_DGNu = [None] * number_of_files
    TimeStep_DGNv = [None] * number_of_files
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        futures = {}
        for i, file_name in enumerate(sorted(os.listdir(directory))):
            print(f'Submited for reading {directory}...', end='\r')
            file_path = os.path.join(directory, file_name)
            future = executor.submit(load_file, file_path, i)
            futures[future] = i
        print()
        complete = 0
        for future in concurrent.futures.as_completed(futures):
            print(f'Read {complete}/{number_of_files}', end='\r')
            complete += 1
            u, v = future.result()
            TimeStep_DGNu[futures[future]] = u
            TimeStep_DGNv[futures[future]] = v
        print()
        return np.array(TimeStep_DGNu), np.array(TimeStep_DGNv), number_of_files


def create_mesh_from_file(file_path):
    meshio_mesh = meshio.read(file_path)
    mesh = CartesianStructuredMesh(meshio_mesh)
    return mesh


def load_meshes(directory, workers=8):
    nfiles = len(os.listdir(directory))
    meshes = [None] * nfiles
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor, tqdm(total=nfiles, ncols=72) as pbar:
        futures = {}
        for i, file_name in enumerate(sorted(os.listdir(directory))):
            file_path = os.path.join(directory, file_name)
            future = executor.submit(create_mesh_from_file, file_path)
            print(f'Submited for reading {file_path} = {i}...', end='\r')
            futures[future] = i
        print()
        complete = 0
        for future in concurrent.futures.as_completed(futures):
            # print(f'Read {complete}/{number_of_files}', end='\r')
            pbar.update(1)
            meshes[futures[future]] = future.result()
        return meshes
