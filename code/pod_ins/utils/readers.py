import concurrent.futures
import os
import time
from xml.etree import ElementTree

import meshio
import numpy as np
from geometry.mesh.cartesian_mesh import CartesianStructuredMesh
from geometry.mesh.rotational_mesh import RotationalStructuredMesh
from snapshots import Snapshots
from tqdm import tqdm

BAR_FMT_STR = '{desc:<20.20s}: {percentage:3.0f}% |{bar}| {n_fmt:>5.5s}/{total_fmt:<5.5s} [{elapsed}]'


def create_mesh_from_file(file_path):
    print(f'Read mesh from {file_path}...'.ljust(65), end='', flush=True)
    start = time.monotonic()
    meshio_mesh = meshio.read(file_path)
    mesh = CartesianStructuredMesh(meshio_mesh)
    took = time.monotonic() - start
    print(f'[{int(took//60):02d}:{int(took%60):02d}]')
    return mesh


def extract_point_data(file_path):
    point_data = {}
    tree = ElementTree.parse(file_path)
    root = tree.getroot()
    # Find all DataArray elements
    point_data_elements = root.iterfind('./UnstructuredGrid/Piece/PointData/DataArray')
    for i, point_data_elem in enumerate(point_data_elements):
        name = point_data_elem.get('Name', f'pointdata_{i:03d}')
        data = np.fromstring(point_data_elem.text, dtype=np.float32, sep=' ')
        if 'NumberOfComponents' in point_data_elem.attrib:
            noc = int(point_data_elem.get('NumberOfComponents', '-1'))
            if noc > 1:
                data = data.reshape(-1, noc)
        point_data[name] = data
        # print(f'Added {name} with {data.shape}')
    return point_data


def load_results(directory, workers=8):
    file_list = [os.path.join(directory, fname) for fname in sorted(os.listdir(directory))]
    nfiles = len(file_list)
    # Read one file for the mesh
    mesh = create_mesh_from_file(file_list[0])
    snapshots = Snapshots(nfiles)
    # Read every file and add results to snapshots
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor, \
         tqdm(total=nfiles,ncols=72,desc='Read Results',bar_format=BAR_FMT_STR) as pbar:
        futures = {}
        for i, file_path in enumerate(file_list):
            future = executor.submit(extract_point_data, file_path)
            pbar.write(f'Submited for reading {file_path:30.20s} [{i:>5d}/{nfiles:>5d}]', end='\r')
            futures[future] = i
        for future in concurrent.futures.as_completed(futures):
            pbar.update(1)
            time_idx = futures[future]
            point_data = future.result()['Velocity']
            snapshots.add_snapshot_data(time_idx, point_data[:, 0], point_data[:, 1])
    return mesh, snapshots
