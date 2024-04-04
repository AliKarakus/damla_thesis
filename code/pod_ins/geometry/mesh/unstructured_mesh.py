import numpy as np
import meshio
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


class Vertex:

    def __init__(self, id, coord):
        self.id = id
        self.x, self.y = coord


class Face:

    def __init__(self, id, verts, neighbors):
        self.id = id
        self.verts = verts
        self.neighbors = neighbors


class Cell:

    def __init__(self, id, verts):
        self.id = id
        self.verts = verts
        self.faces = []
        self.neighbors = []
        self.center = [-99, -99]


class UnstructuredMesh:

    def __init__(self, meshio_mesh):
        self.mesh = meshio_mesh
        self.etoe = np.loadtxt("results/mesh_etoe.txt", dtype=np.int64)  # Element to element
        self.etob = np.loadtxt("results/mesh_etob.txt", dtype=np.int64)  # Element to boundary
        self.etof = np.loadtxt("results/mesh_etof.txt", dtype=np.int64)  # Element to face

        # get triangle cellblock from meshio
        self.cellblock = None
        for cb in self.mesh.cells:
            if cb.type == 'triangle':
                self.cellblock = cb
        if self.cellblock == None:
            raise Exception("No quad cellblock")

        self.points = self.mesh.points[:, :2]  # x and y coordinates of points
        self.vertices = [Vertex(id, coords) for id, coords in enumerate(self.points)]
        self.num_vertices = len(self.vertices)

        self.cells = [Cell(id, verts) for id, verts in enumerate(self.cellblock.data)]
        self.num_cells = len(self.cells)

        self.calculate_cell_centers()
        self.set_cell_neighbors()
        self.faces = self.set_cell_faces()

    def calculate_cell_centers(self):
        for cell in self.cells:
            tot_x, tot_y = 0, 0
            for v in cell.verts:
                tot_x += self.vertices[v].x
                tot_y += self.vertices[v].y
            cell.center = [tot_x / len(cell.verts), tot_y / len(cell.verts)]

    def set_cell_neighbors(self):
        for cell in self.cells:
            cell.neighbors = self.etoe[cell.id]

    def set_cell_faces(self):
        faces = {}
        for cell in self.cells:
            for i in range(len(cell.verts)):
                v1, v2 = cell.verts[i], cell.verts[(i + 1) % len(cell.verts)]
                face_verts = tuple(sorted([v1, v2]))
                face = faces.get(face_verts, None)
                if face == None:
                    n1 = cell.id
                    n2 = self.etoe[cell.id][i]
                    if n2 == -1:
                        n2 = n1
                    face = Face(len(faces), face_verts, [n1, n2])
                    faces[face_verts] = face
                cell.faces.append(face.id)
        return list(faces.values())


meshio_mesh = meshio.read("results/1t_sqr_unstruc/quad_ins_0000_0000.vtu")
mesh = UnstructuredMesh(meshio_mesh)

fig, ax = plt.subplots()
color_step = 1 / len(mesh.faces)
for i, face in enumerate(mesh.faces):
    v1, v2 = face.verts
    x1, y1 = mesh.points[v1]
    x2, y2 = mesh.points[v2]
    n1, n2 = face.neighbors
    c1 = mesh.cells[n1].center
    c2 = mesh.cells[n2].center
    ax.plot([x2, x1], [y2, y1], c=hsv_to_rgb([i * color_step, 1, 1]), linewidth=0.75)
    ax.plot([c2[0], c1[0]], [c2[1], c1[1]], c='black', linewidth=0.3)

# ax.scatter([n.x for n in um.nodes], [n.y for n in um.nodes])
ax.set_aspect('1.0')
plt.show()
