import numpy as np
from geometry.cell import cell
from geometry.node import node


class CartesianStructuredMesh:
    ### initializing class attributes
    mesh = None

    # length scales
    number_nodes = None  # number of nodes
    number_cells = None  # number of cells

    # coordinates
    points = None  # cartesian node coordinates

    # subclass attributes
    nodes = None
    cells = None

    # problem attributes
    volume_weights = None  # skalar product weights in state based space

    def __init__(self, mesh):

        self.mesh = mesh  # handing over meshio object
        # number of nodes
        self.number_nodes = len(self.mesh.points)
        # number of cells
        quad_cellblock = None
        for cellblock in self.mesh.cells:
            if cellblock.type == 'quad':
                quad_cellblock = cellblock
        if quad_cellblock == None:
            raise Exception("No quad cellblock")
        self.number_cells = len(quad_cellblock)  # len(self.mesh.cells[0][1])

        # node coordinates
        self.points = self.mesh.points[:, :2]
        self.points = np.around(self.points, 12)

        self.xlevels = np.unique(self.points[:, 0])
        self.ylevels = np.unique(self.points[:, 1])

        # initializing node and cell instances enumerate(self.mesh.cells[0][1])
        self.cells = [cell(nodes, i) for i, nodes in enumerate(quad_cellblock.data)]
        self.nodes = [node(x, y, i) for i, (x, y) in enumerate(self.points)]
        for i, n in enumerate(self.nodes):
            n.u_value, n.v_value = self.mesh.point_data['Velocity'][i]

        # computing cell and node based attributes
        self.compute_cell_volumes()
        self.compute_cell_centers()

        # compute node volumes
        self.compute_node_volume()
        #compute node based weights
        self.compute_volume_weights()
        # assert node neighbors
        self.compute_node_neighbors()
        self.classify_nodes()

    def node_coords(self):
        return [n.x for n in self.nodes], [n.y for n in self.nodes]

    def compute_cell_volumes(self):
        # computing cell volumes with cell subroutine
        for cel in self.cells:
            cel.compute_volume(self)

    def compute_cell_centers(self):
        # computing cell centers with cell subroutine
        for cel in self.cells:
            cel.compute_center(self)

    def compute_node_volume(self):
        # iterate cells
        for cel in self.cells:
            node_list = cel.nodes
            # iterate nodes
            for node_idx in node_list:
                # assert weighted cell volume to cell constituting nodes
                self.nodes[node_idx].volume += cel.volume / len(node_list)

    def compute_volume_weights(self):
        weights = np.empty(self.number_nodes)
        for nod in self.nodes:
            weights[nod.i] = nod.volume
        self.volume_weights = np.concatenate([weights, weights])

    def compute_node_neighbors(self):
        # iterate over radius levels
        for i, y in enumerate(self.ylevels):
            # find indexes on same radius level
            same_y = self.points[:, 1] == y
            same_y_indices = np.where(same_y == True)[0]

            # find indices of next y level
            if y != self.ylevels[-1]:
                next_y = self.points[:, 1] == self.ylevels[i + 1]
                next_y_indices = np.where(next_y == True)[0]
            for j, x in enumerate(self.xlevels):
                # find indexes of same x levelnod
                if (-0.5 < x < 0.5) and (-0.5 < y < 0.5):
                    continue
                same_x = self.points[:, 0] == x
                same_x_indices = np.where(same_x == True)[0]
                index = int(np.intersect1d(same_x_indices, same_y_indices))

                # if we are not on the right edge
                if x != self.xlevels[-1]:
                    next_x = self.points[:, 0] == self.xlevels[j + 1]
                    next_x_indices = np.where(next_x == True)[0]
                    if (-0.5 >= y or y >= 0.5) or (-0.5 > x or x >= 0.5):
                        right = int(np.intersect1d(next_x_indices, same_y_indices))
                        self.nodes[index].r = right
                        self.nodes[right].l = index
                # if we are not on the top edge
                if y != self.ylevels[-1]:
                    if (-0.5 > y or y >= 0.5) or (-0.5 >= x or x >= 0.5):
                        up = int(np.intersect1d(next_y_indices, same_x_indices))
                        self.nodes[index].u = up
                        self.nodes[up].b = index

    def classify_nodes(self):
        self.inflow_node_indices = []
        self.outflow_node_indices = []
        self.wall_node_indices = []
        self.boundary_node_indices = []
        for node in self.nodes:
            if node.x == -8:
                self.inflow_node_indices.append(node.i)
                continue
            if node.x == 17:
                self.outflow_node_indices.append(node.i)
                continue
            if node.y == -12.5 or node.y == 12.5:
                self.boundary_node_indices.append(node.i)
                continue
            if -0.5 <= node.x <= 0.5 and -0.5 <= node.y <= 0.5:
                self.wall_node_indices.append(node.i)
                continue
        self.inflow_node_indices = set(self.inflow_node_indices)
        self.outflow_node_indices = set(self.outflow_node_indices)
        self.wall_node_indices = set(self.wall_node_indices)
        self.boundary_node_indices = set(self.boundary_node_indices)

    def finite_differences(self, data, compute_laplacian=False):
        # initialize data vectors
        resdx = np.zeros(self.number_nodes)
        resdy = np.zeros(self.number_nodes)
        if compute_laplacian:
            reslaplacian = np.zeros(self.number_nodes)

        # Compute derivatives using central differencing
        for nod in self.nodes:
            # 5 point stencil indizes
            i = nod.i
            r = nod.r  # special boundary case
            l = nod.l
            u = nod.u
            b = nod.b

            if nod.i in self.inflow_node_indices:
                # Inflow nodes don't have left neighbor
                dy = 0
                dx = (data[i] - data[r]) / (nod.x - self.nodes[r].x)
            elif nod.i in self.outflow_node_indices:
                # Outflow nodes don't have right neighbor
                dx = 0
                if not u:
                    dy = (data[i] - data[b]) / (nod.y - self.nodes[b].y)
                elif not b:
                    dy = (data[u] - data[i]) / (self.nodes[u].y - nod.y)
                else:
                    dy = (data[u] - data[b]) / (self.nodes[u].y - self.nodes[b].y)
            elif nod.i in self.boundary_node_indices:
                # Boundary nodes does not have upper or lower nodes
                dx = (data[l] - data[r]) / (self.nodes[l].x - self.nodes[r].x)
                dy = 0
            elif nod.i in self.wall_node_indices:
                # 1. Bottom wall
                if not (u):
                    dx = 0
                    dy = (data[i] - data[b]) / (nod.y - self.nodes[b].y)
                # 2. Upper wall
                if not (b):
                    dx = 0
                    dy = (data[u] - data[i]) / (self.nodes[u].y - nod.y)
                # 3. Right wall
                if not (l):
                    dx = (data[i] - data[r]) / (nod.x - self.nodes[r].x)
                    dy = 0
                # 4. Left wall
                if not (r):
                    dx = (data[l] - data[i]) / (self.nodes[l].x - nod.x)
                    dy = 0
            else:
                dx = (data[l] - data[r]) / (self.nodes[l].x - self.nodes[r].x)
                dy = (data[u] - data[b]) / (self.nodes[u].y - self.nodes[b].y)

            resdx[i] = dx
            resdy[i] = dy
            #####
            if compute_laplacian:

                if nod.i in self.inflow_node_indices:
                    # Inflow nodes don't have left neighbor
                    two_over_neighbor = self.nodes[r].r
                    ddx = (data[i] - 2 * data[r] + data[two_over_neighbor]) / (
                        (nod.x - self.nodes[r].x) * (self.nodes[r].x - self.nodes[two_over_neighbor].x))
                    ddy = 0
                elif nod.i in self.outflow_node_indices:
                    # Outflow nodes don't have right neighbor
                    ddx = 0
                    if not u:
                        # Upper right corner
                        two_over_neighbor = self.nodes[b].b
                        ddy = (data[i] - 2 * data[b] + data[two_over_neighbor]) / (
                            (nod.y - self.nodes[b].y) * (self.nodes[b].y - self.nodes[two_over_neighbor].y))
                    elif not b:
                        # Lower right corner
                        two_over_neighbor = self.nodes[u].u
                        ddy = (data[two_over_neighbor] - 2 * data[u] + data[i]) / (
                            (self.nodes[two_over_neighbor].y - self.nodes[u].y) * (self.nodes[u].y - nod.y))
                    else:
                        ddy = (data[u] - 2 * data[i] + data[b]) / ((self.nodes[u].y - nod.y) *
                                                                   (nod.y - self.nodes[b].y))
                elif nod.i in self.boundary_node_indices:
                    # Boundary nodes does not have upper or lower nodes
                    ddx = (data[l] - 2 * data[i] + data[r]) / ((self.nodes[l].x - nod.x) * (nod.x - self.nodes[r].x))
                    ddy = 0
                elif nod.i in self.wall_node_indices:
                    # 1. Bottom wall
                    if not (u):
                        two_over_neighbor = self.nodes[b].b
                        ddx = 0
                        ddy = (data[i] - 2 * data[b] + data[two_over_neighbor]) / (
                            (nod.y - self.nodes[b].y) * (self.nodes[b].y - self.nodes[two_over_neighbor].y))
                    # 2. Upper wall
                    elif not (b):
                        two_over_neighbor = self.nodes[u].u
                        ddx = 0
                        ddy = (data[two_over_neighbor] - 2 * data[u] + data[i]) / (
                            (self.nodes[two_over_neighbor].y - self.nodes[u].y) * (self.nodes[u].y - nod.y))
                    # 3. Right wall
                    elif not (l):
                        two_over_neighbor = self.nodes[r].r
                        ddx = (data[i] - 2 * data[r] + data[two_over_neighbor]) / (
                            (nod.x - self.nodes[r].x) * (self.nodes[r].x - self.nodes[two_over_neighbor].x))
                        ddy = 0
                    # 4. Left wall
                    elif not (r):
                        two_over_neighbor = self.nodes[l].l
                        ddx = (data[two_over_neighbor] - 2 * data[l] + data[i]) / (
                            (self.nodes[two_over_neighbor].x - self.nodes[l].x) * (self.nodes[l].x - nod.x))
                        ddy = 0
                else:
                    ddx = (data[l] - 2 * data[i] + data[r]) / ((self.nodes[l].x - nod.x) * (nod.x - self.nodes[r].x))
                    ddy = (data[u] - 2 * data[i] + data[b]) / ((self.nodes[u].y - nod.y) * (nod.y - self.nodes[b].y))

                laplacian = ddx + ddy
                reslaplacian[i] = laplacian

        # structured return
        if compute_laplacian:
            return [resdx, resdy, reslaplacian]
        else:
            return [resdx, resdy]