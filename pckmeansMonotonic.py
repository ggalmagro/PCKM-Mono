import numpy as np
import networkx as nx


class PCKmeans:
    data = False
    k = False
    maxIter = False
    distOrder = False
    mustLink = False
    cannotLink = False
    neighborhoods = False
    clusters = False
    w = False
    prototypes = False
    grad = False
    M = False
    di = False
    do = False
    alpha = False

    def __init__(self, k: int = 2, w: int = 100, maxIter: int = 1000, distOrder: int = 2, grad: float = 1.1):

        # Numero de vecinos
        self.k = k
        # Numero máximo de iteraciones
        self.maxIter = maxIter
        # Valor de p para la distancia de Manhattan
        self.distOrder = distOrder
        # Peso de las restricciones
        self.w = w
        self.grad = grad

    def fit(self, x: np.ndarray, consMatrix: np.matrix = False, ml: list = [], cl: list = []):
        """
        Metodo fit para meter datos en el objeto y entrenar el COP-kmeans

        :param consMatrix: -> Matrix with constrains
        :param cl: List -> Cannot Link constrains
        :param ml: List -> Must Link constrains
        :param x: conjunto de datos de entrenamiento

        :return: Datos asignados a cada cluster
        """

        self.data = x

        if consMatrix:
            self.matrix_to_constrains(consMatrix)
        else:
            self.preprocess_constrains(ml, cl)

        self.calculate_clusters()
        self.calculate_m()
        self.calculate_di()
        self.calculate_do()
        self.calculate_alpha()

    def calculate_clusters(self):
        """
        Ejecuta algoritmo PC-kmeans.

        :return: asignacion de clusters a cada dato.
        """

        # initialization of prototypes based on the lambda neighborhoods from ML and :
        prototypes = self.initialize_prototypes()

        # Creating clusters
        n_cluster = [i for i in range(self.k)]
        cluster = np.array([-1 for i in range(len(self.data))])

        # TODO: proceso greedy, meter en una función que repita esto n veces por si se produce algun error

        # While max iterations
        for iteration in range(self.maxIter):

            # Assign cluser
            cluster = self.assign_clusters(prototypes, cluster)

            # Estimate mean
            new_prototypes = np.array([self.data[cluster == k].mean(0) for k in n_cluster])

            # If new_prototypes == prototypes then converge else update prototypes
            if np.all(new_prototypes == prototypes):
                break
            else:
                prototypes = np.array(new_prototypes)

        self.prototypes = prototypes
        self.clusters = cluster

    def objective_function(self, id: int, value: int, cluster: list, prototypes: list) -> float:
        """
        Funcion objetivo a minimizar, es la suma de la distanca de cada punto a un centroide junto con
        una penalización asociada a la violacion de alguna restricción

        :param id: id de la instancia sobre la que calculamos la funcion objetivo
        :param value: id del cluster al que queremos asignar la instancia
        :param cluster: lista de etiquetas de asignación a cada cluster
        :param prototypes: lista con los centroides
        :return: valor de la funcion objteivo a minimizar
        """
        #Distancia del cluster al centroide
        dist = np.subtract(self.data[id], prototypes[value]).sum()

        # Suma de W por cada restriccion que viole
        mlp = sum([self.w for element in self.mustLink[id] if cluster[element] > 0 and cluster[element] != value])
        clp = sum([self.w for element in self.cannotLink[id] if cluster[element] == value])

        return dist + mlp + clp

    def assign_clusters(self, prototypes: list, cluster: list) -> list:
        """
        Funcion para asignar a cada instancia un cluster, se asigna el cluster que minimize la funcion objetivo

        :param prototypes: el valor actual de los centroides
        :param cluster: lista de etiquetas de asignación a cada cluster
        :return: lista de etiquetas de asignación a cada cluster actualizada
        """

        # Asignamos el Cluster
        for x_i in range(self.data.shape[0]):
            distances = [self.objective_function(x_i, c_i, cluster, prototypes) for c_i in range(self.k)]
            # Distancia absoluta
            cluster[x_i] = np.argmin(np.absolute(distances))

        # Handle empty clusters
        samplesInCLuster = np.bincount(cluster, minlength=self.k)
        emptyCluster = np.where(samplesInCLuster == 0)[0]

        if len(emptyCluster) > 0:
            # raise Exception("Error! not possible to assign data to some cluster")
            print("Error! not possible to assign data to some cluster")

        return cluster

    def initialize_prototypes(self) -> list:
        """
        Inicialización de los centroides basandonos en los veciones creados por las restricciones

        :return: list -> centroides.
        """
        # Sort the indices p in decreasing size of N_p:
        # Calculate the centers of each neighborhoods
        nhCenters = np.array([self.data[nh].mean(axis=0) for nh in self.neighborhoods])
        # calculate the size of each neighborhoods
        nhSizes = np.array([len(nh) for nh in self.neighborhoods])

        # Sort centers of each neighborhoods based on decreasing size of each one
        prototypes = nhCenters[np.argsort(nhSizes)]

        # Check if there is the same number of prototypes than k:
        # If there is more prototypes than k
        if len(self.neighborhoods) > self.k:
            # Select K largest neighborhoods' centroids
            prototypes = prototypes[-self.k:]
        # If there is less prototypes than k
        elif len(self.neighborhoods) < self.k:
            # If there is 0 restrictions
            if len(self.neighborhoods) == 0:
                # random pick all prototypes
                rs = np.random.RandomState()
                prototypes = self.data[rs.permutation(self.data.shape[0])[:self.k]]
            # If there is at least one centroid, random select the others
            else:
                rcc = self.data[np.random.choice(self.data.shape[0], self.k - len(self.neighborhoods), replace=False), :]
                prototypes = np.concatenate([prototypes, rcc])

        return prototypes

    def preprocess_constrains(self, ml: list, cl: list):
        """
        Preprocesamos la lista de constrains como un diccionario de conjuntos.
        cada clave del diccionario hace referencia a los ids de las instancias
        para cada instancia almacenamos los items con los que hay un MUST LINK
        directa o indirictamente.

        :param ml: list -> MustLink constrains
        :param cl: list -> CannotLink constrains
        """
        # Create the dicts:
        mustLink, cannotLink, neighborhoodsM = self.create_dict(ml, cl)

        # Check for inconsistences
        for m in mustLink:
            for c in mustLink[m]:
                if c != m and c in cannotLink[m]:
                    raise Exception('Inconsistencia entre las restricciones %d y %d' % (m, c))

        # Creamos los centroides basado en los  de MustLink.
        self.neighborhoods = neighborhoodsM  # + neighborhoodsC
        self.mustLink, self.cannotLink = mustLink, cannotLink

    def create_dict(self, ml: list, cl: list) -> tuple:
        """
        Creamos los diccionarios a partir de una lista de vertices.

        :param ml: MUST LINK constrains
        :param cl: CANNOT LINK constrains
        :return: diccionario con todos los valores
        """
        # Inicializamos un grafo
        mlGraph = nx.Graph()
        n = self.data.shape[0]

        # Creamos un grafo de las restricciones MUST-LINK
        mlGraph.add_edges_from(ml)

        # Inicializamos un diccionario para almacenar las MustLink y las CannotLink
        mlDict = {i: set() for i in range(n)}
        clDict = {i: set() for i in range(n)}

        # Creamos el diccionario de MUSTLINK a partir de los nodos que foraman cada grafo
        for node in mlGraph.nodes():
            # Por cada nodo del un grafo conectado, añadimos al indice del diccionario que hace referencia
            # a dicho nodo, todos los DEMAS nodos que estan conectados.
            mlDict[node] = nx.node_connected_component(mlGraph, node) - {node}

        # Creamos el diccionatio de CANNOTLINK:
        # En primer lugar añadimos una a una las restricciones
        # Para cada par de restricciones i, j
        for i, j in cl:
            # Añadimos la restriccion i al indice j
            clDict[i].add(j)
            # Añadimos la restriccion j al indice i
            clDict[j].add(i)

        # Para indice del Diccionario CL
        for key, values in clDict.items():
            # Para cada valor de dicho indice
            for value in values:
                # Añadimos a CL[valor] CL[Indice]
                clDict[key] = clDict[key].union(mlDict[value])
                # De forma que si CL[3] = 2, 7 y CL[2] = 4, 5:
                # CL[3] = 2, 4, 5, 7

        # Para cada indice de MUSTLINK
        for key, values in mlDict.items():
            # Para cada elemento de indice de MUSTLINK
            for value in values:
                # Añadimos al indice == elemento de CANNOTLINK el valor del indice de MUSTLINK
                clDict[value] = clDict[value].union(clDict[key])
                # De forma que si 1, 2 y 3 tienen un enlace ML y 7 y 3 un enlace CL:
                # al indice CL[7] se les añada 1 y 2. De forma que 7 no se pueda unir a 1 o 2.

        return mlDict, clDict, [list(i) for i in nx.connected_components(mlGraph)]

    def matrix_to_constrains(self, consMatrix: np.matrix):
        """
        Transform constrains matrix into ml and cl constrains by
        iterating over the LOWER MATRIX TRIANGLE

        :param consMatrix: np.matrix -> Matrix with constrains
        """
        ml = []
        cl = []
        # Iterate over the Matrix
        for row_index in range(consMatrix.shape[0]):
            for column_index in range(consMatrix.shape[1]):
                # Only over lower triangle of a matrix
                if row_index > column_index:
                    value = consMatrix[row_index, column_index]
                    # If must Link
                    if value == 1:
                        ml.append([row_index, column_index])
                    # If cannot Link
                    if value == -1:
                        cl.append([row_index, column_index])

        self.preprocess_constrains(ml, cl)

    def __str__(self):
        return "PCK Means Monotonic: K = {}".format(self.k)

    def calculate_m(self):
        """
        Calcula la matriz M necesaria para los indices DI y DO.
        Se basa en la formula (7) de Rosenfeld et al. 2020

        :return: np.matrix k*k
        """

        n_cluster = [i for i in range(self.k)]
        m = np.zeros(shape=(self.k, self.k))

        # Iterate over cartesian product of clusters
        for i in n_cluster:
            for j in n_cluster:
                p1 = self.data[self.clusters == i]
                p2 = self.data[self.clusters == j]
                # For each pair of clusters we calculate the R_kl and R_lk index
                r_kl = self.calculate_r(p1, p2)
                r_lk = self.calculate_r(p2, p1)
                # Append the results to the matrix
                m[i, j] = r_kl + r_lk

        self.M = m

    # TODO: joint into single method calculate_di and calculate_do
    def calculate_di(self):
        """
        Calcula el indice DI (Dunn-like index) (6)
        propuesto en Rosenfeld et al. 2020

        :return: int
        """
        dividend = []
        divider = []

        # Iterate over the matrix
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):
                # if the element is in the diagonal append it to divider list
                if i == j:
                    divider.append(self.M[i, j])
                # if element is in the upper triangle(?) append it to dividend
                elif i < j:
                    dividend.append(self.M[i, j])

        # DI = min(item in diagonal) / max(item in upper triangle)
        self.di = np.min(dividend) / np.max(divider)

    def calculate_do(self):
        """
        Calcula el indice DO (Dunn-like index) (10)
        propuesto en Rosenfeld et al. 2020
        :return: int
        """

        dividend = []
        divider = []

        # Iterate over the cartesian product of the clusters
        for i in range(self.M.shape[0]):
            for j in range(self.M.shape[1]):

                p1 = self.data[self.clusters == i]
                p2 = self.data[self.clusters == j]
                # if both clusters are equal:
                if i == j:
                    # Calculate the R_jj and append it to divider
                    divider.append(self.calculate_r(p1, p2))
                # when i > j we calculate the R_ij and R_ji
                elif i < j:
                    # Calculate the max(R_ij, R_ji) and append it to dividend
                    dividend.append(np.max([self.calculate_r(p1, p2), self.calculate_r(p2, p1)]))

        # DO = min(dividend) / 2 * max(divider)
        self.do = np.min(dividend) / (2 * np.max(divider))

    def calculate_alpha(self, grad: float = False):
        """
        Calcula la matriz de preferencias alpha (13),(14),(15),(16)
        propuesta en  Rosenfeld et al. 2020

        :param grad: int, Grado en por el que la preferencia por cierto orden es mayor que por el inverso
        :return: np.matrix k*k
        """

        if not grad: grad = self.grad
        n_cluster = [i for i in range(self.k)]
        alpha = np.zeros(shape=(self.k, self.k))

        # Iterate over cartesian product of clusters
        for i in n_cluster:
            for j in n_cluster:
                p1 = self.data[self.clusters == i]
                p2 = self.data[self.clusters == j]
                # if both clusters are equal
                if i == j:
                    # there is no preference
                    alpha[i, j] = np.nan
                else:
                    r_ij = self.calculate_r(p1, p2)
                    r_ji = self.calculate_r(p2, p1)

                    # R_ij MUST be >= max(R_ii, R_jj)
                    if r_ij < np.max([self.calculate_r(p1, p1), self.calculate_r(p2, p2)]):
                        alpha[i, j] = np.nan
                    # R_ij MUST be > grad * R_ji:
                    elif r_ij <= grad * r_ji:
                        alpha[i, j] = np.nan
                    else:
                        if r_ji == 0:
                            alpha[i, j] = np.inf
                        else:
                            alpha[i, j] = r_ij/r_ji

        self.alpha = alpha

    def calculate_r(self, p1: np.ndarray, p2: np.ndarray) -> int:
        """
        Calcula R porpuesta por Rosenfeld et al. 2020 y
        definida en (7), (8) y (3)

        :param p1: np.array, Incluye todos los elementos que pertenecen al cluster p1
        :param p2: np.array, Incluye todos los elementos que pertenecen al cluster p2
        :return: int, el R para los clusters p1 y p2
        """

        pref = []
        # iterate over the cartesian product clusters p1 x p2
        for x_i in p1:
            for x_j in p2:
                # for each element calculate the preference (formula 2)
                pref.append(
                    sum([x_i[d] - x_j[d] if x_i[d] > x_j[d] else 0 for d in range(len(x_i))])
                )

        return sum(pref) / (p1.shape[0] * p2.shape[0])


if __name__ == '__main__':
    from sklearn.datasets._samples_generator import make_blobs

    X, y_true = make_blobs(n_samples=20, centers=4,
                           cluster_std=0.40, random_state=1)

    a = PCKmeans(k=3)
    a.fit(x=X, ml=[[0, 7],
                   [1, 3],
                   [1, 4],
                   [1, 5]], cl=[[0, 6]])
    print(a.clusters)
