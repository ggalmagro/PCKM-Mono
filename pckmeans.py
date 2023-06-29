from base.kmeans import KMEANS
import numpy as np
from scipy.spatial import distance
import random
import warnings


class PCKmeans(KMEANS):
    distOrder = False  # valor p de la formula de la distancia de Minkowsky
    w = False  # Peso de las restricciones.

    def __init__(self, distOrder: int = 2, w: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = w
        self.distOrder = distOrder

    def initialize_prototypes(self) -> list:
        """
        Inicializacion de los centroides basandonos en los veciones creados por las restricciones.
        Incializacion propuesta por los autores de PCKmeans.
        :return: list -> centroides.
        """
        # Calculamos el centro de cada vecindario (centroides de los vecindarios)
        nhCenters = np.array([self.data[nh].mean(axis=0) for nh in self.neighborhoods])
        # calculamos el tamaño de cada vecindario
        nhSizes = np.array([len(nh) for nh in self.neighborhoods])

        # Ordenamos los centroides de cada vecindario en orden decreciente.
        prototypes = nhCenters[np.argsort(nhSizes)]

        # SI hay más vecindarios(grupos de restricciones) que k
        if len(self.neighborhoods) > self.k:
            # Seleccionamos los k vecindarios(grupos de restricciones) más grandes.
            prototypes = prototypes[-self.k:]
        # SI hay menos vecindarios(grupos de restricciones) que k
        elif len(self.neighborhoods) < self.k:
            # Si no hay restriciones
            if len(self.neighborhoods) == 0:
                # Seleccionamos aleatoriamente los centroides.
                rs = np.random.RandomState()
                prototypes = self.data[rs.permutation(self.data.shape[0])[:self.k]]
            # Si hay al menos un centroide, se seleccionan aleatoriamente el resto
            else:
                rcc = self.data[np.random.choice(self.data.shape[0], self.k - len(self.neighborhoods), replace=False),
                      :]
                prototypes = np.concatenate([prototypes, rcc])

        return prototypes

    def assign_clusters(self, prototypes: list, cluster: list) -> list:
        """
        Funcion para asignar a cada instancia un cluster, se asigna el cluster que minimize la funcion objetivo

        :param prototypes: el valor actual de los centroides
        :param cluster: lista de etiquetas de asignación a cada cluster
        :return: lista de etiquetas de asignación a cada cluster actualizada
        """

        for j in range(self.maxTry):
            try:
                # Asignamos el Cluster
                r = list(range(self.data.shape[0]))
                random.shuffle(r)
                for x_i in r:
                    distances = [self.objective_function(x_i, c_i, cluster, prototypes) for c_i in range(self.k)]
                    # Distancia absoluta
                    cluster[x_i] = np.argmin(np.absolute(distances))

                #self.plot_me(cluster, prototypes)

                # Comprobacion clusters vacios.
                samplesInCLuster = np.bincount(cluster, minlength=self.k)
                emptyCluster = np.where(samplesInCLuster == 0)[0]

                if len(emptyCluster) > 0:
                    warnings.warn('PCKmeans: [W] Cluster sin instancias, reiniciando asignacion ...')
                    raise Exception()

                return cluster

            except:
                cluster = np.array([-1 for i in range(len(self.data))])
                prototypes = self.initialize_prototypes()

        return 0

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

        # Distancia del cluster al centroide
        if self.monotonic:
            # Con restriccion de monotonia.
            dist = np.subtract(self.data[id], prototypes[value]).sum()
            # Distancia absoluta
            dist = abs(dist)
        else:
            # Sin restriccion de monotonia.
            dist = distance.minkowski(self.data[id], prototypes[value], p=self.distOrder)
            # Distancia absoluta
            dist = abs(dist)

        # Suma de W por cada restriccion que viole
        mlp = sum([self.w for element in self.mustLink[id] if cluster[element] > 0 and cluster[element] != value])
        clp = sum([self.w for element in self.cannotLink[id] if cluster[element] == value])

        return dist + mlp + clp

    def __str__(self):
        if self.monotonic:
            return "PC-MONO: K = {}".format(self.k)
        else:
            return "PC: K = {}".format(self.k)
