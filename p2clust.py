import numpy as np
import random
import warnings
from base.kmeans import KMEANS


class P2Clust(KMEANS):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.monotonic = True

    def initialize_prototypes(self) -> list:
        """
        Se inicializan los prototipos de forma aleatoria.

        :return: Prototipos
        """
        rs = np.random.RandomState()
        return self.data[rs.permutation(self.data.shape[0])[:self.k]]

    def assign_clusters(self, prototypes: list, cluster: list) -> list:
        """
        Funcion para asignar a cada instancia un cluster, se asigna el cluster que minimize la funcion objetivo.

        :param prototypes: el valor actual de los centroides
        :param cluster: lista de etiquetas de asignación a cada cluster
        :return: lista de etiquetas de asignación a cada cluster actualizada
        """
        # Definimos un numero maximo de intentos
        for j in range(self.maxTry):
            try:
                # Asignacion aleatoria de los datos
                r = list(range(self.data.shape[0]))
                random.shuffle(r)
                for i in r:
                    x = self.data[i]
                    # Calcularmos la distancia monotonica de cada dato a los centroides
                    distances = [np.subtract(x, prototypes[k]).sum() for k in [i for i in range(self.k)]]
                    # Asignamos al cluster con centroide mas cercano.
                    cluster[i] = np.argmin(np.absolute(distances))

                # Comprobacion clusters vacios.
                samplesInCLuster = np.bincount(cluster, minlength=self.k)
                emptyCluster = np.where(samplesInCLuster == 0)[0]

                if len(emptyCluster) > 0:
                    warnings.warn('P2CLUST: [W] Cluster sin instancias, reiniciando asignacion ...')
                    raise Exception()

                return cluster

            except:
                cluster = np.array([-1 for i in range(len(self.data))])
                prototypes = self.initialize_prototypes()
        return 0

    def __str__(self):
        return "P2Clust: K = {}".format(self.k)
