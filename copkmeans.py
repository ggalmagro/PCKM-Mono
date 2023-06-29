from base.kmeans import KMEANS
from scipy.spatial import distance
import numpy as np
import random
import warnings


class COPKmeans(KMEANS):
    distOrder = False  # valor p de la formula de la distancia de Minkowsky

    def __init__(self, distOrder: int = 2,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distOrder = distOrder

    def initialize_prototypes(self) -> list:
        """
        Se inicializan los prototipos de forma aleatoria.

        :return: Prototipos
        """
        rs = np.random.RandomState()
        return self.data[rs.permutation(self.data.shape[0])[:self.k]]

    def assign_clusters(self, prototypes: list, cluster: list) -> list:
        """
        Funcion para asignar a cada instancia un cluster, se asigna el cluster que minimize la funcion objetivo

        :param prototypes: el valor actual de los centroides
        :param cluster: lista de etiquetas de asignación a cada cluster
        :return: lista de etiquetas de asignación a cada cluster actualizada
        """
        # Definimos un numero maximo de intentos
        for j in range(self.maxTry):
            try:
                r = list(range(self.data.shape[0]))
                random.shuffle(r)
                # Asignacion aleatoria de los datos
                for i in r:
                    x = self.data[i]
                    # Computamos la distancia
                    if self.monotonic:
                        # Distancia monotonica
                        distances = [np.subtract(x, prototypes[k]).sum() for k in [i for i in range(self.k)]]
                    else:
                        # Distancia minkowski
                        distances = [distance.minkowski(x, prototypes[k], p=self.distOrder) for k in
                                     [i for i in range(self.k)]]
                    # Ordenamos el id de los clusters en base a su distancia
                    bestClust = np.argsort(np.absolute(distances))
                    # Iteramos sobre cada id
                    for value in bestClust:
                        # SI no se viola ninguna restriccion se incluye.
                        if not self.violate_constrains(i, value, cluster):
                            cluster[i] = value
                            break

                    # Si ningun cluster es asignado porque todos violan alguna restriccion:
                    if cluster[i] == -1:
                        #warnings.warn('COPKmeans: [W] Instancia no ha podido ser asignada, reiniciando asignacion ...')
                        # Lanzamos una excepcion y volvemos a incializar los clusters de forma aleatoria.
                        raise Exception()

                #self.plot_me(cluster, prototypes)

                # Comprobacion clusters vacios.
                samplesInCLuster = np.bincount(cluster, minlength=self.k)
                emptyCluster = np.where(samplesInCLuster == 0)[0]

                if len(emptyCluster) > 0:
                    #warnings.warn('COPKmeans: [W] Cluster sin instancias, reiniciando asignacion ...')
                    raise Exception()

                return cluster
            except:
                cluster = np.array([-1 for i in range(len(self.data))])
                prototypes = self.initialize_prototypes()
        return 0


    def violate_constrains(self, id: int, value: int, cluster: list) -> bool:
        """
        Funcion que itera sobre las restricciones MustLink y Cannot Link para una determinada
        instancia

        :param id: int -> ID de la instancia sobre la que vamos a estudiar las restricciones
        :param value: int -> valor del Cluster al que queremos asignar nuestro elemento
        :param cluster: list -> lista con los valores del resto de los clusters

        :return: bool -> Indica si se ha violado alguna restriccion o no
        """
        # Por cada restriccion MustLink sobre la instancia con este id
        for element in self.mustLink[id]:
            # Si las instancias del mustlink[id] estan en otro cluster:
            if cluster[element] >= 0 and cluster[element] != value:
                # Return violate constrin = TRUE
                return True
        # Por cada restriccion CannotLink sobre la instancia con este id
        for element in self.cannotLink[id]:
            # Si las instancias del cannotLink[id] estan en el mismo cluster:
            if cluster[element] == value and cluster[element] >= 0:
                # Return violate constrin = TRUE
                return True

        return False

    def __str__(self):
        if self.monotonic:
            return "COP-MONO: K = {}".format(self.k)
        else:
            return "COP: K = {}".format(self.k)
