import numpy as np
import seaborn as sns
import random
from itertools import combinations
import copkmeans
import p2clust
import pckmeans
from sklearn import metrics
import pandas as pd
from sklearn.metrics import adjusted_rand_score
import PCSKMeans
from sklearn.cluster import KMeans

sns.set()


class Experiments:
    models = False
    k = False
    x = False
    y = False
    ml = False
    cl = False
    rep = False
    results = False

    def __init__(self, models: list = 0, x: np.ndarray = 0, y: np.ndarray = 0, centers: list = 0, sigmas: list = 0,
                 numb_data: int = 0,
                 ml: list = 0, cl: list = 0, nRest: int = 20, k: int = 0, rep: int = 1):
        
        # Generamos los datos
        if x.any() and y.any():
            self.x = x
            self.y = y
        elif centers and sigmas and numb_data:
            self.generate_data_2D(centers, sigmas, numb_data)
        else:
            raise Exception("Error! No data was found.")

        # Generamos las restricciones
        if isinstance(ml, list) and isinstance(cl, list):
            self.ml = ml
            self.cl = cl
        else:
            self.generateConstrains(nRest)

        if models:
            self.models = models
        else:
            # Generamos el numero de vecinos
            if k:
                self.k = k
            else:
                self.k = len(np.unique(self.y))

            self.models = [copkmeans.COPKmeans(k=self.k),
                           #copkmeans.COPKmeans(k=self.k, monotonic=True),
                           p2clust.P2Clust(k=self.k),
                           #pckmeans.PCKmeans(k=self.k),
                           pckmeans.PCKmeans(k=self.k, monotonic=True)]

        # Gestionamos el numero de repeticiones:
        self.rep = rep
       
        self.experiments()

    def generate_data_2D(self, centers: list, sigmas: list, numb_data: int):
        """
        Función creada por Germán Rodriguez Almagro para crear conjuntos de datos
        en 2d etiquetados para Clustering.

        :param centers: Localización del centro de los clusters (lista de listas)
        :param sigmas: Dispersion del centro (lista de listas)
        :param numb_data: Número de elementos a crear para cada cluster

        :return: crea los datos X e Y
        """
        np.random.seed(7)
        xpts = np.zeros(1)
        ypts = np.zeros(1)
        labels = np.zeros(1)
        for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
            xpts = np.hstack((xpts, np.random.standard_normal(numb_data) * xsigma + xmu))
            ypts = np.hstack((ypts, np.random.standard_normal(numb_data) * ysigma + ymu))
            labels = np.hstack((labels, np.ones(numb_data) * i))

        X = np.zeros((len(xpts) - 1, 2))
        X[:, 0] = xpts[1:]
        X[:, 1] = ypts[1:]

        y = labels[1:]

        self.x, self.y = X, y

    def generateConstrains(self, n: int):
        """
        Generamos n(n-1)/2 restricciones aleatorias de X.

        :param self.x: Conjunto de datos
        :param self.y: Etiquetas de los datos
        :param n: subconjunto de datos sobre el que hacer las restricciones

        :return: MustLink y CannotLink
        """
        # Opcion full aleatorio no estratificado DA PROBLEMAS:
        #pairs = set()
        #tries = 0
        #while len(pairs) < (n*(n-1)/2):
        #    pairs.add(tuple(sorted(random.sample([i for i in range(len(self.x))], 2))))
        #    tries += 1

        # Opcion aleatoria estratificado
        from itertools import combinations
        # Seleccionamos aleatoriamente n elementos de x:
        subset = random.sample([i for i in range(len(self.x))], n)

        # Creamos las combinaciones posibles de 2 elementos de nuestro subset:
        pairs = list(combinations(subset, 2))

        # Iteramos sobre cada combinacion asignandola a ml o cl
        ml = []
        cl = []
        for pair in pairs:
            if self.y[pair[0]] == self.y[pair[1]]:
                ml.append(pair)
            else:
                cl.append(pair)

        self.ml, self.cl = ml, cl

    def unsat(self, cluster: list, ml: list, cl: list) -> float:
        """
        Porcentaje de restricciones violadas.

        :param cluster: Asignacion de cluster a cada instancia.
        :param ml: lista de Must Link constrains.
        :param cl: lista de Cannot Link constrains.
        :return: porcentaje de restricciones violadas.
        """
        # Calculamos el número de restricciones ml violadas
        ml_ = [1 if cluster[i] != cluster[j] else 0 for i, j in ml]
        # Calculamos el número de restricciones cl violadas
        cl_ = [1 if cluster[i] == cluster[j] else 0 for i, j in cl]
        # Devolvermos el porcentaje
        return sum(ml_ + cl_) / len(ml + cl)

    # Non-Monotonic Index

    def NMI1(self, x: list, clusters: list) -> float:
        """
        Non-Monocity Index (https://sci2s.ugr.es/sites/default/files/1-s2.0-S0169023X16303585-main.pdf)

        Numero de instancias no-montónicas dividido entre el total de instancias

        :param x: Conjunto de datos.
        :param cluster: Asignacion de cluster a cada instancia.
        :return: nuemro de instancias no-montónicas dividido entre el total de instancias
        """
        nmi = 0
        for i in range(len(x)):
            for j in range(i, len(x)):
                if not self.isMonotonic(i, j, x, clusters):
                    nmi += 2

        return nmi / (len(x) * len(x) - len(x))

    def NMI2(self, x: list, clusters: list) -> float:
        """
        Non-Monocity Index (https://sci2s.ugr.es/sites/default/files/1-s2.0-S0169023X16303585-main.pdf)

        Numero de instancias no-montónicas dividido entre el total de instancias

        :param x: Conjunto de datos.
        :param cluster: Asignacion de cluster a cada instancia.
        :return: nuemro de instancias no-montónicas dividido entre el total de instancias
        """
        nmi = 0
        mon = [True] * len(x)

        for i in range(len(x)):
            j = 0
            while mon[i] and j < len(x):
                if not self.isMonotonic(i, j, x, clusters):
                    nmi += 1.0
                    mon[i] = False
                    if mon[j]:
                        nmi += 1.0
                        mon[j] = False
                j += 1
        return nmi / len(x)

    def isMonotonic(self, a: int, b: int, x: list, clusters: list) -> bool:
        """
        Compruba si dos instancias (a y b) se pueden comparar
        y en el caso de que lo sea devuelve si la asignación
        de dichas instancias ha sido monotónica (no ha violado
        la monotonía).
        :param a: Instancia a
        :param b: Instancia b
        :param x: Conjunto de datos.
        :param cluster: Asignacion de cluster a cada instancia.
        :return: Booleano indicando si se viola o no las restricciones de monotonia
        """

        state = self.comparableState(a, b, x)
        if state == '=':
            return clusters[a] == clusters[b]
        elif state == '<':
            return clusters[a] <= clusters[b]
        elif state == '>':
            return clusters[a] >= clusters[b]
        elif state == '<>':
            return True

    def comparableState(self, a: int, b: int, x: list) -> str:
        """
        Funcion que comprueba si dos elementos son comparables y cual es
        su condicion de comparabilidad (<, >, =, <>)
        :param a: Instancia a
        :param b: Instancia b
        :param x: Conjunto de datos.
        :return: string que indica la relacion que hay entre las instancias a y b
        """
        i = 0
        state = '='

        while i < len(x[a]):
            if state == '=':
                if x[a][i] < x[b][i]:
                    state = '<'
                elif x[a][i] > x[b][i]:
                    state = '>'
                else:
                    state = '='

            elif state == "<":
                if x[a][i] < x[b][i]:
                    state = '<'
                else:
                    state = '<>'

            elif state == ">":
                if x[a][i] > x[b][i]:
                    state = '>'
                else:
                    state = '<>'
            i += 1
        return state

    #make a function that convert the constraint lists to a constraint numpy matrix
    def make_constraint_matrix(self, ml: list, cl: list) -> np.ndarray:
        """
        Funcion que crea una matriz de restricciones de tamaño n x n
        donde n es el numero de instancias.
        :param ml: lista de Must Link constrains.
        :param cl: lista de Cannot Link constrains.
        :return: matriz de restricciones
        """
        # Creamos una matriz de restricciones de tamaño n x n
        constraint_matrix = np.zeros((len(self.x), len(self.x)))
        # Iteramos sobre cada restriccion
        for i, j in ml:
            constraint_matrix[i][j] = 1
            constraint_matrix[j][i] = 1
        for i, j in cl:
            constraint_matrix[i][j] = 1
            constraint_matrix[j][i] = 1
        return constraint_matrix

    def experiments(self):
        results = []
        for model in self.models:
            model.fit(x=self.x, ml=self.ml, cl=self.cl)
            print(str(model) + ' [x]  -->  DONE')
            result_model = [str(model), metrics.normalized_mutual_info_score(self.y, model.clusters),
                            adjusted_rand_score(self.y, model.clusters), self.unsat(model.clusters, self.ml, self.cl),
                            self.NMI1(self.x, model.clusters), self.NMI2(self.x, model.clusters)]
            results.append(result_model)
            #save clusters to csv
            print("Saving clusters to csv from model " + str(model) + " ...")
            np.savetxt("partitions/" + str(model) + ".csv", model.clusters, delimiter=",")

        #run the PCSKmeans algorithm over the dataset with constraints passed as matrix and generate the model results
        pcsk_means_results = PCSKMeans.PCSKMeans(x = self.x, k = self.k, const_mat = self.make_constraint_matrix(self.ml, self.cl))
        pcsk_means_results = pcsk_means_results[0]
        print('PCSKMeans [x]  -->  DONE')
        result_model = ["PCSKMeans", metrics.normalized_mutual_info_score(self.y, pcsk_means_results),
                            adjusted_rand_score(self.y, pcsk_means_results), self.unsat(pcsk_means_results, self.ml, self.cl),
                            self.NMI1(self.x, pcsk_means_results), self.NMI2(self.x, pcsk_means_results)]
        results.append(result_model)
        #save clusters to csv
        print("Saving clusters to csv from model PCSKMeans ...")
        np.savetxt("partitions/PCSKMeans.csv", pcsk_means_results, delimiter=",")

        k_means_results = KMeans(n_clusters=self.k, init='k-means++')
        k_means_results = k_means_results.fit(self.x).labels_
        print('KMeans [x]  -->  DONE')
        result_model = ["KMeans", metrics.normalized_mutual_info_score(self.y, k_means_results),
                            adjusted_rand_score(self.y, k_means_results), self.unsat(k_means_results, self.ml, self.cl),
                            self.NMI1(self.x, k_means_results), self.NMI2(self.x, k_means_results)]
        results.append(result_model)
        #save clusters to csv
        print("Saving clusters to csv from model KMeans ...")
        np.savetxt("partitions/KMeans.csv", k_means_results, delimiter=",")


        self.results = results
        self.pd_results = pd.DataFrame(results, columns=["name", "Mutual_info", "adjusted_rand_score", "unsat", "NMI-1",
                                                         "NMI-2"])


if __name__ == '__main__':
    prueba = Experiments(centers=[[1, 1], [5, 5], [10, 10]], sigmas=[[5, 5], [5, 5], [5, 5]], numb_data=25)

    print(pd.DataFrame(prueba.results, columns=["name", "Mutual_info", "adjusted_rand_score", "unsat"]))

