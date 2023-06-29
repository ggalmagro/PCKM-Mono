import random
import copkmeans
import p2clust
import pckmeans
import time
import os
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from sklearn import metrics


class Experiments:

    def __init__(self, directory: str, k: int = 0, rep: int = 30, propCons: list = [10, 15, 20]):
        """
        Constructor, lanzamos los experimentos

        :param directory: Direcorio donde estan las carpetas con los datos en forma x.csv, y.csv
        :param k: Numero de clusters, por defecto se calcula como el numero de valores unicos en y
        :param rep: Numero de repeticiones de cada experimento
        :param propCons: Porcentaje de restricciones en forma de lista.
        """

        # Almacenamos la lista de proporciones de restricciones:
        self.propCons = [i/100 for i in propCons]

        # Gestionamos el numero de repeticiones:
        self.rep = rep

        # Lanzamos los experimentos:
        for carpeta in os.listdir(directory):
            ruta_1 = os.path.join(directory, carpeta)
            print(ruta_1)
            if os.path.isdir(ruta_1):
                print('--------------------------------')
                print('PATH: ', ruta_1)
                for archivo in os.listdir(ruta_1):
                    ruta_2 = os.path.join(ruta_1, archivo)
                    if 'x.csv' in str(archivo):
                        self.x = pd.read_csv(ruta_2, index_col=0).to_numpy()
                    elif 'y.csv' in str(archivo):
                        self.y = np.concatenate(pd.read_csv(ruta_2, index_col=0).values)

                if k:
                    self.k = k
                else:
                    self.k = len(np.unique(self.y))
                # Instanciamos los modelos
                self.models = [copkmeans.COPKmeans(k=self.k),
                               copkmeans.COPKmeans(k=self.k, monotonic=True),
                               p2clust.P2Clust(k=self.k),
                               pckmeans.PCKmeans(k=self.k),
                               pckmeans.PCKmeans(k=self.k, monotonic=True)]
                # Una vez establecidos x e y llevamos a cabo los experimentos
                for numConst in self.propCons:
                    print()
                    print('Proporcion de restricciones =', numConst)
                    print()
                    self.ml, self.cl = self.generateConstrains(int(len(self.y) * numConst))

                    pd.DataFrame({'ml': self.ml}).to_csv(ruta_1 + '/restrictions/ML_{}.csv'.format(numConst))
                    pd.DataFrame({'cl': self.cl}).to_csv(ruta_1 + '/restrictions/CL_{}.csv'.format(numConst))
                    for model in self.models:
                        results = dict()
                        metric = dict()
                        func = partial(self.experiments, model, results, metric)
                        pool = multiprocessing.Pool(7)
                        resp = pool.map(func, [i for i in range(self.rep)])
                        results = {k: v for d in [r[0] for r in resp] for k, v in d.items()}
                        metric = {k: v for d in [r[1] for r in resp] for k, v in d.items()}
                        # Cerramos los procesos en paralelo
                        pool.close()
                        pool.join()
                        print("process", os.getpid(), "model", model, "done")
                        self.save_results(ruta_1, results, metric, model, numConst)

    def generate_data_2D(self, centers: list, sigmas: list, numb_data: int):
        """
        Función creada por Germán González Almagro para crear conjuntos de datos
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

    def generateConstrains(self, n: int) -> tuple:
        """
        Generamos n(n-1)/2 restricciones aleatorias de X.

        :param self.x: Conjunto de datos
        :param self.y: Etiquetas de los datos
        :param n: subconjunto de datos sobre el que hacer las restricciones

        :return: MustLink y CannotLink
        """
        # Opcion full aleatorio no estratificado DA PROBLEMAS:
        pairs = set()
        tries = 0
        while len(pairs) < (n*(n-1)/2):
            pairs.add(tuple(sorted(random.sample([i for i in range(len(self.x))], 2))))
            tries += 1

        # Opcion aleatoria estratificado
        #from itertools import combinations
        # Seleccionamos aleatoriamente n elementos de x:
        #subset = random.sample([i for i in range(len(self.x))], n)

        # Creamos las combinaciones posibles de 2 elementos de nuestro subset:
        #pairs = list(combinations(subset, 2))

        # Iteramos sobre cada combinacion asignandola a ml o cl
        ml = []
        cl = []
        for pair in pairs:
            if self.y[pair[0]] == self.y[pair[1]]:
                ml.append(pair)
            else:
                cl.append(pair)

        #self.ml, self.cl = ml, cl
        return ml, cl

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

    def NMI1(self, x: np.ndarray, clusters: list) -> float:
        """
        Non-Monocity Index (https://sci2s.ugr.es/sites/default/files/1-s2.0-S0169023X16303585-main.pdf)

        Numero de instancias no-montónicas dividido entre el total de instancias

        :param x: Conjunto de datos.
        :param clusters: Asignacion de cluster a cada instancia.
        :return: nuemro de instancias no-montónicas dividido entre el total de instancias
        """
        nmi = 0
        for i in range(len(x)):
            for j in range(i, len(x)):
                if not self.isMonotonic(i, j, x, clusters):
                    nmi += 2

        return nmi / (len(x) * len(x) - len(x))

    def NMI2(self, x: np.ndarray, clusters: list) -> float:
        """
        Non-Monocity Index (https://sci2s.ugr.es/sites/default/files/1-s2.0-S0169023X16303585-main.pdf)

        Numero de instancias no-montónicas dividido entre el total de instancias

        :param x: Conjunto de datos.
        :param clusters: Asignacion de cluster a cada instancia.
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

    def isMonotonic(self, a: int, b: int, x: np.ndarray, clusters: list) -> bool:
        """
        Compruba si dos instancias (a y b) se pueden comparar
        y en el caso de que lo sea devuelve si la asignación
        de dichas instancias ha sido monotónica (no ha violado
        la monotonía).
        :param a: Instancia a
        :param b: Instancia b
        :param x: Conjunto de datos.
        :param clusters: Asignacion de cluster a cada instancia.
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

    def comparableState(self, a: int, b: int, x: np.ndarray) -> str:
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

    def save_results(self, path: str, results: dict, metric: dict, model: object, numConst: float):
        """
        Transformamos en dataframes de pandas los diccionarios y guardamos los valores como csv

        :param path: Path donde guardar los CSV
        :param results: Resultados obtenidos (etiquetas)
        :param measures: Medidas calculadas
        :param model: Nombre del Modelo
        :param numConst: Porcentaje de restricciones
        """
        results_df = pd.DataFrame.from_dict(results)
        metric_df = pd.DataFrame.from_dict(metric)
        name = str(model).replace(' ', '').replace(':', '_')+'_Rest_{}.csv'.format(numConst)
        results_df.to_csv(path+'/results/'+name)
        metric_df.to_csv(path+'/results/metrics/'+name)


    def experiments(self, model: object, results: dict, metric: dict, i: int):
        """
        Funcion que ejecuta los experimentos. se ha paralelizado.

        :param path: Path en el que se encuentran los datos
        :param model: Modelo con el que se hacen los experimentos
        """

        start = time.process_time()
        model.fit(x=self.x, ml=self.ml, cl=self.cl)
        end = time.process_time()

        labels = model.clusters[:]
        labels.append(end-start)
        results[i] = labels
        if -1 in model.clusters[:]:
            metric[i] = [str(model),
                         -1,
                         -1,
                         -1,
                         -1,
                         -1]
        else:
            metric[i] = [str(model),
                         metrics.normalized_mutual_info_score(self.y, model.clusters),
                         metrics.adjusted_rand_score(self.y, model.clusters),
                         self.unsat(model.clusters, self.ml, self.cl),
                         self.NMI1(self.x, model.clusters),
                         self.NMI2(self.x, model.clusters)]

        return results, metric



if __name__ == '__main__':
    p1 = Experiments('/home/pablo-snz/tfm/Code/src/data')
    print('[x]  -->  Done!')

