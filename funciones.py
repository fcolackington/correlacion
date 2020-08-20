#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Importar módulos
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


def binarize_hist(data, var):
    # genero temporal del dataframe
    tmp = data
    # creamos nueva variable con condicion de acuerdo a la variable ingresada
    tmp['binarize'] = np.where(tmp[var] > np.mean(tmp[var]), 1, 0)
    # generamos la grilla de trabajo e indicamos tipo de grafico y variable
    grid = sns.FacetGrid(data, col = 'binarize', col_wrap = 2, sharex=False).map(sns.distplot, var, kde = False)
    # accedemos a cada subgrilla creada
    axes = grid.fig.axes
    # generamos un dataframe para cada valor de variable creada
    hist_0 = tmp[tmp['binarize'] == 0][var].dropna()
    hist_1 = tmp[tmp['binarize'] == 1][var].dropna()
    # graficamos de acuerdo a cada subgrilla accediendo por axes[valor]
    axes[0].axvline(np.mean(hist_0), color = 'red')
    axes[1].axvline(np.mean(hist_1), color = 'red')


# Definir función grouped_boxplot usando seaborn
def group_boxplot(data, var, grupo):
    tmp = data
    plt.figure(figsize=(10,5))
    sns.boxplot(x=grupo, y=var,
                data=tmp, order=tmp.groupby(grupo)[var].mean().sort_values(ascending=False).index)
    plt.show()


# Definir función grouped_scatterplot sando seaborn
def grouped_scatterplot(data, x, y, grupo):
    tmp = data
    grid = sns.FacetGrid(data, col = grupo, col_wrap = 3).map(sns.scatterplot, x, y)


# genero una funcion para eliminar ruido en heatmap
def high_correlation(data, valor):
    '''
    Esta funcion entrega un DataFrame con solo aquellos valores de correlacion mayores al valor
    ingresado como parametro

    Input:
        - data: DataFrame a evaluar
        - valor: valor numerico entre 0 y 1

    Out:
        - DataFrame con valores de correlacion mayores al valor ingresado
    '''

    tmp = data
    df_high = pd.DataFrame(np.where((tmp.corr() > valor) | (tmp.corr() < -valor), tmp.corr(), np.nan))
    df_high.columns = tmp.corr().columns
    df_high.index = tmp.corr().columns
    return df_high
