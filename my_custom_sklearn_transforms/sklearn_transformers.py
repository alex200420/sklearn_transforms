from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')
    
# All sklearn Transforms must have the `transform` and `fit` methods
class FillNA(BaseEstimator, TransformerMixin):
    def __init__(self):
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.fillna(0)
    
# All sklearn Transforms must have the `transform` and `fit` methods
class AddColumns(BaseEstimator, TransformerMixin):
    def __init__(self, skewed_cols):
        self.skewed_cols = skewed_cols

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primero copiamos el dataframe de datos de entrada 'X'
        df2 = X.copy()
        #
        df2['ratio_CXC_TOTAL_GASTOS'] = df2['CXC']/df2['TOTAL_GASTOS']
        df2['ratio_GASTOS_UTILIDAD_BRUTA'] = df2['TOTAL_GASTOS']/df2['UTILIDAD_BRUTA']
        df2['ratio_VENTAS_TRANSPORTE'] = df2['TOTAL_VENTAS']/df2['EQ_TRANSPORTE']
        df2['ratio_UTILIDADES_TOTAL'] = df2['UTILIDAD_O_PERDIDA']/df2['UTILIDADES_ACUMULADAS']
        df2['ratio_UTILIDADES_GASTOS'] = df2['UTILIDAD_BRUTA']/df2['TOTAL_GASTOS']
        df2['ratio_CXC_CXP'] = df2['CXC']/df2['CXP']
        df2['ratio_CXC_UTILIDAD'] = df2['UTILIDAD_BRUTA']/df2['CXC']
        df2['ratio_CXC_GASTOS'] = df2['TOTAL_GASTOS']/df2['CXP']
        df2['ratio_CXC_VENTAS'] = df2['TOTAL_VENTAS']/df2['CXP']
        df2['exp_VENTAS_UTILIDAD'] = df2['TOTAL_VENTAS']**(1/df2['UTILIDADES_ACUMULADAS'])
        df2['ratio_CXC_INVENTARIO'] = df2['CXC']/df2['INVENTARIO']
        df2['ratio_CXP_INVENTARIO'] = df2['CXP']/df2['INVENTARIO']
        ##
        for column in df2.columns[~np.isin(df2.columns, ['OBJETIVO'])]:
            if column[-6:] != 'isnull':
                df2[f'{column}_isnull'] = np.int32(df2[column].isnull())
        ##
        for skewed_col in self.skewed_cols:
            df2[f'{skewed_col}_outlier'] = np.int32(abs((df2[skewed_col] - df2[skewed_col].mean())/df2[skewed_col].std())>3)
            df2[skewed_col] = np.log1p(df2[skewed_col])
        # Devolvemos un nuevo dataframe de datos sin las columnas no deseadas
        return df2
