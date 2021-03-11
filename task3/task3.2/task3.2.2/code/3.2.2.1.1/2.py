import pandas as pd
import numpy as np
df1 = pd.read_csv('./train.csv', dtype=str)
df2 = pd.read_csv('./santander-product-recommendation/test_ver2.csv/test_ver2.csv', dtype=str)
#处理ind_nomina_ult1
df1.drop(df1[df1.ind_nomina_ult1.isnull()].index.tolist(), axis=0, inplace=True)
#合并df1,df2
file = pd.concat([df1, df2])
file = file.reset_index(drop=True)
#处理age --年龄
file.loc[file.age == ' NA', 'age'] = 'NaN'
file.age = file.age.astype(float)
file.loc[file.age < 18, 'age']  = file.loc[(file.age >= 18) & (file.age <= 30), 'age'].mean(skipna=True)
file.loc[file.age > 100, 'age'] = file.loc[(file.age >= 30) & (file.age <= 100), 'age'].mean(skipna=True)
file.age.fillna(file.age.mean(), inplace=True)
#处理ind_nuevo --客户新/旧
file.ind_nuevo = file.ind_nuevo.astype(float)
file.loc[file['ind_nuevo'].isnull(), 'ind_nuevo'] = 1
#处理antiguedad --客户工龄
file.antiguedad = pd.to_numeric(file.antiguedad, errors='coerce')
file.loc[file.antiguedad.isnull(), 'antiguedad'] = file.antiguedad.min()
file.loc[file.antiguedad < 0, 'antiguedad'] = 0
#处理indrel
file.loc[file.indrel.isnull(), 'indrel'] = 1
file.indrel = file.indrel.astype(float)
#处理nomprov --省名称
file.loc[file.nomprov.isnull(), 'nomprov'] = 'UNKNOWN'
file.loc[file.nomprov == 0, 'nomprov'] = 'UNKNOWN'
code = file.nomprov.unique()
for i in range(code.size):
    file.loc[file.nomprov==code[i], 'nomprov'] = i+1
file.nomprov = file.nomprov.astype(float)
#处理无用特征
file.drop(['fecha_dato', 'ncodpers', 'tipodom', 'cod_prov', 'pais_residencia','ult_fec_cli_1t'], axis=1, inplace=True)
#处理renta --家庭总收入
file.loc[file.renta=='         NA', 'renta'] = 'NaN'
file.loc[file.renta=='NA', 'renta'] = 'NaN'
file.renta     = file.renta.astype(float)
grouped        = file.groupby('nomprov').agg({'renta':lambda x: x.median(skipna=True)}).reset_index()
new_incomes    = pd.merge(file, grouped, how='inner', on='nomprov').loc[:, ['nomprov', 'renta_y']]
new_incomes    = new_incomes.rename(columns={'renta_y':'renta'}).sort_values('renta').sort_values('nomprov')
file.sort_values('nomprov', inplace=True)
file           = file.reset_index()
new_incomes    = new_incomes.reset_index()
file.loc[file.renta.isnull(), 'renta'] = new_incomes.loc[file.renta.isnull(), 'renta'].reset_index()
file.loc[file.renta.isnull(), 'renta'] = file.loc[file.renta.notnull(), 'renta'].median()
file = file.sort_values('index')
file = file.reset_index(drop=True)
file.drop(['index'], axis=1, inplace=True)
#处理ind_empleado
file.loc[file.ind_empleado == 0, 'ind_empleado'] = file.ind_empleado.mode()[0]
file.loc[file.ind_empleado.isnull(), 'ind_empleado'] = file.ind_empleado.mode()[0]
code = file.ind_empleado.unique()
for i in range(code.size):
    file.loc[file.ind_empleado==code[i], 'ind_empleado'] = i+1
file.ind_empleado = file.ind_empleado.astype(float)
#处理segmento
file.loc[file.segmento.isnull(), 'segmento'] = file.segmento.mode()[0]
code = file.segmento.unique()
for i in range(code.size):
    file.loc[file.segmento==code[i], 'segmento'] = i+1
file.segmento = file.segmento.astype(float)
#处理sexo
file.loc[file.sexo.isnull(), 'sexo'] = file.sexo.mode()[0]
code = file.sexo.unique()
for i in range(code.size):
    file.loc[file.sexo==code[i], 'sexo'] = i+1
file.sexo = file.sexo.astype(float)
#处理fecha_alta
file.loc[file.fecha_alta.isnull(), 'fecha_alta'] = '2014'
file.fecha_alta = file.fecha_alta.apply(lambda x:x[:4])
file.fecha_alta = file.fecha_alta.astype(float)
file.fecha_alta = 2016 - file.fecha_alta
#处理indrel
file.loc[file.indrel.isnull(), 'indrel'] = 1
file.indrel = file.indrel.astype(float)
#处理indrel_1mes
file.loc[file.indrel_1mes.isnull(), 'indrel_1mes'] = file.indrel_1mes.mode()[0]
file.indrel_1mes = file.indrel_1mes.astype(float)
#处理tiprel_1mes
file.loc[file.tiprel_1mes.isnull(), 'tiprel_1mes'] = file.tiprel_1mes.mode()[0]
code = file.tiprel_1mes.unique()
for i in range(code.size):
    file.loc[file.tiprel_1mes==code[i], 'tiprel_1mes'] = i+1
file.tiprel_1mes = file.tiprel_1mes.astype(float)
#处理indresi
file.loc[file.indresi.isnull(), 'indresi'] = file.indresi.mode()[0]
code = file.indresi.unique()
for i in range(code.size):
    file.loc[file.indresi==code[i], 'indresi'] = i+1
file.indresi = file.indresi.astype(float)
#处理indext
file.loc[file.indext.isnull(), 'indext'] = file.indext.mode()[0]
code = file.indext.unique()
for i in range(code.size):
    file.loc[file.indext==code[i], 'indext'] = i+1
file.indext = file.indext.astype(float)
#处理conyuemp
file.loc[file.conyuemp.isnull(), 'conyuemp'] = 3.0
file.loc[file.conyuemp=='N', 'conyuemp'] = 1.0
file.loc[file.conyuemp=='S', 'conyuemp'] = 2.0
file.conyuemp = file.conyuemp.astype(float)
#处理canal_entrada 
file.loc[file.canal_entrada.isnull(), 'canal_entrada'] = file.canal_entrada.mode()[0]
code = file.canal_entrada.unique()
for i in range(code.size):
    file.loc[file.canal_entrada==code[i], 'canal_entrada'] = i+1
file.canal_entrada = file.canal_entrada.astype(float)
#处理indfall
file.loc[file.indfall.isnull(), 'indfall'] = 3.0
file.loc[file.indfall=='N', 'indfall'] = 1.0
file.loc[file.indfall=='S', 'indfall'] = 2.0
file.indfall = file.indfall.astype(float)
#处理ind_actividad_cliente
file.loc[file.ind_actividad_cliente.isnull(), 'ind_actividad_cliente'] = 3.0
file.loc[file.ind_actividad_cliente=='1.0', 'ind_actividad_cliente'] = 2.0
file.loc[file.ind_actividad_cliente=='0.0', 'ind_actividad_cliente'] = 1.0
file.ind_actividad_cliente = file.ind_actividad_cliente.astype(float)
#归一化
mean = file.iloc[:,0:18].mean()
std = file.iloc[:,0:18].std()
file.iloc[:,0:18] = (file.iloc[:,0:18]-mean)/std
#分割训练与测试集
df1 = file.iloc[:630284,:]
df2 = file.iloc[630284:1559900,:]
df1.to_csv('./pre_train.csv')
df2.to_csv('./pre_test.csv')