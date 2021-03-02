import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.models import Sequential
import torch
from nni.feature_engineering.feature_selector import FeatureSelector
def main():
    train_file_nm="../data/train.csv" #训练集
    test_file_nm="../data/test.csv"   #测试集
    train_df= pd.read_csv(train_file_nm)
    test_df= pd.read_csv(test_file_nm)
    '''<class 'pandas.core.frame.DataFrame'>
    RangeIndex: 878049 entries, 0 to 878048
    Data columns (total 9 columns):
    #   Column      Non-Null Count   Dtype
    ---  ------      --------------   -----
    0   Dates       878049 non-null  object
    1   Category    878049 non-null  object
    2   Descript    878049 non-null  object
    3   DayOfWeek   878049 non-null  object
    4   PdDistrict  878049 non-null  object
    5   Resolution  878049 non-null  object
    6   Address     878049 non-null  object
    7   X           878049 non-null  float64
    8   Y           878049 non-null  float64
    dtypes: float64(2), object(7)
    memory usage: 60.3+ MB'''
    '''<class 'pandas.core.frame.DataFrame'>
    RangeIndex: 884262 entries, 0 to 884261
    Data columns (total 7 columns):
    #   Column      Non-Null Count   Dtype
    ---  ------      --------------   -----
    0   Id          884262 non-null  int64
    1   Dates       884262 non-null  object
    2   DayOfWeek   884262 non-null  object
    3   PdDistrict  884262 non-null  object
    4   Address     884262 non-null  object
    5   X           884262 non-null  float64
    6   Y           884262 non-null  float64
    dtypes: float64(2), int64(1), object(4)
    memory usage: 47.2+ MB'''
    '''让我们对旧金山犯罪分类数据进行说明：一共34243行
        - id :【测试集独有】
        - Dates :日期戳
        - Category : 罪名种类 一共 39 种【目标预测】【训练集独有】
        - Descript : 罪行描述 一共 879种 【训练集独有】【去除】
        - DayOfWeek: 犯罪日期(星期X) 显而易见7种
        - PdDistrict: 警察局区名称 一共10种
        - Resolution:处理方法 一共17种 【训练集独有】【去除】
        - Address: 犯罪大致街区地址？？一共23228种【去除】
        - X: 经度
        - Y: 纬度
        '''
    #记录一下新思路：加入特征值的处理：训练一个可以预测处理方法的玩意，把Resolution的值补充上
    print("------------------------------")
    PdDistrict_ls=["BAYVIEW","CENTRAL","INGLESIDE","MISSION","NORTHERN", 
    "PARK", "RICHMOND", "SOUTHERN","TARAVAL","TENDERLOIN"]
    #Year_ls=["2003","2004","2005","2006","2007","2008","2009","2010","2011","2012","2013","2014","2015"]
    x,y,test,useless_data=prepare()

    #print(x.info())
    x_dict=["Month","Day","Hours","Minute","DayOfWeek"
    ,"X","Y"]+PdDistrict_ls
    x_c= pd.DataFrame()
    for s in x_dict:
        x_c[s]= x[s]
    x=x_c
    
    print(x.info())
    print("------------------------------")
    #非常显然，某些列的数据我们处理起来非常麻烦...需要有所舍弃
    
    #在经过处理之后
    # x是: Year,Month,Day,Hours,DayOfWeek,PdDistrict,X,Y
    # y是: Category
    #x.drop(del_list,inplace=True,axis=1) 
    
    x["X"] /= 10
    x["Y"] /= 10
    x["Month"] /= 1.2
    x["Day"] /= 3
    x["Hours"] /= 2.4
    x["Minute"] /= 6
    train(x,y,test)
def normize(x):
    max_x= x.max()
    min_x= x.min()
    return (x-min_x)/(max_x-min_x)
def prepare(): #进行数据的 导入 和 预处理  【应该是特征工程进行的地方】
    train_file_nm="../data/train.csv" #训练集
    test_file_nm="../data/test.csv"   #测试集
    train_df= pd.read_csv(train_file_nm)
    test_df= pd.read_csv(test_file_nm)
    date_dict= ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    PdDistrict_dict=["BAYVIEW","CENTRAL","INGLESIDE","MISSION","NORTHERN", 
    "PARK", "RICHMOND", "SOUTHERN","TARAVAL","TENDERLOIN"]
    #数据清除 训练集去除Descript、Resolution、Address；测试集去除Address
    useless_data= train_df.drop(['Descript','Resolution','Address'],inplace=True,axis=1)
    test_df.drop(['Address'],inplace=True,axis=1)
    #对dates列数据进行初始化:0~6
    train_df['DayOfWeek'].replace(to_replace =date_dict , value = [i for i in range(7)],inplace=True)
    test_df['DayOfWeek'].replace(to_replace = date_dict, value = [i for i in range(7)],inplace=True)
    #对PdDistrict列数据进行onehot处理且接合/数值的初始化
    #train_df['PdDistrict'].replace(to_replace =PdDistrict_dict , value = [i for i in range(10)],inplace=True)
    #test_df['PdDistrict'].replace(to_replace =PdDistrict_dict , value = [i for i in range(10)],inplace=True)
    train_dummy= pd.get_dummies(train_df["PdDistrict"])
    train_df.drop('PdDistrict',inplace =True, axis=1)
    '''BAYVIEW  CENTRAL  INGLESIDE  MISSION  NO
    RTHERN  PARK  RICHMOND  SOUTHERN  TARAVAL  TENDERLOIN'''
    train_df = pd.concat([train_df,train_dummy], axis=1)
    test_dummy= pd.get_dummies(test_df["PdDistrict"])
    test_df.drop('PdDistrict',inplace =True, axis=1)
    test_df = pd.concat([test_df,test_dummy], axis=1)

    #对date进行分离:年-月-日-时-分
    data_list=list(train_df["Dates"])
    year_list=[]
    month_list=[]
    day_list=[]
    hour_list=[]
    minute_list=[]
    for data in data_list:
        #由于date数据是：year-month-day hour:minute,所以如下分割
        sp=data.split("-")
        year_list.append(int(sp[0]))
        month_list.append(int(sp[1]))
        sp=sp[2].split(" ")  
        day_list.append(int(sp[0]))
        sp=sp[1].split(":")
        hour_list.append(int(sp[0]))
        minute_list.append(int(sp[1]))
    train_df["Year"]=year_list
    train_df["Month"]=month_list
    train_df["Day"]=day_list
    train_df["Hours"]=hour_list
    train_df["Minute"]=minute_list
    data_list=list(test_df["Dates"])
    year_list=[]
    month_list=[]
    day_list=[]
    hour_list=[]
    minute_list=[]
    for data in data_list:
        sp=data.split("-")
        year_list.append(int(sp[0]))
        month_list.append(int(sp[1]))
        sp=sp[2].split(" ")  
        day_list.append(int(sp[0]))
        sp=sp[1].split(":")
        hour_list.append(int(sp[0]))
        minute_list.append(int(sp[1]))
    test_df["Year"]=year_list
    test_df["Month"]=month_list
    test_df["Day"]=day_list
    test_df["Hours"]=hour_list
    test_df["Minute"]=minute_list

    train_df.drop('Dates',axis=1,inplace=True)
    test_df.drop('Dates',axis=1,inplace=True)  #相当于把date的信息详细化了，所以dates列没存在的必要了
    #标准化
    #train_df["X"]=normize(train_df["X"])
    #train_df["Y"]=normize(train_df["Y"])
    #test_df["X"]=normize(test_df["X"])
    #test_df["Y"]=normize(test_df["Y"])
    #现在回想起我们的目的——预测Category，所以：X是其他数据 而Y是Category

    #试试对Years onehot——效果不好 ！！！
    #train_df['Year'].astype(str)
    #test_df['Year'].astype(str)
    #train_dummy= pd.get_dummies(train_df["Year"])
    #train_df.drop('Year',inplace =True, axis=1)
    #train_df = pd.concat([train_df,train_dummy], axis=1)
    #test_dummy= pd.get_dummies(test_df["Year"])
    #test_df.drop('Year',inplace =True, axis=1)
    #test_df = pd.concat([test_df,test_dummy], axis=1)

    Y= pd.get_dummies(train_df['Category']) #照例onehot处理
    X= train_df.drop(['Category'],axis=1)

    X = X.astype(float)
    
    print(X.info()) 
    return X,Y,test_df,useless_data 
def train(x,y,test): #模型的搭建和训练
    #构建模型
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=42)
    model = Sequential()
    model.add(Dense(100, input_shape=(x.shape[1],),kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Activation('relu'))
    model.add(Dense(100,activation="sigmoid",kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dense(80,activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(Dense(60,activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)))
    #model.add(Dense(40,activation="relu",kernel_regularizer=keras.regularizers.l2(0.001)))

    model.add(Dense(39,activation="softmax"))
    model.compile(optimizer ='rmsprop',loss = 'categorical_crossentropy',metrics=['accuracy'])

    r = model.fit(x_train,y_train, batch_size = 32, epochs = 50, verbose = 2, validation_data=(x,y))
    results = model.evaluate(x_test,y_test)
    
if __name__ == "__main__":
    main()
    #瞎猜的概率是1/39==2.56%