# task3.2.1实验报告

> 本个项目任务围绕特征工程（Feature Engineering）和自动特征工程（Automated Feature Engineering）开展，包括但不仅限于这些问题：
> - 特征筛选和特征重要性的估计
>   - 自动特征工程的核心就是发现不同特征的重要程度。同学们可以使用 NNI 中已有的算法进行特征生成和特征筛选，找出相对重要的特征。当然，我们更希望同学们能沿着前人的脚步，尝试为 NNI 贡献新的 operation ，或特征筛选的算法。
> - 特征搜索空间的设计
>   - 表格数据的特征搜索空间的设计：对于表格数据，特征工程起到非常重要的作用。除了搜索空间的设计，处理表格数据的时候还需要
> 别注意数据预处理、数据编码方式、高阶特征挖掘和基于其他分类的的特征提取等问题。这些问题有些已经有了成熟的解决方案，有些值得我们去挖掘研究。如果能够在处理的时候考虑到它们，可以更加完善和丰富自动特征工程的功能。
>   - 典型问题的特征搜索空间的设计：对于不同领域，特征的搜索空间可能有很大的差别。例如，对于 CTR 预估这个问题来说，N阶交叉特征可能起到比较重要作用；对于时间序列预测问题来说，时间相关的特征可能起比较重要的作用。为不同领域设计特征搜索的空间，并提供相关特征抽取的模块，是一件非常有意义的事情。

**在本项目中选择了：旧金山犯罪分类：San Francisco Crime Classification**

## 原始数据

以下是对数据的说明：
--- 
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
    ''' 让我们对旧金山犯罪分类数据进行说明：一共34243行
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
---   
## 数据预处理

对于原始数据，我们预先对Descript、Resolution、Address列暂时去除————处于其种类过多的考虑，同时对DayOfWeek、PdDistrict进行了onehot处理；对经纬度X、Y进行了标准化

## 数据编码方式的处理

对于时间型数据Dates，我们将其分解为Year,Month,Day,Hours四列数据，其他数值型的数据全部astype为float

## 实验结果


