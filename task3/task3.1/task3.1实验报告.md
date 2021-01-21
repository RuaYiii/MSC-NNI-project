# Task3.1实验报告

# 关于特征工程：



## 本机运行实验配置

> nni版本：1.9
>
> Python： 3.8.3
>
> 

`config.yml`:

```yml
authorName: default
experimentName: example-auto-fe
trialConcurrency: 1
maxExecDuration: 10h
maxTrialNum: 2000
#choice: local, remote
trainingServicePlatform: local
searchSpacePath: search_space.json
#choice: true, false
useAnnotation: false
tuner:
  codeDir: .
  classFileName: autofe_tuner.py
  className: AutoFETuner
  classArgs:
    optimize_mode: maximize
trial:
  command: python main.py
  codeDir: .
  gpuNum: 0

```

`search_space.json`

```json
{
    "count":[
        "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
        "C11","C12","C13","C14","C15","C16","C17","C18","C19",
        "C20","C21","C22","C23","C24","C25","C26"
    ],
    "aggregate":[
        ["I9","I10","I11","I12"],
        [
            "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
            "C11","C12","C13","C14","C15","C16","C17","C18","C19",
            "C20","C21","C22","C23","C24","C25","C26"
        ]
    ],
    "crosscount":[
        [
            "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
            "C11","C12","C13","C14","C15","C16","C17","C18","C19",
            "C20","C21","C22","C23","C24","C25","C26"
        ],
        [
            "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10",
            "C11","C12","C13","C14","C15","C16","C17","C18","C19",
            "C20","C21","C22","C23","C24","C25","C26"
        ]
    ]
}
```



## 实验代码

`main.py`:

```python

import nni
import logging
import numpy as np
import pandas as pd
import json
from fe_util import *
from model import *
if __name__ == '__main__':
    file_name = 'train.tiny.csv'
    target_name = 'Label'
    id_index = 'Id# get parameters from tuner
	RECEIVED_PARAMS = nni.get_next_parameter()
	logger.info("Received params:\n", RECEIVED_PARAMS)

# list is a column_name generate from tuner
	df = pd.read_csv(file_name)
	if 'sample_feature' in RECEIVED_PARAMS.keys():
    	sample_col = RECEIVED_PARAMS['sample_feature']
	else:
    	sample_col = []

# raw feaure + sample_feature
    df = name2feature(df, sample_col, target_name)
    feature_imp, val_score = lgb_model_train(df,  _epoch = 1000, target_name = target_name, id_index = id_index)
    nni.report_final_result({
        "default":val_score, 
        "feature_importance":feature_imp
    })
```