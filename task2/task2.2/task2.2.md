# task2.2实验报告

> 实验环境: 
> - System: windows10
> - NNI version: 2.0
> - Python version: 3.8.3
> - Pytorch version: 1.6.0
> - Tensorflow version: 2.3.0
> - Numpy version: 1.18.5
> - Matplotlib version: 3.2.2 
> - Torchvision version: 0.7.0

## 配置文件

`config_windows.yml`:
```yml
authorName: RuaYiii
experimentName: NNi_task2.2
trialConcurrency: 4
maxExecDuration: 240h
maxTrialNum: 10
trainingServicePlatform: local
searchSpacePath: NNI_test.json
useAnnotation: false
tuner:
  builtinTunerName: TPE
  classArgs:
    optimize_mode: maximize
trial:
  command: python main.py
  codeDir: .
  gpuNum: 1
localConfig:
  maxTrialNumPerGpu: 1
```

`NNI_test.json`:

```json
{  
    "optimizer":{"_type":"choice", "_value":["SGD", "Adadelta", "Adagrad", "Adam", "Adamax"]},
    "model":{"_type":"choice",
             "_value":["vgg11","vgg13","vgg16", "vgg19",
                       "googlenet",
                    "densenet121","denseet161","denseet169","denseet201"
                      ]},
    "lr": {"_type": "choice", "_value": [0.1, 0.01,0.001,0.0001]},
    "epochs": {"_type": "choice", "_value":[20,200]}
}
```


##代码

`main.py`:

```python
if __name__ == "__main__": 
parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batches", type=int, default=-1)
    args, _ = parser.parse_known_args()
    parameter=nni.get_next_parameter()
    _logger.debug(parameter)
    train_acc,test_acc= main_pytorch(parameter)
    print(parameter)
    print(f"best train acc: {train_acc}")
    print(f"best test acc: {test_acc}")
    #nni.report_final_result(end)
    print("OK")
```


## 结果展示:

> 鉴于本机低劣的内存和显卡的性能限制，googlenet的batch_size为64，densenet的batch_size为32————仅为笔者在实验中跑程序之用

vgg模型下表现：

```
{'optimizer': 'Adamax', 'model': 'vgg19', 'lr': 0.01, 'epochs': 200}
best train acc: 99.546
best test acc: 91.28
{'optimizer': 'Adam', 'model': 'vgg13', 'lr': 0.001, 'epochs': 200}
bestacc: 92.38
{'optimizer': 'Adam', 'model': 'vgg11', 'lr': 0.001, 'epochs': 200}
bestacc: 90.5
{'optimizer': 'Adamax', 'model': 'vgg16', 'lr': 0.005, 'epochs': 200}
bestacc: 92.02
{'optimizer': 'SGD', 'model': 'vgg19', 'lr': 0.001, 'epochs': 200}
bestacc: 90.42
{'optimizer': 'SGD', 'model': 'vgg16', 'lr': 0.001, 'epochs': 200}
bestacc: 90.32
{'optimizer': 'Adam', 'model': 'vgg19', 'lr': 0.001, 'epochs': 200}
bestacc: 92.09
{'optimizer': 'Adam', 'model': 'vgg16', 'lr': 0.001, 'epochs': 200}
bestacc: 92.26
```

googlenet的表现：

```
{'optimizer': 'Adam', 'model': 'googlenet', 'lr': 0.01, 'epochs': 200}
best train acc: 99.604
best test acc: 92.8
```

 densenet的表现：

```
{'optimizer': 'Adam', 'model': 'densenet121', 'lr': 0.001, 'epochs': 200}
best train acc: 99.79599999999999
best test acc: 93.78
```

```
{'optimizer': 'Adam', 'model': 'densenet169', 'lr': 0.001, 'epochs': 200}
best train acc: 99.8
best test acc: 93.82
```

```

```

```

```

```

```
