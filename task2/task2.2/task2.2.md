# task2.2实验报告

> 实验环境: 
> - System: windows10
> - NNI version: 1.9
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
  #choice: TPE, Random, Anneal, Evolution, BatchTuner, MetisTuner, GPTuner
  #SMAC (SMAC should be installed through nnictl)
  builtinTunerName: TPE
  classArgs:
    #choice: maximize, minimize
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
    "model":{"_type":"choice", "_value":["vgg", "resnet18", "googlenet", "densenet121", "mobilenet", "dpn92", "senet18"]},
    "lr": {"_type": "choice", "_value": [0.1, 0.01,0.001,0.0001]},
    "epochs": {"_type": "choice", "_value":[20,200]}
}
```



##代码

`main.py`:

```

```




##结果展示