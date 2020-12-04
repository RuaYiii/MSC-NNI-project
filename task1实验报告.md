# task1

> by 石基宽(RuaYiii)

##安装

安装时首次遇到了`ERROR: Command errored out with exit status 1` 的问题

仔细检查后发现了`Microsoft Visual C++ 14.0 or greater is required. Get it with "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/`这条消息，于是在安装了mvc++更高版本后安装成功（使用`python -m pip install --upgrade nni`）

笔者环境是: 

`Python 3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32`

![image-20201204123152120](img\image-20201204123152120.png)

## 示例代码

之后笔者由于之前的环境是Tensorflow2，这里没有回滚

也就是`nnictl create --config nni\examples\trials\mnist-tfv1\config_windows.yml`

我改成了`nnictl create --config nni\examples\trials\mnist-tfv2\config_windows.yml`

## 实验结果

测试结果如下：

<img src="img\image-20201204123414154.png" alt="image-20201204123414154" style="zoom:67%;" />

<img src="img\image-20201204123429165.png" alt="image-20201204123429165" style="zoom:67%;" />

<img src="img\image-20201204123439559.png" alt="image-20201204123439559" style="zoom:67%;" />![image-20201204123500694](img\image-20201204123500694.png)

<img src="img\image-20201204123439559.png" alt="image-20201204123439559" style="zoom:67%;" />![image-20201204123500694](img\image-20201204123500694.png)

同时也导出了`dispatcher.log` `experiment.json`

`experiment.json` :

```json
    "experimentParameters": {
        "id": "mZjrgupn",
        "revision": 726,
        "execDuration": 6993,
        "logDir": "C:\\Users\\hp\\nni-experiments\\mZjrgupn",
        "nextSequenceId": 21,
        "params": {
            "authorName": "NNI Example",
            "experimentName": "MNIST TF v2.x",
            "trialConcurrency": 1,
            "maxExecDuration": 7200,
            "maxTrialNum": 20,
            "searchSpace": {
                "dropout_rate": {
                    "_type": "uniform",
                    "_value": [
                        0.5,
                        0.9
                    ]
                },
                "conv_size": {
                    "_type": "choice",
                    "_value": [
                        2,
                        3,
                        5,
                        7
                    ]
                },
                "hidden_size": {
                    "_type": "choice",
                    "_value": [
                        124,
                        512,
                        1024
                    ]
                },
                "batch_size": {
                    "_type": "choice",
                    "_value": [
                        16,
                        32
                    ]
                },
                "learning_rate": {
                    "_type": "choice",
                    "_value": [
                        0.0001,
                        0.001,
                        0.01,
                        0.1
                    ]
                }
            },
            "trainingServicePlatform": "local",
            "tuner": {
                "builtinTunerName": "TPE",
                "classArgs": {
                    "optimize_mode": "maximize"
                },
                "checkpointDir": "C:\\Users\\hp\\nni-experiments\\mZjrgupn\\checkpoint"
            },
            "versionCheck": true,
            "clusterMetaData": [
                {
                    "key": "codeDir",
                    "value": "C:\\Users\\hp\\nni\\examples\\trials\\mnist-tfv2\\."
                },
                {
                    "key": "command",
                    "value": "python mnist.py"
                }
            ]
        },
        "startTime": 1607039245718,
        "endTime": 1607050433545
    },
    "trialMessage": []
}
```

`dispatcher.log`: 

```log
[12/04/2020, 07:47:27 AM] INFO (nni.msg_dispatcher_base/MainThread) Start dispatcher
[12/04/2020, 07:47:27 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.001956 seconds
[12/04/2020, 07:47:27 AM] INFO (hyperopt.tpe/Thread-1) TPE using 0 trials
[12/04/2020, 07:58:03 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.002110 seconds
[12/04/2020, 07:58:03 AM] INFO (hyperopt.tpe/Thread-1) TPE using 1/1 trials with best loss -0.991200
[12/04/2020, 08:04:45 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.002020 seconds
[12/04/2020, 08:04:45 AM] INFO (hyperopt.tpe/Thread-1) TPE using 2/2 trials with best loss -0.991200
[12/04/2020, 08:08:58 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.002013 seconds
[12/04/2020, 08:08:58 AM] INFO (hyperopt.tpe/Thread-1) TPE using 3/3 trials with best loss -0.991200
[12/04/2020, 08:17:24 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.001511 seconds
[12/04/2020, 08:17:24 AM] INFO (hyperopt.tpe/Thread-1) TPE using 4/4 trials with best loss -0.991500
[12/04/2020, 08:25:42 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.000997 seconds
[12/04/2020, 08:25:42 AM] INFO (hyperopt.tpe/Thread-1) TPE using 5/5 trials with best loss -0.991500
[12/04/2020, 08:30:23 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.001785 seconds
[12/04/2020, 08:30:23 AM] INFO (hyperopt.tpe/Thread-1) TPE using 6/6 trials with best loss -0.991500
[12/04/2020, 08:39:45 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.001967 seconds
[12/04/2020, 08:39:45 AM] INFO (hyperopt.tpe/Thread-1) TPE using 7/7 trials with best loss -0.991500
[12/04/2020, 08:50:02 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.001020 seconds
[12/04/2020, 08:50:02 AM] INFO (hyperopt.tpe/Thread-1) TPE using 8/8 trials with best loss -0.991500
[12/04/2020, 08:58:22 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.002114 seconds
[12/04/2020, 08:58:22 AM] INFO (hyperopt.tpe/Thread-1) TPE using 9/9 trials with best loss -0.991500
[12/04/2020, 09:06:38 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.002953 seconds
[12/04/2020, 09:06:38 AM] INFO (hyperopt.tpe/Thread-1) TPE using 10/10 trials with best loss -0.992200
[12/04/2020, 09:55:46 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.001789 seconds
[12/04/2020, 09:55:46 AM] INFO (hyperopt.tpe/Thread-1) TPE using 11/11 trials with best loss -0.992200
[12/04/2020, 10:02:10 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.002026 seconds
[12/04/2020, 10:02:10 AM] INFO (hyperopt.tpe/Thread-1) TPE using 12/12 trials with best loss -0.992600
[12/04/2020, 10:06:54 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.000997 seconds
[12/04/2020, 10:06:54 AM] INFO (hyperopt.tpe/Thread-1) TPE using 13/13 trials with best loss -0.992600
[12/04/2020, 10:19:23 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.002028 seconds
[12/04/2020, 10:19:23 AM] INFO (hyperopt.tpe/Thread-1) TPE using 14/14 trials with best loss -0.992600
[12/04/2020, 10:25:23 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.002003 seconds
[12/04/2020, 10:25:23 AM] INFO (hyperopt.tpe/Thread-1) TPE using 15/15 trials with best loss -0.992600
[12/04/2020, 10:30:33 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.001420 seconds
[12/04/2020, 10:30:33 AM] INFO (hyperopt.tpe/Thread-1) TPE using 16/16 trials with best loss -0.992600
[12/04/2020, 10:34:58 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.214294 seconds
[12/04/2020, 10:34:58 AM] INFO (hyperopt.tpe/Thread-1) TPE using 17/17 trials with best loss -0.992600
[12/04/2020, 10:42:51 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.002024 seconds
[12/04/2020, 10:42:51 AM] INFO (hyperopt.tpe/Thread-1) TPE using 18/18 trials with best loss -0.992600
[12/04/2020, 10:48:48 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.000998 seconds
[12/04/2020, 10:48:48 AM] INFO (hyperopt.tpe/Thread-1) TPE using 19/19 trials with best loss -0.993400
[12/04/2020, 10:53:53 AM] INFO (hyperopt.tpe/Thread-1) tpe_transform took 0.001802 seconds
[12/04/2020, 10:53:53 AM] INFO (hyperopt.tpe/Thread-1) TPE using 20/20 trials with best loss -0.993400
```



