# 数据集

数据集的链接如下：[数据集](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy)

请在根目录新建`datasets`文件夹，将数据集解压并整理为以下结构：

```bash
./
├── baseline_rnn.py
├── datasets
│   ├── all_six_datasets.zip
│   ├── electricity
│   │   └── electricity.csv
│   ├── ETT-small
│   │   ├── ETTh1.csv
│   │   ├── ETTh2.csv
│   │   ├── ETTm1.csv
│   │   └── ETTm2.csv
│   ├── exchange_rate
│   │   └── exchange_rate.csv
│   ├── illness
│   │   └── national_illness.csv
│   ├── traffic
│   │   └── traffic.csv
│   └── weather
│       └── weather.csv
├── models
│   └── rnn.py
└── utils
    └── data_preprocessing.py
```

# 代码

对于一个新的网络及其对应的实验结果，请在models文件夹实现一个新的网络结构（参考`./models/rnn.py`），在根目录实现一个实验代码（参考`./baseline_rnn.py`）。

# 运行

在根目录，使用以下命令即可运行实验：

```bash
python baseline_rnn.py --data_path './datasets/weather/weather.csv' --input_len 192 --output_len 96 --hidden_size 64 --lr 0.0001 --epochs 10 --batch_size 32
```