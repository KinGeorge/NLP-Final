# NLP Final Project

```shell
├── configs
├── data
│   ├── poetry.txt
│   ├── poetry_7.txt
│   ├── org_poetry.txt
│   ├── split_poetry.txt
│   └── word_vec.pkl
├── inference.py
├── src
│   ├── apis
│   │   └── train.py
│   ├── datasets
│   │   └── dataloader.py
│   ├── models
│   │   └── model
│   └── utils
│       └── utils.py
├── test.py
└── train.py
```

## Train and Inference
You can train the model by running the following command:
```shell
python train.py --model [model_name]
```
You can inference the model by running the following command:
```shell
python inference.py --model [model_name] --save_path 'save_models/xxx.pth'
```
Now we have model cards:
- lstm
- gru