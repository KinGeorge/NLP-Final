#!/bin/bash

# 选择模型
echo "This is a training script of Poetry Generation Models"
echo "Please select the model you want to use:"
echo "1. LSTM"
echo "2. GRU"
echo "3. Seq2Seq (with attention)"
echo "4. Transformer"
echo "5. GPT-2"
read -p "Please input the number to select the model: " model_choice

# 选择数据处理方法
echo "Please select the data processing method:"
echo "1. Original Data + No Word2Vec"
echo "2. Original Data + Word2Vec"
echo "3. Augmented Data + No Word2Vec"
echo "4. Augmented Data + Word2Vec"
read -p "Please input the number to select the data processing method: " data_choice

echo "You choose model $model_choice"
echo "You choose data processing method $data_choice"

# 根据用户选择运行不同的训练程序
if [ $model_choice -eq 1 ]; then
    if [ $data_choice -eq 1 ]; then
        python train.py --model lstm --data original --use_w2v False
    elif [ $data_choice -eq 2 ]; then
        python train.py --model lstm --data original --use_w2v True
    elif [ $data_choice -eq 3 ]; then
        python train.py --model lstm --data augmented --use_w2v False
    elif [ $data_choice -eq 4 ]; then
        python train.py --model lstm --data augmented --use_w2v True
    else
        echo "Invalid input!"
    fi
elif [ $model_choice -eq 2 ]; then
    if [ $data_choice -eq 1 ]; then
        python train.py --model gru --data original --use_w2v False
    elif [ $data_choice -eq 2 ]; then
        python train.py --model gru --data original --use_w2v True
    elif [ $data_choice -eq 3 ]; then
        python train.py --model gru --data augmented --use_w2v False
    elif [ $data_choice -eq 4 ]; then
        python train.py --model gru --data augmented --use_w2v True
    else
        echo "Invalid input!"
    fi
elif [ $model_choice -eq 3 ]; then
    if [ $data_choice -eq 1 ]; then
        python train.py --model seq2seq --data original --use_w2v False
    elif [ $data_choice -eq 2 ]; then
        python train.py --model seq2seq --data original --use_w2v True
    elif [ $data_choice -eq 3 ]; then
        python train.py --model seq2seq --data augmented --use_w2v False
    elif [ $data_choice -eq 4 ]; then
        python train.py --model seq2seq --data augmented --use_w2v True
    else
        echo "Invalid input!"
    fi
elif [ $model_choice -eq 4 ]; then
    if [ $data_choice -eq 1 ]; then
        python train.py --model transformer --data original --use_w2v False
    elif [ $data_choice -eq 2 ]; then
        python train.py --model transformer --data original --use_w2v True
    elif [ $data_choice -eq 3 ]; then
        python train.py --model transformer --data augmented --use_w2v False
    elif [ $data_choice -eq 4 ]; then
        python train.py --model transformer --data augmented --use_w2v True
    else
        echo "Invalid input!"
    fi
elif [ $model_choice -eq 5 ]; then
    if [ $data_choice -eq 1 ]; then
        python train.py --model gpt2 --data original --use_w2v False
    elif [ $data_choice -eq 2 ]; then
        python train.py --model gpt2 --data original --use_w2v True
    elif [ $data_choice -eq 3 ]; then
        python train.py --model gpt2 --data augmented --use_w2v False
    elif [ $data_choice -eq 4 ]; then
        python train.py --model gpt2 --data augmented --use_w2v True
    else
        echo "Invalid input!"
    fi
else
    echo "Invalid input!"
fi