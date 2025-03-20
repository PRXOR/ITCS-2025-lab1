#!/bin/bash

# 清空或创建结果文件
> data

# 定义所有需要测试的顺序组合
orders=(mnk mkn kmn nmk nkm knm)

# 遍历每个顺序
for order in "${orders[@]}"; do
    # 先执行不带I参数的命令
    echo "===== Running: make $order =====" >> data
    make $order >> data 2>&1

    # 循环执行I=1到I=8
    for i in {1..8}; do
        echo "===== Running: make $order I=$i =====" >> data
        make $order I=$i >> data 2>&1
    done
done

echo "所有任务已完成，结果保存至 data 文件"