# -*- coding: utf-8 -*-
# @File    : eval.py
# @Author  : AaronJny
# @Time    : 2019/12/30
# @Desc    :
from dataset import tokenizer
from model import model
import settings
import utils



# 加载训练好的模型
model.load_weights(settings.BEST_MODEL_PATH)
# 随机生成一首诗
print(utils.generate_random_poetry(tokenizer, model))
# 给出部分信息的情况下，随机生成剩余部分
print(utils.generate_random_poetry(tokenizer, model, s='会当凌绝顶，一览众山小。'))
print(utils.generate_random_poetry(tokenizer, model, s='大漠孤烟直，长河落日圆。'))
print(utils.generate_random_poetry(tokenizer, model, s='狗吠深巷中，鸡鸣桑树颠。'))
print(utils.generate_random_poetry(tokenizer, model, s='海内存知己，天涯若比邻。'))
print(utils.generate_random_poetry(tokenizer, model, s='归园田居'))
print(utils.generate_random_poetry(tokenizer, model, s='边塞壮景'))
print(utils.generate_random_poetry(tokenizer, model, s='归家心切'))

# 生成藏头诗
print(utils.generate_acrostic(tokenizer, model, head='中山大学'))
print(utils.generate_acrostic(tokenizer, model, head='自然语言'))
print(utils.generate_acrostic(tokenizer, model, head='山高水长'))
print(utils.generate_acrostic(tokenizer, model, head='鸡你太美'))
