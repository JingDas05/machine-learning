# -*- coding: UTF-8 -*-


def test():
    print(1)
    # 路径是相对路径，从当前的根目录下进行寻找
    fr = open('test.txt')
    print(type(fr))
    for line in fr.readlines():
        print(type(line))
        print(line)
