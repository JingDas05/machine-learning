# -*- coding: UTF-8 -*-
import os
import shutil
import sys


class Copy(object):
    """Copy classifier.
    将源文件的所有文件都拷贝到目标文件夹
    Parameters
    ------------
    target_file_path : string 目标文件夹

    """

    def __init__(self, target_file_path):
        if target_file_path == "":
            return
        self.target_file_path = target_file_path
        # 创建目标文件夹
        os.mkdir(self.target_file_path)

    # 递归 将文件夹下面的文件全部拷贝到目标文件夹里面  initial_path : string 源文件文件夹
    def recursion_copy(self, initial_path):
        file_names = os.listdir(initial_path)
        if len(file_names) == 0:
            return
        for file_name in os.listdir(initial_path):
            file_name_path = os.path.join(initial_path, file_name)
            if os.path.isdir(file_name_path):
                self.recursion_copy(file_name_path)
            else:
                target_file_path = os.path.join(self.target_file_path, file_name)
                if os.path.exists(target_file_path):
                    continue
                shutil.copyfile(file_name_path, target_file_path)


# if __name__ == "__name__":
#     print("开始2")
#     if len(sys.argv) == 3:
#         print("开始")
#         copy = Copy(sys.argv[2])
#         copy.recursion_copy(sys.argv[1])
#     else:
#         raise Exception("请输入正确的源文件和目标文件")
