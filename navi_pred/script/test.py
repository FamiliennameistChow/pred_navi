#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > test.py
# Author: bornchow
# Date:20210924
#
# ------------------------------------

num_list = [1, 2, 3, 4, 5]
print(num_list)

for i in range(len(num_list)):
    if i >= len(num_list):
        break
    print("i ", i)
    if num_list[i] == 2:
        num_list.remove(num_list[i])
    else:
        print(num_list[i])

print(num_list)

