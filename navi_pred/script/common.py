#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > common.py.py
# Author: bornchow
# Date:20210922
#
# ------------------------------------

import sys


class Logger(object):
    def __init__(self, file_name='Default.log'):
        self.terminal = sys.stdout
        self.log = open(file_name, 'a')

    def write(self, message):
        '''print实际相当于sys.stdout.write'''
        self.terminal.write(message)
        try:
            self.log.write(message)
        except ValueError:
            pass

    def flush(self):
        pass

    def close(self):
        self.log.close()


if __name__ == "__main__":
    sys.stdout = Logger("test.txt")
    i = 10
    while i > 0:
        print(i)
        if i < 5:
            sys.stdout.close()
        i = i - 1
