#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > model_conert.py.py
# Author: bornchow
# Date:20210926
# 本程序将模型转换为libtorch 可以调用的版本
# https://www.jianshu.com/p/02f16439892d
# ------------------------------------

import torch
from model import PointModel, ElevaModelSelf, ElevaModel

# MODEL = "pointModel"
# MODEL = "CNNModel_self"
MODEL = "CNNModel_resnet"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if MODEL == "pointModel":

    print("point model convert ....")
    SAMPLE_POINTS = 500

    model = PointModel().to(device=device)
    model.load_state_dict(torch.load("/home/bornchow/workFile/navi_net/model/model_1576_25.226139.pth"))
    model.eval()

    sim_input = torch.rand(289, 3, SAMPLE_POINTS).to(device=device)  # 这里的batch_size到底是1还是多少尼


    traced_script_module = torch.jit.trace(model, sim_input)

    traced_script_module.save("./model_1576_25.226139.pt")
elif MODEL == "CNNModel_resnet":

    # model is CNN model
    print("cnn resnet model convert ....")

    IMG_SIZE = 20

    model = ElevaModel().to(device=device)
    model.load_state_dict(torch.load("/home/bornchow/workFile/navi_net/model/cnnresnet_model_119_141.968216.pth"))
    model.eval()

    print(model)

    sim_input = torch.rand(289, 1, IMG_SIZE, IMG_SIZE).to(device=device)

    traced_script_module = torch.jit.trace(model, sim_input)

    out = traced_script_module(sim_input)

    traced_script_module.save("/home/bornchow/workFile/navi_net/model/cnnresnet_model_119_141.968216.pt")

    print("model converted !!!")

elif MODEL == "CNNModel_self":

    # model is CNN model
    print("cnn self model convert ....")

    IMG_SIZE = 20

    model = ElevaModelSelf().to(device=device)
    model.load_state_dict(torch.load("/home/bornchow/workFile/navi_net/model/cnnself_model_926_33.497849.pth"))
    model.eval()

    print(model)

    sim_input = torch.rand(289, 1, IMG_SIZE, IMG_SIZE).to(device=device)

    traced_script_module = torch.jit.trace(model, sim_input)

    out = traced_script_module(sim_input)

    # traced_script_module.save("/home/bornchow/workFile/navi_net/model/cnnself_model_926_33.497849.pt")

    print("model converted !!!")
