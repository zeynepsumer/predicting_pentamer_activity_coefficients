# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:08:16 2022
@author: zsumer
"""

import joblib
from sklearn.preprocessing import minmax_scale

# INPUT PARAMETERS###
alpha = 0       #min 0, max 1.17
beta = 0.1      #min -0.08, max 1.43
pi = 0.1        #min -0.41, max 1.24
S_ratio = 0.3
H_ratio = 0.3
#####################

model = joblib.load("Kamlet_Taft_Model.joblib")
input_parameters = [alpha, beta, pi, S_ratio, H_ratio]

KT = [[0, -0.08, -0.41],[1.17, 1.43, 1.24],[alpha, beta, pi]]
scaled_KT = minmax_scale(KT)[2]

result = model.predict([[scaled_KT[0],scaled_KT[1],scaled_KT[2],S_ratio,H_ratio]])[0]

print("Activity coefficient is predicted as: {:.2f}".format(result))