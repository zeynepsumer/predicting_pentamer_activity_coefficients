# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:08:16 2022
@author: zsumer
"""

import joblib
from sklearn.preprocessing import minmax_scale

# INPUT PARAMETERS#######
model_type = "KamletTaft"   #"KamletTaft" or "Hansen"

# if model type is KamletTaft
alpha = 1       #min 0, max 1.17
beta = 1     #min -0.08, max 1.43
pi = 1        #min -0.41, max 1.24
# if model type is Hansen
dD = 15        #min 12.10 max 20.40
dP = 10        #min 0 max 18
dH = 10        #min 0 max 42.30
#
S_ratio = 0.3
H_ratio = 0.3
#########################
if model_type == "KamletTaft":
    model = joblib.load("Kamlet_Taft_Model.joblib")
    input_parameters = [alpha, beta, pi, S_ratio, H_ratio]

    KT = [[0, -0.08, -0.41],[1.17, 1.43, 1.24],[alpha, beta, pi]]
    scaled_KT = minmax_scale(KT)[2]

    result = model.predict([[scaled_KT[0], scaled_KT[1], scaled_KT[2], S_ratio, H_ratio]])[0]
    
    print("Activity coefficient is predicted as: {:.2f}".format(result))
    
elif model_type == "Hansen":
    model = joblib.load("Hansen_SP_Model.joblib")
    input_parameters = [dD, dP, dH, S_ratio, H_ratio]

    HS = [[12.10, 0, 0],[20.40, 18, 42.3],[dD, dP, dH]]
    scaled_HS = minmax_scale(HS)[2]

    result = model.predict([[scaled_HS[0], scaled_HS[1], scaled_HS[2], S_ratio, H_ratio]])[0]
    
    print("Activity coefficient is predicted as: {:.2f}".format(result))
    
else:
    print("Please pick a model.")
