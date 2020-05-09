# Multiple-Boundary Clustering and Prioritization to Promote Neural Network Retraining

MCP(Multiple-Boundary Clustering and Prioritization)

This repository stores our experimental codes and part of the simulated datasets for paper `Multiple-Boundary Clustering and Prioritization to Promote Neural Network Retraining'.

Part of the datasets lies in the folder 'dataset'. Because some of our simulated test datasets exceed GitHub's file size limit of 100.00 MB, we can only upload part of our datasets in file '/mnist'. 

The operational details of simulated test datasets are listed in the "README" under the folder 'dataset'.


You can easily implement our method and 5 baseline methods by yourself or modifying this code.

The main experimental codes are 
samedist_mnist_retrain.py, 
samedist_cifar_retrain.py, 
samedist_svhn_retrain.py. 
You can modify the list variables 'baselines'(methods) and 'operators'(simulated dataset) to run what you prefer.
baselines =['MCP','LSA','DSA','CES','AAL','SRS']
operators =['fgsm','jsma','bim-a','bim-b','cw-l2','scale','rotation','translation','shear','brightness','contrast']


MCP. Our method "MCP" is implemented in the "samedist_***_retrain.py" as the function "select_my_optimize".


LSA/DSA. You can directly invoke the functions "fetch_lsa" and "fetch_dsa" from "/LSA_DSA/sa.py" which is downloaded online from the paper "Guiding Deep Learning System Testing Using Surprise Adequacy".

CES. You can directly invoke the functions "conditional_sample" from "CES/condition.py" which is downloaded online from the paper "Boosting Operational DNN Testing Efficiency Through Conditioning".

AAL. This approach is written by Matlab. We have write a individual README in the file "AAL" along with all the codes and our experimental results. For short, you can run MATLAB programs and get the intermediate results stored in mnist_finalResults, cifar_finalResults, svhn_finalResults. Then you can run it in our python codes just as the other baseline methods. 

SRS. This method is implemented in the "samedist_***_retrain.py" as the function "select_rondom".


