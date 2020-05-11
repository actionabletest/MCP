## Introduction of the baseline implementation

paper：[《Adaptive Active Learning for Image Classification》](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Li_Adaptive_Active_Learning_2013_CVPR_paper.pdf)·   

[code](https://astro.temple.edu/~tud51700/research.html) （code provided with the paper）（MATLAB code）

#### summary:  

​	In this paper,  the author first used precomputed **dense SIFT features** for Pascal VOC 2007 and **PHOW features**  for Caltech 101 .  

​	then , based on these precomputed features,  we will **get a fixed dimensional feature** for each test input and **the mutual information** for each test input is computed.  

​	finally, we get **the adaptive combination** between Uncertainty Measure and Information Density Measure for each test input . 

#### our adaption:

1. We only compute **the Dense SIFT features** for different test sets. (mnist, cifar and svhn)
2. For different test sets, we will **compute a 300 dimensional feature for each test input**, which is based on the Dense SIFT features. 
   1. note: we choose the 300 dimensional feature for each test input  according to the code provided by the author.
3. For the adaptive combination between Uncertainty Measure and Information Density Measure  which can be seen from the formula 9 in Page 4 in the [paper](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Li_Adaptive_Active_Learning_2013_CVPR_paper.pdf) : $$h_\beta(x_i)=f(x_i)^{\beta}*d(x_i)^{1-\beta}$$, where $\beta$ is a tradeoff controlling parameter over the two terms. We tested all the parameters $\beta$, from 0.1 to 0.9, and finally found that 0.5 is the best choice, so we chose 0.5 as the fixed parameter of this experiment.

#### steps to reproduce（on windows）：

   1. download MATLAB

   2. use MATLAB to open the **baseline_AdaptAL** folder

   3. to precompute  dense SIFT features for different datasets, we should complete [One-time setup for vlfeat](https://www.vlfeat.org/install-matlab.html)

         ```mariadb
         run('vlfeat-0.9.21\toolbox\vl_setup.m')
         ```

         To check that VLFeat is successfully installed, try to run the `vl_version` command:

         ```matlab
         vl_version verbose
         ```

   4. ***compute the fixed 300 dimensional feature*** for each test input in specific test set:(such as: mnist 10000\*28 \*28 ===>>> 300\*10000)

               1. first, due to the process of computing the final 300 dimensional feature involving **k-means cluster**,which will make the result a little bit different during each running. we ***provide our precomputed  300 dimensional features for different test set***, which is like the code provided by the author .
               2. if you want to compute the  intermediate 300 dimensional features by yourself, you can do as follows:
                        1. to compute the Dense SIFT features for the test set :
                                 1. open `vlfeat-0.9.21\apps\recognition` in MATLAB.
                                 2. you should prepare two folders under `vlfeat-0.9.21\apps\recognition\testSet`  for each test set . which is like what we have provided under `vlfeat-0.9.21\apps\recognition\testSet`
                                          1. under folder  `mnist_bim-a_compound8`, you should provide a .m file which contains the whole test set. (such as mnist, you should provide 10000\*28\*28)
                                          2. to `mnist_bim-a_compound8_Features`,  it will store the mediate value. which you can delete after getting the final 300 dimensional feature
                              3. to run newPrecess.m successfully,  you should modify the directory.
            3. finally, we have provided the precomputed 300 dimensional features for different test sets (mnist, cifar, svhn) under `baseline_AdaptAL\CVPR13`. 

   5. to ***compute the mutual information*** for each test input :

            1. this computation will **take most of the time** to get the final metric values. so we also **provide the mutual information measure results for each test set under ** `baseline_AdaptAL\CVPR13`
            2. if you want to compute the mutual information measure results by yourself. then:
                     1. to `run the myscript.m` successfully, you should modify the input and output directory for the final 300 dimensional features and the mutual information measure results respectively.

   6. to ***compute the final AdaptAL_value*** for each test input:

            1. you should provide ***the softmax values for each test input*** . note we have provided them under `baseline_AdaptAL\CVPR13`
            2. to run `mycalentropy.m` successfully, you should **modify** the argument of two `load` function. one represents **the file storing softmax values**, the other represents **the file storing mutual information measure results** . then you can get the final results.
            3. finall, we also provide the final the final AdaptAL_values for different test sets .
           
   7. finally,  you can **get the sorted test set** based on the final AdaptAL_value. (Descending order) 



#### conclusion:

​		you can run `baseline_AdaptAL\vlfeat-0.9.21\apps\recognition\newPrecess.m`  with some modifications to get the final 300 dimensional feature for each test set.  we ***<u>suggest that you use our precomputed results</u>***.   which is under `baseline_AdaptAL\CVPR13\cifar`, `baseline_AdaptAL\CVPR13\mnist`, `baseline_AdaptAL\CVPR13\svhn` respectively for different test set .

​		you can run `baseline_AdaptAL\CVPR13\myscript.m`  with some modifications to get the mutual information measure for each test set.  because the computation takes most time during our experiment, so ***<u>we suggest that you should use our precomputed results</u>***. which is under `baseline_AdaptAL\CVPR13\cifar_finalmutualinfo`, `baseline_AdaptAL\CVPR13\mnist_finalmutualinfo`, `baseline_AdaptAL\CVPR13\svhn_finalmutualinfo` respectively for different test set . 

​		you can run `baseline_AdaptAL\CVPR13\mycalentropy.m `  with some modifications to get the  AdaptAL_value for each test input.  we also provide the results, which is under `baseline_AdaptAL\CVPR13\cifar_finalResults`, `baseline_AdaptAL\CVPR13\mnist_finalResults`, `baseline_AdaptAL\CVPR13\svhn_finalResults` respectively for different test set . 

​		all those modifications are just different specific directories for different test set, which are described in different files. 

