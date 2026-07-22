#!/bin/bash



for pb in NK3;do
for K in  4 2 1 8;do
for N in 64 128 256 ;do
for activation in relu; do
for numberHiddenLayersG in -1 ; do
for nh  in -1 ; do

python main_rl_eda.py $pb $N $K --verbose --device cuda:0 --numberHiddenLayersG $numberHiddenLayersG --activation $activation --nh $nh 

done
done
done
done
done
done



