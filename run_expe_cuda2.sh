#!/bin/bash



for pb in QUBO;do
for K in 0 1 2 3 4 5 ;do
for N in 64 128 256 ;do
for activation in relu; do
for nh in -1; do
for numberHiddenLayersG in -1 ; do


python main_rl_eda.py $pb $N $K --verbose --device cuda:2 --numberHiddenLayersG $numberHiddenLayersG --nh $nh --activation $activation 

done
done
done
done
done
done
