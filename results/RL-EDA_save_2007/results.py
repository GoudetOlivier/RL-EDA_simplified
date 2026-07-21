import pandas as pd
import numpy as np
import os


for L in range(0,3):
	for nh in [64, 128, 256, 512, 1024]:
		for activation in ["relu"]:

			print("L : " + str(L))
			print("nh : " + str(nh))
			print("activation : " + activation)

			

			for pb in [ "QUBO"]: 

				avg_pb = 0
				cpt = 0

				for dim in [64, 128, 256]:

					for K in [0,1,2,3,4,5]:

					#for K in [1,2,4,8]:



						path = "./" + pb + "/" + str(dim) + "/" + str(K) + "/"

						if(os.path.exists(path)):
						
							list_file = os.listdir(path)
							
							for file_ in list_file:
								if("L_" + str(L) + "_nh_" + str(nh) + "_activation_" + activation in file_):

									try:
										data = pd.read_csv(path + file_)
										score = -data.iloc[-1,1]
										print(pb + " " + str(dim) + " " + str(K) + " : " + str(score))

										avg_pb += score
										cpt += 1
									except:
										print("An exception occurred")

				if(cpt > 0):
					print("avg " + pb + " :" + str(avg_pb/cpt))



