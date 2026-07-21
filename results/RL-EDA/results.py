import pandas as pd
import numpy as np
import os


for L in range(0,1):
	for nh in [-1,100]:
		for activation in ["relu"]:

			print("L : " + str(L))
			print("nh : " + str(nh))
			print("activation : " + activation)

			

			for pb in [ "NK3"]: 

				avg_pb = 0
				cpt = 0

				for dim in [64, 128, 256]:

					#for K in [0,1,2,3,4,5]:

					for K in [1,2,4,8]:



						path = "./" + pb + "/" + str(dim) + "/" + str(K) + "/"

						if(os.path.exists(path)):
						
							list_file = os.listdir(path)
							
							if(nh ==-1):
                                                        	test = dim
							else:
								test = nh

							for file_ in list_file:
								if("_nh_" + str(test) + "_activation_" + activation in file_):

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



