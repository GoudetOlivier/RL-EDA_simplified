import numpy as np
import os
import shutil 

class Nk_generator():

    def __init__(self, n, k, instances, base_folder):
 
        self.N = n
        self.K = k
        self.instances = instances
        self.base_folder = base_folder


        self.train_folder = base_folder


        os.makedirs(self.train_folder, exist_ok=True)


        for i in range(self.instances):

            print(self.train_folder)
            fichier = open(os.path.join(self.train_folder, f"nk_{self.N}_{self.K}_{i}.txt"), "w")
            fichier.write(str(self.N) + " " + str(self.K))


            for x in range(self.N):
                neigh = []
                neigh.append(x)
                for y in range(self.K):
                    x1 = np.random.randint(0, self.N)
                    while x1 in neigh:
                        x1 = np.random.randint(0, self.N)
                    neigh.append(x1)

                neigh.sort()  
                for x2 in neigh:
                    fichier.write("\n" + str(x2))  


            for x in range(self.N):
                for y in range(2 ** (self.K + 1)):
                    fichier.write(
                        "\n" + str(round(np.random.random(), 6)))  
            fichier.close()


    def clean_train_folder(self):
        if os.path.exists(self.train_folder):
            shutil.rmtree(self.train_folder)



for N in [1024]:
    for K in [1, 2, 4, 8]:
        path = "./nk/" + str(N) + "/" + str(K) + "/"  
        Nk_generator(N, K, 10, path)


