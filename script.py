import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

end = time.time()
#different k
k_list = [4, 8, 16, 32, 64, 128]
a_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
for k in k_list:
    for a in a_list:
        print('k = {}, a = {}'.format(k, a))
        os.system("python main.py --k {} --a {} ".format(k, a))  #(first, sec)

#different k different batch for k


#different alpha