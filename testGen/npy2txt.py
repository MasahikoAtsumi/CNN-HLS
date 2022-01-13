import numpy as np
import random
import argparse

parser = argparse.ArgumentParser(description='create header files')
parser.add_argument('--data_dir',  help='path numpy')
parser.add_argument('--data_type', help='data type: float or int')
parser.add_argument('--save_type', help='header file: flatten or asArray')
args = parser.parse_args()

def saver(save_name, param, str(data_type)):
    f = open(str(save_name)+'.h', 'a')
    f.write(data_type+str(save_name)+"["+str(len(param))+"] = {" )
    print("length", len(param))
    for id_pram in range(0, len(param)):
        f.write(str(param[id_pram])+",")
        
    f.write("};")
    f.close()

# def asArray_saver(save_name, param, str(data_type)):
#     num_dim    = param.ndim
#     shape      = param.shape
#     size_array = np.array([])
#     f = open(str(save_name)+'.h', 'a')
#     f.write(data_type+str(save_name))
#     for id_dim in range(0, num_dim):
#         f.write("["+shape[id_dim]+"]")
#         np.append(size_array, shape[id_dim])

#     # opening
#     f.write(" = {" )

#     # content
    
#     for id_pram in range(0, len(param)):
#         f.write(str(param[id_pram])+",")

#     # closing    
#     f.write("};")
#     f.close()

def_root = args.data_dir
bias     = np.load(def_root + './npy/features.0.bias.npy'  )
weight   = np.load(def_root + './npy/features.0.weight.npy')
bias1    = np.load(def_root +'./npy/features.3.bias.npy'  )
weight1  = np.load(def_root +'./npy/features.3.weight.npy')
bias2    = np.load(def_root +'./npy/features.5.bias.npy'  )
weight2  = np.load(def_root +'./npy/features.5.weight.npy')
in_data  = np.load(def_root +'./npy/in_data.npy')
out_data = np.load(def_root +'./npy/out_data.npy')

if args.data_type == 'int':
    bias     = bias.astype(int)
    weight   = weight.astype(int)
    bias1    = bias1.astype(int)
    weight1  = weight1.astype(int)
    bias2    = bias2.astype(int)
    weight2  = weight2.astype(int)
    in_data  = in_data.astype(int)
    out_data = out_data.astype(int)
else:
    bias     = bias.astype(float)
    weight   = weight.astype(float)
    bias1    = bias1.astype(float)
    weight1  = weight1.astype(float)
    bias2    = bias2.astype(float)
    weight2  = weight2.astype(float)
    in_data  = in_data.astype(float)
    out_data = out_data.astype(float)

if args.sava_type == 'flatten':
    saver('bias_l0'  , bias.flatten(), args.data_type)
    saver('weight_l0', weight.flatten(), args.data_type)
    saver('bias_l1'  , bias1.flatten(), args.data_type)
    saver('weight_l1', weight1.flatten(), args.data_type)                     
    saver('bias_l2'  , bias2.flatten(), args.data_type)
    saver('weight_l2', weight2.flatten(), args.data_type)                    
 
# saver('in_data'  , in_data.flatten() )
# saver('out_data', out_data.flatten())                     
