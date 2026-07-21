import sys
sys.path.append('/code')
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as krs

from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from tkt.utils import *
from tkt.math import *
from tkt.tf import *
#指定数据集路径
Path = r'/MOFoutput-Chop'

#加载数据
def loader(size: tuple, batch_size=8, cache=False, seed=0):
    def fn(inp):
        x = tf.io.read_file(inp)
        x = tf.io.decode_jpeg(x)
        #x = x[p0[0]:p1[0], p0[1]:p1[1]]
        x = tf.image.resize(x, size[:2])
        x = x / 255.0

        return x
#指定x和y
    def build(inp, inpN, training=True):
        x1 = DSLoader.normal(inp['x1'], map_fn=fn)
        x2 = DSLoader.normal(inp['x2'], map_fn=fn)
        x3 = DSLoader.normal(inp['x3'], map_fn=fn)
        x4 = DSLoader.normal(inp['x4'], map_fn=fn)
        x5 = DSLoader.normal(inp['x5'], map_fn=fn)
        xn = DSLoader.normal(inpN)
        x = tf.data.Dataset.zip((x1, x2, x3, x4, x5, xn))
        y1 = DSLoader.normal(inp['y1'])        
        y2 = DSLoader.normal(inp['y2'])
        y3 = DSLoader.normal(inp['y3'])
        y = tf.data.Dataset.zip((y1, y2, y3))

        if training:
            ds = tf.data.Dataset.zip((x, y))
            ds = ds.cache() if cache else ds
            ds = ds.batch(batch_size)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds
        else:
            x = tf.data.Dataset.zip((x,))
            x = x.batch(batch_size)
            return x, inp[['y1', 'y2', 'y3']].to_numpy()

    x = File.getFiles(Path, 'Dirs')[0]
    y = pd.read_csv(Path + '/MOFdataYN-3.csv')
    data = []
    dataN = []
    for file in x:
        if '.ipynb_checkpoints' in file:
            continue
        id = file.split('/')[-1]
        temp = ['/{}/{}-{}.jpg'.format(file, id, i) for i in range(1, 6)]
        for t in temp:
            assert File.exists(t), t
        temp.append(y.loc[y['ID'] == float(id), 'H2 '].values[0])
        temp.append(y.loc[y['ID'] == float(id), 'CO2*100'].values[0])
        temp.append(y.loc[y['ID'] == float(id), 'surface'].values[0])
        
        data.append(temp)
        dataN.append(y.loc[y['ID'] == int(id)].to_numpy()[0, 1:-3])
    data = pd.DataFrame(data, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'y1', 'y2', 'y3'])
    dataN = np.array(dataN)
    train, test = train_test_split(data, test_size=0.2, random_state=seed)
    trainN, testN = train_test_split(dataN, test_size=0.2, random_state=seed)
    train = build(train, trainN)    

    val, test = train_test_split(test, test_size=0.5, random_state=seed)
    valN, testN = train_test_split(testN, test_size=0.5, random_state=seed)
    val = build(val, valN)
    xt, yt = build(test, testN, False)

    

    return train, val, xt, yt

#整体框架
def getModel(sub_model, para, shape, out_dim=3, activation='relu'):
    """
    inp1 = krs.Input(shape)
    x1 = sub_model(para, inp1, 1)
    inp2 = krs.Input(shape)
    x2 = sub_model(para, inp2, 2)
    inp3 = krs.Input(shape)
    x3 = sub_model(para, inp3, 3)
    inp4 = krs.Input(shape)
    x4 = sub_model(para, inp4, 4)
    inp5 = krs.Input(shape)
    x5 = sub_model(para, inp5, 5)

    x = krs.layers.Concatenate(axis=-1, name='Merge')([x1, x2, x3, x4, x5])
    """
    inp1 = krs.Input(shape, name="input_1")
    inp2 = krs.Input(shape, name="input_2")
    inp3 = krs.Input(shape, name="input_3")
    inp4 = krs.Input(shape, name="input_4")
    inp5 = krs.Input(shape, name="input_5")
    inpn = krs.Input((54,), name="input_n")
    # 将 (None, 90) 扩展为 (None, 300, 300, 3)
    img_features = krs.layers.Concatenate(axis=-1, name='merge')([inp1, inp2, inp3, inp4, inp5])
    x = sub_model(para, img_features, 0)
    
    inpnx1 = krs.Sequential([
        krs.layers.Dense(256, activation),
        krs.layers.Dense(128, activation)])(inpn)
    
    x_merged = krs.layers.Concatenate(axis=-1)([x, inpnx1])
    
        
    out1 = krs.Sequential([
        krs.layers.Dense(256, activation),
        krs.layers.Dense(128, activation),
        krs.layers.Dense(1)
    ], name='Dense_out1')(x_merged)
    
    out2 = krs.Sequential([
        krs.layers.Dense(256, activation),
        krs.layers.Dense(128, activation),
        krs.layers.Dense(1)
    ], name='Dense_out2')(x_merged)
    
    out3 = krs.Sequential([
        krs.layers.Dense(256, activation),
        krs.layers.Dense(128, activation),
        krs.layers.Dense(1)
    ], name='Dense_out3')(x_merged)
    
    md = krs.Model([inp1, inp2, inp3, inp4, inp5, inpn], [out1, out2, out3])
    #md.compile('Nadam', 'MAE', ['MSE', ])
    '''
    md.compile(krs.optimizers.Nadam(0.001), 
    ['out1':'MAE', 'out2':'MAE','out3':'MAE'], #各Y损失函数指定（多Y指定）
    ['out1':'MAE', 'out2':'MAE','out3':'MAE'])#损失函数验证
    '''
    md.compile(
    krs.optimizers.Nadam(0.001),
    loss={
        'Dense_out1': 'mae',
        'Dense_out2': 'mae',
        'Dense_out3': 'mae'
    },  # 为每个输出定义对应的损失函数
    metrics={
        'Dense_out1': [tf.keras.metrics.MeanAbsoluteError(name="mae_out1"),
                    tf.keras.metrics.MeanSquaredError(name="mse_out1")],
        'Dense_out2': [tf.keras.metrics.MeanAbsoluteError(name="mae_out2"),
                    tf.keras.metrics.MeanSquaredError(name="mse_out2")],
        'Dense_out3': [tf.keras.metrics.MeanAbsoluteError(name="mae_out3"),
                    tf.keras.metrics.MeanSquaredError(name="mse_out3")],
    }  # 使用字典显式定义每个输出的指标
    )
    return md

#单个图像对应模型（模型可更改）


def MOFNetseraHP(shape, out_dim=3, activation='relu'):
    def SubModel(para, inp, index):
        x = Layers.ImgArgumentation(True, (-0.2, 0.2), (-0.125, 0.125), None, None)(inp)
        x = Blocks.MOFNetsera.getModel(x.shape[1:], out_dim, activation, 'V1', False, 'Net{}'.format(index),
                                   input_block_dim=para[0], cablock_num=para[1],
                                   cablock_conv_dim=para[2], cablock_att_dim=para[3])(x)#定义模型参数
        return x

    def get(para):
        return getModel(SubModel, para, shape, out_dim, activation)

    return get

#多Y指定
def wrapper(fn, idx):
    def loss_fn(x, y):
        return fn(x[:, idx], y[idx].reshape(-1))
    return loss_fn
    
#任务实际运行    
def job(epoch, shape):
    out_dim = 3
    para = (32,3,32,64)
    name = 'L18-MOFNetsera-in5Nout3-3-{}-{}'.format(Path.split('/')[-1], shape[0])#跑参数用
    #name = 'CAPNet-in3out1-{}'.format(Path.split('/')[-1])#跑完参数后优化用
   
    
    train, val, xt, yt = loader(shape, 16, False)
    
    bt = MTO.BasicTrainer(name, './Models/YN/test', dtype='float32', seed=0)#指定模型训练结果存放路径
    #bt.setData(train, val, xt, yt, [wrapper(Loss.R2,0), wrapper(Loss.MAE,0), wrapper(Loss.RMSE,0), wrapper(Loss.R2,1), wrapper(Loss.MAE,1), wrapper(Loss.RMSE,1)])#单输出的损失函数
    #bt.setData(train, val, xt, yt, [Loss.R2, Loss.MAE, Loss.RMSE])
    bt.setData(train, val, xt, yt, {'fn1': wrapper(Loss.R2,0), 'fn2': wrapper(Loss.MAE,0), 'fn3': wrapper(Loss.RMSE,0), 
        'fn4': wrapper(Loss.R2,1), 'fn5': wrapper(Loss.MAE,1), 'fn6': wrapper(Loss.RMSE,1), 
        'fn7': wrapper(Loss.R2,2), 'fn8': wrapper(Loss.MAE,2), 'fn9': wrapper(Loss.RMSE,2)})
    
#跑参数优化用
    bt.addHPModel(MOFNetseraHP(shape, out_dim), Permutation.getOrthTable(
      [[32, 64, 96], [2, 3, 4], [32, 64, 96], [32, 64, 96]], Permutation.L18_3_7))#正交L18，L9
#跑模型用    
    #bt.addModel('{}-res{}-e{}-0'.format(para, shape[0], epoch), CAPNetHP(shape, out_dim)(para))
    
    #for i in range(0):
        #bt.addModel(f'{para}-res{shape[0]}-e{epoch}-{i+1}', CAPNetHP(shape, out_dim)(para))#指定调用的模型
    #bt.test(restart=1)#测试运行
    bt.fit(epoch, restart=1, auto_adjust_lr=1)#实际运行


if __name__ == '__main__':
    #Config.setMemSize(0, 8*1024)#设定GPU内存，一共24G不能写满
    job(300, (300, 300, 3))#跑参数用迭代次数，分辨率
    #for i in [900,800,700,600,500]:
        #for j in [300, 400, 500]:#指定优化时的分辨率范围
            #job(i, (j, j, 3))#优化用迭代次数和分辨率
    

