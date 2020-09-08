import numpy as np
import math
from sklearn.datasets import load_boston
from sklearn.model_selection  import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import random
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers.core import Dense, Activation

from keras.models import load_model


class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0, max_rmse=22):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate
        self.max_rmse = max_rmse

    def init_args(self, M):
        # 弱分类器数目和集合
        self.clf_sets = []
        self.M = M
        # 初始化weights,共有M个样本
        self.weights = np.array([1.0 / self.M] * self.M)
        self.Em = 0
        self.emrate = []
        # 误差率
        self.alpha = []
        # G(x)系数 alpha
        self.clfseri = 0

    def mod_Em(self, i, data, label):
        # 计算某个模型最大误差
        model_cls=self.clf_sets[i]
        model=model_cls.load_model()
        self.Em = self.configure_mape(label[:, 0],model.predict(data))
        return self.Em

    def mod_G(self, data, label):
        # 线性加权回归器
        a = np.zeros((np.shape(label)))
        #print('a shape=', np.shape(a))

        weight_alpha = sum([math.log(1 / b) for b in self.alpha])
        for i in range(self.clfseri + 1):
            clfmodel = self.clf_sets[i].load_model()
            label_temp = np.array(clfmodel.predict(data)).reshape(np.shape(label))
            #print('label_temp shape=', np.shape(label_temp))
            temp = math.log(1 / self.alpha[i]) / weight_alpha * label_temp
            #print('temp shape', np.shape(temp))
            a += temp
        return a

    def mod_em(self, i, data, label):
        clfmodel = self.clf_sets[i].load_model()
        pred_y=clfmodel.predict(data)
        pred_y=pred_y.reshape(np.shape(label))
        temp = self.configure_mape(label,pred_y)
        self.Em = max(temp)
        print('self.em=',self.Em)
        print(np.shape(pred_y),np.shape(label))
        emi = pow(pred_y - label, 2) / pow(self.Em, 2)
        emi=np.array(emi).reshape(-1)
        print('emi shape=',np.shape(emi))
        self.emrate.append(sum(self.weights * emi))
        #print('self.emrate=', self.emrate)
        self.alpha.append(self.emrate[-1] / (1 - self.emrate[-1]))
        #print('self.alpha=', self.alpha)
        ZM = sum(np.array(self.weights) * np.array([pow(self.alpha[-1], 1 - a) for a in emi]))
        #print('ZM', ZM)
        self.weights = np.array(self.weights / ZM) * np.array([pow(self.alpha[-1], 1 - a) for a in emi])
        # print('weights mod=',self.weights)

    def mod_begin(self, data, label):
        randomnum = random.randint(1, 100)
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=randomnum)
        i = 0
        rmse = 2 * self.max_rmse
        while (rmse > self.max_rmse):
            randomnum = random.randint(1, 10)
            randomnum2 = random.randint(1, 10)
            clfmodel=learn_machine('svr',np.shape(X_train)[1],np.shape(y_train)[1],i)
            self.clf_sets.append(clfmodel)
            clfmodel.fit(X_train, y_train, self.weights)
            y_pred = clfmodel.model.predict(X_train)
            # print('真实值与预测值之差',y_train[:,0]-aaa)
            print('单个模型预测', mean_squared_error(y_train, y_pred))
            self.mod_Em(i, X_train, y_train)
            self.mod_em(i, X_train, y_train)
            rmse = mean_squared_error(y_train, self.mod_G(X_train, y_train))
            print('第', i, '次adaboost，均方偏差为{1}', rmse)
            self.clfseri += 1
            i += 1
        self.clfseri -= 1
        # 保持clfseri为最新的模型序号,例如已经存在3个模型，那么clfseri=2

    def configure_mape(self, label, pred):
        # 用来计算mape误差，输出必须为[sample,mape]的二维数组
        size=np.shape(label)
        pred=np.array(pred).reshape(size)
        return abs(label - pred)

class learn_machine():
    #提供Adaboost所需的弱学习器
    def __init__(self, type='bp',input=5,output=1,id=1):
        self.name=type+str(np.random.random((1))[0])
        self.type=type
        self.id=id
        self.model=self.init_model(input,output)
    def init_model(self,input,output):
        if self.type=='bp':
            model = Sequential()  # 层次模型
            hidd=np.random.randint(output,3*input,1)[0]
            model.add(Dense(hidd, input_dim=input, init='uniform'))  # 输入层，Dense表示BP层
            model.add(Activation('relu'))  # 添加激活函数
            model.add(Dense(output, input_dim=hidd))  # 输出层
            model.compile(loss='mean_squared_error', optimizer='SGD')
        if self.type=='svr':
            randomnum = random.randint(1, 10)
            randomnum2 = random.randint(1, 10)
            c = randomnum * 200
            gamma = randomnum2 / 100

            model = SVR(kernel='rbf', C=c, gamma=gamma)

            #svr_rbf.fit(X_train,y_train,sample_weight=self.weights)
        return model
    def fit(self,x,y,sam_weight):
        self.model.fit(x,y,sample_weight=np.array(sam_weight))
        print('begin save')
        if self.type=='bp':
            # 对于不同的模型，保存方法可能不同，因此在此处增加判断语句
            self.model.save(self.name+'.h5')
        if self.type=='svr':
            print('svr save')
            joblib.dump(self.model, 'saveAdaboost/'+self.name+'.pkl')
        print('save over')
        return 1
    def load_model(self):
        if self.type=='bp':
            model=load_model(self.name+'.h5')
        if self.type=='svr':
            model=joblib.load('saveAdaboost/'+self.name+'.pkl')
        return model

    def predict(self,x):
        if self.type=='bp':
            model=load_model(self.name+'.h5')
        if self.type=='svr':
            model=load_model(self.name+'.pkl')
        return model.predict(x)

class modified_AdaBoost:
    # 有问题版本，勿使用
    def __init__(self, n_estimators=50, learning_rate=1.0,max_rmse=22):
        self.clf_num = n_estimators
        self.learning_rate = learning_rate
        self.max_rmse=max_rmse

    def init_args(self,M):
        # 弱分类器数目和集合
        self.clf_sets = []
        self.M=M
        # 初始化weights
        self.weights = [1.0 / self.M] * self.M
        self.Em=0
        self.emrate=[]
        # 误差率
        self.alpha=[]
        # G(x)系数 alpha
        self.alpha = []
        self.clfseri=0

    def mod_Em(self,i,data,label):
        # 计算某个模型最大误差
        clmodel=joblib.load('saveAdaboost/'+str(i)+'.pkl')
        self.Em=max(label[:,0]-clmodel.predict(data))
        return self.Em

    def mod_G(self,data,label):
        # 线性加权回归器
        a=np.zeros((np.shape(label)))
        print('a shape=',np.shape(a))
        weight_alpha=sum([math.log(1/a) for a in self.alpha])
        for i in range(self.clfseri+1):
            clfmodel=joblib.load('saveAdaboost/'+str(i)+'.pkl')
            label_temp=np.array(clfmodel.predict(data)).reshape(-1,1)
            print('label_temp shape=',np.shape(label_temp))
            temp=math.log(1/self.alpha[i])/weight_alpha*label_temp
            print('temp shape',np.shape(temp))
            a+=temp
        return a

    def mod_em(self,i,data,label):
        clfmodel=joblib.load('saveAdaboost/'+str(i)+'.pkl')
        temp=label[:, 0] - clfmodel.predict(data)
        temp=[abs(a) for a in temp]
        self.Em = max(temp)
        emi=pow(clfmodel.predict(data)-label[:,0],2)/pow(self.Em,2)
        #print('emi=',emi)
        self.emrate.append(sum(np.array(self.weights)*np.array(emi)))
        print('emrate=',self.emrate[-1])
        self.alpha.append(self.emrate[-1]/(1-self.emrate[-1]))
        print('alpha=',self.alpha[-1])
        ZM=sum(np.array(self.weights)*np.array([pow(self.alpha[-1],1-a) for a in emi]))
        print('ZM',ZM)
        self.weights=np.array(self.weights/ZM)*np.array([pow(self.alpha[-1],1-a) for a in emi])
        #print('weights mod=',self.weights)

    def mod_begin(self,data,label):
        randomnum=random.randint(1,100)
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=randomnum)
        i=0
        rmse=2*self.max_rmse
        while(rmse>self.max_rmse):
            randomnum = random.randint(1, 10)
            randomnum2 = random.randint(1, 10)
            c = randomnum * 200
            gamma = randomnum2 / 100

            svr_rbf = SVR(kernel='rbf', C=c, gamma=gamma)
            svr_rbf.fit(X_train,y_train,sample_weight=self.weights)
            aaa=svr_rbf.predict(X_train)
            #print('真实值与预测值之差',y_train[:,0]-aaa)
            print('单个模型预测',mean_squared_error(y_train[:,0],aaa))
            joblib.dump(svr_rbf, 'saveAdaboost/'+str(i)+'.pkl')
            self.mod_Em(i,X_train,y_train)
            self.mod_em(i,X_train,y_train)
            rmse=mean_squared_error(y_train,self.mod_G(X_train,y_train))
            print('第',i,'次adaboost，均方偏差为{1}',rmse)
            self.clfseri += 1
            i+=1
        self.clfseri-=1
    def configure_mape(self,label,pred):
        # 用来计算mape误差，输出必须为[sample,mape]的二维数组
        return abs(label-pred)
    def feature_weight(self,data,label,k):
        # k表示对于任意一个样本，选出k个在该样本上表现最好的学习器，并将这k个学习器权重置为1/k
        a=np.zeros((np.shape(label)))
        b=np.array([a for i in range(self.clfseri+1)])
        print('label shape=',np.shape(label))
        label_mod=np.array([label for i in range(self.clfseri+1)])
        for i in range(self.clfseri+1):
            clfmodel=joblib.load('saveAdaboost/'+str(i)+'.pkl')
            b[i,:]=clfmodel.predict(data).reshape(-1,1)
        mape=self.configure_mape(label_mod,b)
        feature_weight=np.zeros((len(data),self.clfseri+1))
        # 对于每一组数据，存在self.clfseri+1个权重
        for i in range(self.clfseri+1):
            mylist=mape[i,0]
            index_list = [i[0] for i in sorted(enumerate(mylist), key=lambda x: x[1])]
            feature_weight[i,index_list[:k]]=1/k
            #feature_weight[i,index_list[k]]=1-sum(feature_weight[i,:])
        # 接下来使用feature和feature_weight训练一个新模型，该模型的输出为多个，因此使用bp为佳
        input_dim=np.shape(data)[1]
        output_dim=np.shape(label)[1]
        model = Sequential()  # 层次模型
        model.add(Dense(12, input_dim=input_dim, init='uniform'))  # 输入层，Dense表示BP层
        model.add(Activation('relu'))  # 添加激活函数
        model.add(Dense(self.clfseri+1, input_dim=12))  # 输出层
        model.compile(loss='mean_squared_error', optimizer='SGD')
        model.fit(data, feature_weight, nb_epoch=10, batch_size=6)
        model.save('Feature_weight.h5')

    def feature_weight_mod_G(self,data,label):
        # 线性加权回归器
        a=np.zeros((np.shape(label)))
        #weight_alpha=sum([math.log(1/a) for a in self.alpha])
        feat_w_model=load_model('Feature_weight.h5')
        weights=feat_w_model.predict(data)

        for i in range(self.clfseri+1):
            clfmodel=joblib.load('saveAdaboost/'+str(i)+'.pkl')
            wei_temp=np.array(weights[:,i]).reshape(-1,1)
            label_temp=np.array(clfmodel.predict(data).reshape(-1,1))
            a+=wei_temp*label_temp
            print('a=',a)
            # 此处还需要修改，满足label多元情况
        return a
if __name__ == "__main__":
    bosten=load_boston()
    bostpd=pd.DataFrame(bosten.data,columns=bosten.feature_names).values
    bostlabel=pd.DataFrame(bosten.target,columns=['prices']).values
    #print(bostpd)
    boostmodel=AdaBoost(20,0.1)
    boostmodel.init_args(int(np.shape(bostpd)[0]*0.8))
    boostmodel.mod_begin(bostpd,bostlabel)
    pred_g=boostmodel.mod_G(bostpd,bostlabel)
    plt.plot(bostlabel,label='real',color='red')
    plt.plot(pred_g,label='normal predict',color='yellow')
    plt.show()

pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)