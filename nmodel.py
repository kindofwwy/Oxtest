import numpy as np
import math,random,os,copy

class layer:

    def __init__(self,inwid,width,model=None,act=None,rand=None):
        self.inwid=inwid
        self.width=width
        if model is None:
            if rand is None:
                self.model=np.zeros([width,inwid+1])
            else:
                self.model=np.random.uniform(rand[0],rand[1],width*(inwid+1)).reshape([width,inwid+1])
        else:
            self.model=model

        a={"sigmoid":lambda x: 1/(1+math.exp(-x)),
           "relu":lambda x: max(0,x),
           "tanh":lambda x:(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x)),
           "None":lambda x:x}
        if act is None:
            self.act=a["sigmoid"]
        else:
            self.act=a[act]
        self.actname=act

    def go(self,inp):
        out=[]
        for i in self.model:
            outp=np.multiply(i[:-1],inp)
            outp=self.act(outp.sum()+i[-1])
            out.append(outp)
        return np.array(out)

    def flat(self):
        return self.model.flatten()

    def unflat(self,inp):
        self.model=inp.reshape(self.width,self.inwid+1)

class group:
    def __init__(self):
        self.biao=[]

    def append(self,layer):
        self.biao.append(layer)

    def allflat(self):
        a=np.array([])
        for i in self.biao:
            a=np.concatenate((a,i.flat()))
        return a

    def allunflat(self,l):      # 在biao里有layer的时候才能用
        for i in range(len(self.biao)):
            chang=(self.biao[i].inwid+1)*self.biao[i].width
            self.biao[i].unflat(l[:chang])
            l=l[chang:]


    def save(self,name):
        l=[]
        with open(name+".txt","w") as f:
            for i in self.biao:
                l.append(i.model)
                f.write(str(i.inwid)+" "+str(i.width)+" "+str(i.actname)+" "+"\n")
            np.save(name+".npy",l)

    def load(self,name):
        self.biao=[]
        s=np.load(name+".npy",allow_pickle=True)
        with open(name+".txt", "r") as f:
            lines=f.readlines()
            for i in range(len(s)):
                eline=lines[i].split(" ")
                a=layer(int(eline[0]),int(eline[1]),s[i],eline[2])
                self.append(a)
        return self.biao

    def go(self,inp):
        '''
        输入一个np.array，经过网络计算后输出一个np.array
        :param inp: np.array
        :return: np.array
        '''
        pass

    def go_in_group(self,inplist):
        outlist=[]
        for i in inplist:
            outlist.append(self.go(i))
        return np.array(outlist)

    @classmethod
    def getnewgroup(cls):
        '''
        获取新的group
        :return:group
        '''
        pass

    def errforall(self,out,ans):
        '''
        误差函数，输入神经网络的输出（多个），与其期待输出（多个），输出其总误差
        :param out: np.array([np.array(),...])
        :param ans: np.array([np.array(),...])
        :return: float
        '''
        pass

    def sample_err(self,samplelist):
        return self.errforall(self.go_in_group(trainsample.getquestion(samplelist)),
                               np.array(trainsample.getanswer(samplelist)))

    def desgrad(self,samplelist,steplength):
        origin=self.sample_err(samplelist)
        dylist=[]
        dx=0.01
        for eachlayer in self.biao:
            flatlayer=eachlayer.flat()
            for i in range(len(flatlayer)):
                flatlayer[i]+=dx
                eachlayer.unflat(flatlayer)

                new=self.sample_err(samplelist)
                dylist.append(-(new-origin)/dx*steplength)
                flatlayer[i]-=dx
                eachlayer.unflat(flatlayer)

        self.allunflat(self.allflat()+np.array(dylist))


class trainsample:

    samplelist=[]
    def __init__(self,question,answer):
        self.question=question
        self.answer=answer
        trainsample.samplelist.append(self)

    @classmethod
    def get_rand_sample(cls,amount):
        return random.sample(trainsample.samplelist,amount)

    @classmethod
    def getquestion(cls,samplelist):
        return [i.question for i in samplelist]

    @classmethod
    def getanswer(cls, samplelist):
        return [i.answer for i in samplelist]

class test(group):
    def __init__(self):
        super().__init__()

    @classmethod
    def getnewgroup(cls):
        g = test()
        g.biao = [layer(1, 5, act="tanh", rand=[-1, 1]),
                  layer(10, 5, act="tanh", rand=[-1, 1]),
                  layer(5, 1, act="tanh", rand=[-1, 1])]
        return g

    def go(self,inplist):
        last=np.array([0,0,0,0,0])
        for inp in inplist:
            a1=self.biao[0].go(inp)
            a1=np.concatenate((a1,last))
            b1=self.biao[1].go(a1)
            last=b1
        c1=self.biao[2].go(last)
        return c1

    def errforall(self,out,ans):
        return sum((out-ans)**2)[0]
if __name__ == "__main__":

    l = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [0, 1, 2, 3],
         [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [0, 1, 2, 3, 4],
         [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5],
         [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6],
         [1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7],
         [1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9],
         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7], [8, 7, 6], [7, 6, 5], [6, 5, 4], [5, 4, 3], [4, 3, 2], [3, 2, 1],
         [2, 1, 0], [9, 8, 7, 6], [8, 7, 6, 5], [7, 6, 5, 4], [6, 5, 4, 3], [5, 4, 3, 2], [4, 3, 2, 1], [3, 2, 1, 0],
         [9, 8, 7, 6, 5], [8, 7, 6, 5, 4], [7, 6, 5, 4, 3], [6, 5, 4, 3, 2], [5, 4, 3, 2, 1], [4, 3, 2, 1, 0],
         [9, 8, 7, 6, 5, 4], [8, 7, 6, 5, 4, 3], [7, 6, 5, 4, 3, 2], [6, 5, 4, 3, 2, 1], [5, 4, 3, 2, 1, 0],
         [9, 8, 7, 6, 5, 4, 3], [8, 7, 6, 5, 4, 3, 2], [7, 6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1, 0],
         [9, 8, 7, 6, 5, 4, 3, 2], [8, 7, 6, 5, 4, 3, 2, 1], [7, 6, 5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1],
         [8, 7, 6, 5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]

    for i in l:
        trainsample(list(np.array(i[:-1],dtype=float)/10),list(np.array([i[-1]],dtype=float)/10))

    a=test.getnewgroup()
    a.load("shushutest")
    for i in range(10000):
        a.desgrad(trainsample.samplelist,0.001)
        print(i,a.sample_err(trainsample.samplelist))
        a.save("shushutest")
    print("done")


#
#     def newg():
#         g=group()
#         g.biao = [layer(1, 5, act="tanh", rand=[-1, 1]),
#                   layer(10, 5, act="tanh", rand=[-1, 1]),
#                   layer(5, 1, act="tanh", rand=[-1, 1])]
#         return g
#
#     def go(g,inplist):
#         last=np.array([0,0,0,0,0])
#         for inp in inplist:
#             a1=g.biao[0].go(inp)
#             a1=np.concatenate((a1,last))
#             b1=g.biao[1].go(a1)
#             last=b1
#         c1=g.biao[2].go(last)
#         return c1
#     def err(out,tout):
#         return sum((out-tout)**2)
#
#     count=[0]
#     def tout():
#         # biao=[[i for i in range(10)],[i for i in reversed(range(10))]]
#         # biao=random.choice(biao)
#         # chuang=random.randint(3,len(biao))
#         # tiaoguo=random.randint(0,len(biao)-chuang)
#         # biao=biao[tiaoguo:tiaoguo+chuang]
#         l=[[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8], [4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8], [2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7], [8, 7, 6], [7, 6, 5], [6, 5, 4], [5, 4, 3], [4, 3, 2], [3, 2, 1], [2, 1, 0], [9, 8, 7, 6], [8, 7, 6, 5], [7, 6, 5, 4], [6, 5, 4, 3], [5, 4, 3, 2], [4, 3, 2, 1], [3, 2, 1, 0], [9, 8, 7, 6, 5], [8, 7, 6, 5, 4], [7, 6, 5, 4, 3], [6, 5, 4, 3, 2], [5, 4, 3, 2, 1], [4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4], [8, 7, 6, 5, 4, 3], [7, 6, 5, 4, 3, 2], [6, 5, 4, 3, 2, 1], [5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3], [8, 7, 6, 5, 4, 3, 2], [7, 6, 5, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3, 2], [8, 7, 6, 5, 4, 3, 2, 1], [7, 6, 5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1], [8, 7, 6, 5, 4, 3, 2, 1, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]]
#         biao=l[count[0]%len(l)]
#         count[0]+=1
#         biao=np.array(biao,dtype=float)
#         biao/=10
#
#         return biao
#     def errlist(g,l,toutl):
#         out=[]
#         for i in l:
#             out.append(go(g,i)[0])
#         out=np.array(out)
#         return err(out,toutl)
#     def randmix(r1, r2):
#         b = []
#         for i in range(len(r1)):
#             if random.randint(0,3000)==0:
#                 b.append(random.uniform(r1[i] - 1, r2[i] + 1))
#             else:
#                 if random.randint(0, 1):
#                     b.append(r1[i])
#                 else:
#                     b.append(r2[i])
#         return np.array(b)
#
#     def randlize(arr, a, b):
#         return arr + random.uniform(a, b)
#     def rev(name,indinum,genenum,setsize):
#         gbiao = []
#         if name+".npy" in os.listdir() and input("load this file?Y=yes")=="Y":
#
#
#             for i in range(indinum):
#                 g = group()
#                 g.load(name)
#                 g.allunflat(randlize(g.allflat(),-1,1))
#                 gbiao.append(g)
#             print(name,"loaded")
#
#         else:
#             for i in range(indinum):
#                 gbiao.append(group())
#                 gbiao[-1].biao=[layer(1,5,act="tanh",rand=[-1,1]),
#                                 layer(10,5,act="tanh",rand=[-1,1]),
#                                 layer(5,1,act="tanh",rand=[-1,1])]
#         for gene in range(genenum):
#             testbiao=[]
#             toutbiao=[]
#             for i in range(setsize):
#                 t=tout()
#                 testbiao.append(t[:-1])
#                 toutbiao.append(t[-1])
#
#             edic=dict()
#             for echg in gbiao:
#                 edic[errlist(echg,testbiao,toutbiao)]=echg
#             mi0=edic[min(edic)]
#             mi0.save(name)
#             print(gene, min(edic), f"平均 {min(edic) / setsize}")
#             del edic[min(edic)]
#             mi1=edic[min(edic)]
#
#             for i in range(indinum):
#                 gbiao[i].allunflat(randmix(mi0.allflat(),mi1.allflat()))
#
#     def grad(name,times,gap,setsize):
#         g=group()
#         if name + ".npy" in os.listdir() and input("load this file?Y=yes") == "Y":
#             g.load(name)
#         else:
#             g.biao=newg().biao
#         gf=g.allflat()
#         dx=0.01
#         for echt in range(times):
#             testbiao = []
#             toutbiao = []
#             for i in range(setsize):
#                 t = tout()
#                 testbiao.append(t[:-1])
#                 toutbiao.append(t[-1])
#
#             change=np.zeros(len(gf))
#             d1 = errlist(g, testbiao, toutbiao)
#             for i in range(len(gf)):
#
#                 gn=newg()               # new g
#                 gfn = gf.copy()          # new g flat
#                 gfn[i] += dx
#                 gn.allunflat(gfn)
#                 d2 = errlist(gn, testbiao, toutbiao)
#
#                 dy=(d2-d1)/dx
#                 change[i]=-dy*gap
#             gf+=change
#             g.allunflat(gf)
#             g.save(name)
#             test=errlist(g, testbiao, toutbiao)
#             print(echt,f"{test},平均{test/setsize}")
#
#
#
#     #rev("sintest-5",100,50,1000)
#     grad("gtest2",10000,0.001,72)
#     print("done")

    







