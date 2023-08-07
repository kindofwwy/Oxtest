import random,os
import numpy as np
from nmodel import *

def nmax(n,value):
    if n.value is None:
        return value
    return max(n.value,value) if n.ab==0 else min(n.value,value)

def lglist(value, previous, mode): #return cut or not
    if mode:
        for i in previous[0]:
            if value<=i:
                return True
        return False

    else:
        for i in previous[1]:
            if value >= i:
                return True
        return False


class node:
    def __init__(self):
        self.ab=None
        self.next=[]
        self.value=None
        self.previous=None
        self.times=0

    def bianli(self,deep=0,previous=None):
        evnode=self.evalnode(deep==0)
        if evnode is not None:
            self.value=evnode
            #print(f"va,deep{deep},value{evnode}")
            return evnode
        self.next=self.getnext()
        self.ab = deep % 2

        if self.previous is None:
            self.previous=previous

        for i in self.next:
            #print(f"deep{deep},value{self.value},branch{self.next.index(i)},in")
            self.value=nmax(self,i.bianli(deep+1,self))
            if self.cancut():
                #print(f"deep{deep},value{self.value},ab{self.ab},p{previous},cut")
                break

        #print(f"deep{deep},value{self.value},out")
        return self.value

    def cancut(self):
        p=self.previous
        while p is not None:
            if (p.value is not None) and ((self.ab==0 and p.ab==1 and self.value>p.value)or(self.ab==1 and p.ab==0 and self.value<p.value)):
                return True
            p=p.previous
        return False

    def mcts(self,times):
        self.next=[]
        for i in range(1,times+1):
            #print(f"maintime{i}")
            self.mctsingle(i)
            #print("\n")

    def mctsingle(self,maintimes,deep=0):
        self.times+=1
        self.value=0 if self.value is None else self.value

        if self.next == []:
            self.next=self.getnext()
        nononeflag=1
        for i in self.next:
            if i.value is None:
                self=i
                self.times+=1
                i.value=i.randsimulate()
                #print(f"sim deep{self.deep},value:{i.value}")
                nononeflag=0
                break
        if nononeflag and self.evalnode(True) is None:
            #print(f"in deep{self.deep}")
            return self.getbestnext(maintimes).mctsingle(maintimes,deep+1)  # 不返回值，只是作为出口
        value=self.value
        a = self.previous
        while a is not None and deep>0:
            value = -value
            a.value+=value
            #print(f"back deep{a.deep},value{value}")
            a=a.previous
            deep-=1


    def getbestnext(self,maintimes):
        '''

        :param maintimes:
        :return: 返回self.next中信心上限（或是其它的值）最高的节点
        '''
        #print(self.deep,"search")
        a=[]
        for i in self.next:
            a.append((i.ucb(maintimes),i))
        return max(a,key=lambda x:x[0])[1]

    def ucb(self,maintimes):
        #print(f"ucb,value{self.value},times{self.times},maintimes{maintimes},ucb{self.value+(2*np.log(maintimes)/self.times)**0.5}")
        return self.value+(2*np.log(maintimes)/self.times)**0.5

    def randsimulate(self):
        '''

        :return: 返回该节点的估值
        '''

    def evalnode(self,firsttime):
        # if firsttime:
        #     self.computerside=self.side
        '''

        :return: 如果是最末端的节点，返回value，如不是，返回None
        '''
        pass

    def getnext(self):
        '''

        :return: 返回一个列表，里面是下一个局面的所有可能(node)
        '''
        pass

class test(node):
    def __init__(self):
        super().__init__()

    def evalnode(self,firsttime):
        return None if random.choice([0,1]) else random.randint(-10,10)
    def getnext(self):
        biao=[]
        for i in range(random.randint(1,4)):
            biao.append(test())
        return biao

class oxtest(node):
    def __init__(self,board,side,chesspos=None,deep=0,computerside=None):
        super().__init__()
        self.board=board    #[['','',''],['','',''],['','','']]
        self.chesspos=chesspos
        self.side=side  #'o' or 'x'
        self.deep=deep
        self.computerside=computerside

    def findwin(self,side):
        if [side,side,side] in self.board:
            return True
        for i in range(3):
            if self.board[0][i]==side and self.board[1][i]==side and self.board[2][i]==side:
                return True
        if self.board[0][0]==side and self.board[1][1]==side and self.board[2][2]==side:
            return True
        if self.board[0][2]==side and self.board[1][1]==side and self.board[2][0]==side:
            return True
        return False

    def canchess(self,chesspos):
        if self.board[chesspos[0]][chesspos[1]]=="":
            return True
        return False

    def chess(self,chesspos):
        a=oxtest([self.board[0][:],self.board[1][:],self.board[2][:]],
                 'o' if self.side=='x' else 'x',chesspos[:],
                 deep=self.deep+1,
                 computerside=self.computerside)
        a.previous=self
        a.board[chesspos[0]][chesspos[1]]=self.side
        return a

    def randsimulate(self):
        a=oxtest.newgame("rvr",self.board,self.side,look=False,modelname="ox02")
        a.computerside="o" if self.side=="x" else "x"
        #print("look",self.side,a.evalnode(False))
        #a.look()
        return a.evalnode(False)

    def evalnode(self,firsttime):
        if firsttime:
            self.computerside = self.side

        if self.findwin("o"):
            if self.computerside=="o":
                return 1
            else:
                return -1
        elif self.findwin("x"):
            if self.computerside=="x":
                return 1
            else:
                return -1
        for i in self.board:
            if '' in i:
                return None
        return 0

    def someonewin(self):
        '''

        :return: 1==someonewin 0==draw None==mid
        '''
        if self.findwin("o") or self.findwin("x"):
            return 1
        else:
            for i in self.board:
                if '' in i:
                    return None
        return 0


    def getnext(self):
        biao=[]
        for i in range(3):
            for j in range(3):
                if self.canchess([i,j]):
                    biao.append(self.chess([i,j]))

        return biao

    def run(self):
        if self.board==[['','',''],['','',''],['','','']]:  # save time
            return self.chess([0,0])
        #self.next=[]
        a=self.bianli()
        #print(f"predict{a},side{self.computerside},{[i.value for i in self.next]}")
        for i in self.next:
            if i.value==a:
                b=self.chess(i.chesspos)
                return b
        print("err")

    def mctsrun(self,time):
        self.mcts(time)
        #print([(i.times,i.value) for i in self.next])
        return self.chess(max(self.next,key=lambda x:x.times).chesspos)

    def human(self):
        while True:
            pos=eval(input(":"))
            if self.canchess(pos):
                break
            print("invalid")
        return self.chess(pos)

    def randomchess(self):
        chesslist=[]
        for i in range(3):
            for j in range(3):
                if self.canchess([i,j]):
                    chesslist.append([i,j])
        return self.chess(random.choice(chesslist))

    @classmethod
    def multi(cls,oxtestb,who,
              modelname=None,model=None,
              modelbname=None,modelb=None,
              mctstime=100):   # 优先读取model，然后是modelname
        if who=="p":
            return oxtestb.human()
        elif who=="e":
            return oxtestb.run()
        elif who=="a":
            if model is None:
                model=nox.getnewgroup()
                model.load(modelname)
            return model.chessonboard(oxtestb)
        elif who=="b":
            if modelb is None:
                modelb=nox.getnewgroup()
                modelb.load(modelbname)
            return modelb.chessonboard(oxtestb)
        elif who=="r":
            return oxtestb.randomchess()
        elif who=="t":
            if model is None:
                model=nox.getnewgroup()
                model.load(modelname)
            return model.testchessonborad(oxtestb)
        elif who=="u":
            if modelb is None:
                modelb=nox.getnewgroup()
                modelb.load(modelbname)
            return modelb.testchessonborad(oxtestb)
        elif who=="m":
            return oxtestb.mctsrun(mctstime)

    @classmethod
    def newgame(cls,mode,board=None,hand="o",look=True,
                modelname=None,model=None,
                modelbname=None,modelb=None,
                mctstime=100):
        if board is None:
            board=[['','',''],['','',''],['','','']]
        a=oxtest(board,hand)
        a.look() if look else None
        first=mode[0]
        last=mode[-1]
        while a.someonewin() is None:
            if a.side==hand:
                a=oxtest.multi(a,first,modelname,model,modelbname,modelb,mctstime)
            else:
                a=oxtest.multi(a,last,modelname,model,modelbname,modelb,mctstime)
            a.look() if look else None
        print("end") if look else None
        return a

    def look(self):
        print(self.board)
        for i in self.board:
            for j in i:
                print((j if j != "" else " ") + "   ",end="")
            print("\n")

    #for deeplearn,"o"=1, "x"=-1, ""=0

    def flat(self):
        return self.board[0]+self.board[1]+self.board[2]
    def chessonarray(self, chesspos):
        bo=[]
        ko=[]
        f=self.flat()
        for i in range(len(f)):
            if f[i]=="o":
                bo.append(1)
            elif f[i]=="x":
                bo.append(-1)
            else:
                bo.append(0)
            if chesspos[0]*3+chesspos[1]==i:
                ko.append(1 if self.side=="o" else -1)
            else:
                ko.append(0)
        return np.array(bo+ko)

class nox(group):
    def __init__(self):
        super().__init__()

    @classmethod
    def getnewgroup(cls):
        g=nox()
        g.biao=[layer(18,20,rand=[-1,1], act="tanh"),
                layer(20,20,rand=[-1,1], act="tanh"),
                #layer(20, 20, rand=[-1, 1], act="tanh"),
                layer(20,1,rand=[-1,1], act="tanh")]
        return g

    def go(self,inp):
        a=self.biao[0].go(inp)
        for i in self.biao[1:]:
            a=i.go(a)
        return a

    def errforall(self,out,ans):
        #return sum(-ans*np.log(out))[0]
        #print(f"out{out},ans{ans},err{sum((out - ans) ** 2)}")
        return sum((out - ans) ** 2)[0]


    def chessonboard(self,oxtestborad):
        choose=dict()
        for i in range(3):
            for j in range(3):
                if oxtestborad.canchess([i,j]):
                    choose[self.go(oxtestborad.chessonarray([i,j]))[0]]=[i,j]
        #print(choose)
        return oxtestborad.chess(choose[max(choose)])

    def testchessonborad(self,oxtestborad):
        choose=[]
        poses=[]
        for i in range(3):
            for j in range(3):
                if oxtestborad.canchess([i, j]):
                    choose.append(self.go(oxtestborad.chessonarray([i, j]))[0])
                    poses.append([i, j])

        sumup=sum([(i+1)/2 for i in choose])
        for i in range(len(choose)):
            choose[i]=(choose[i]+1)/2/sumup
        #print(choose)
        return oxtestborad.chess(random.choices(poses,weights=choose)[0])

    def getsample(self):
        #trainsample.samplelist=[]
        mode=random.choice(["tvt"])
        a=oxtest.newgame(mode,look=False,model=self)

        end=a.someonewin() # 0==draw,1==someonewin
        endside=a.previous.side
        getv=lambda x:  0 if end==0 else \
                        1 if endside==x.previous.side else -1

        while a.previous is not None:
            #trainsample.samplelist=[]
            trainsample(np.array(a.previous.chessonarray(a.chesspos),dtype="float"),
                        np.array([getv(a)],dtype="float"))
            # print(f"go{self.go(a.previous.chessonarray(a.chesspos))},anwser{trainsample.samplelist[-1].answer},err{self.sample_err([trainsample.samplelist[-1]])}")
            # a.look()
            a=a.previous

    def getsamples(self,amount):
        trainsample.samplelist = []
        for i in range(amount):
            self.getsample()

    def randevaluate(self,halfamount,mode="rva",modelbname=None,modelb=None,mctstime=100):   # mode中第一位是统计位，最后一位是对手
        win = {0: 0, 1: 0, -1: 0}
        for i in range(halfamount):
            a = oxtest.newgame(mode, model=self,look=False,modelbname=modelbname,modelb=modelb,mctstime=mctstime)
            a.computerside = "o"
            win[a.evalnode(False)] += 1
        for i in range(halfamount):
            a = oxtest.newgame("".join([i for i in reversed(mode)]), model=self,look=False,modelbname=modelbname,modelb=modelb,mctstime=mctstime)
            a.computerside = "x"
            win[a.evalnode(False)] += 1
        return win



if __name__ == "__main__":
    modelname="oxrand"
    a=nox.getnewgroup()
    if modelname+".txt" not in os.listdir():
        input("new")
        a.save(modelname)
    a.load(modelname)
    count=0
    maincount=0
    while count<500:
        a.load(modelname)
        a.getsamples(5)
        for j in range(8):
            a.desgrad(trainsample.samplelist,0.001)
            print(maincount,j,a.sample_err(trainsample.samplelist))

        eva=a.randevaluate(200,mode="tvu",modelbname=modelname)
        print(eva,eva[1]/(eva[1]+eva[-1]),eva[1]/sum([eva[k] for k in eva]))
        if eva[1]/(eva[1]+eva[-1])>0.51:
            a.save(modelname)
            print("saved",count)
            count+=1
        maincount+=1
    print("done")



