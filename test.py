from abtest import *
import random,numpy
a=nox.getnewgroup()
a.load("ox02")
# #a.save("ox023")
print(a.randevaluate(50,mode="mve",mctstime=50))
#oxtest.newgame("tvt",modelname="ox02")

#oxtest.newgame("mve",mctstime=50,modelname="ox02")




