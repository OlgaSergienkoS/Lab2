from oneD import oneD
from twoD import  twoD

def mainOneD():
    d = oneD(3000, 100)
    d.showGaus()
    d.showF()

def mainTwoD():
    t = twoD(3000, 100)
    t.showGaus()
    t.showF()

if __name__ == '__main__':
    mainOneD()
    mainTwoD()