
import numpy as np
import copy
import re
p = re.compile('\d+\.\d+')
label = re.compile('NEG|POS')
from scipy import stats


def read_feature(filename):
    readData = []
    N_STATES = 125
    
    f = open(filename,'r')
    label_1 = f.readline()
    label_list = label.findall(label_1)
    for line in f.readlines():
        number_backup = []
        
        a = line[:5]
        if a == "38433"or a =="38430"or a =="38907"or a =="41082"or a =="36693"or a =="41203"or a =="32419"or a =="33899"or a =="33655" or a=="38828":
            
            number = p.findall(line)
            
            for i in number:
                
                i = float(i)
                
                number_backup.append(i)
            readData.append(number_backup)
    
    readData = np.array(readData)
    readData = readData.T           

    f.close()
    #print(readData)
    #print(label_list)

    return readData,label_list
              
if __name__ == '__main__':
    read_feature("ALL3.txt")
        
