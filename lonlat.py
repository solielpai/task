#import chardet
#codelist=['utf-8','GB2312','gbk']
#with open('/home/paigu/download/tude.xls') as t:
#    text=t.readlines()
#    code=chardet.detect(text)['encoding']
#    if code in codelist:
#        te.insert(END,text.decode(code))
#    else:
#        print(chardet.detect(text))
#   
import pickle     
import xlrd  
import numpy as np
    #import chardet  
data = xlrd.open_workbook("//home/paigu/download/tude.xls")  
table = data.sheets()[0]  
nrows = table.nrows  
dict={}
a=0
value=[0,0]
for i in range(nrows):  
    
    #print chardet.detect(str(table.row_values(i)))  
    #print (str(table.row_values(i)).encode("utf-8").decode("utf-8"))
    key=str(table.row_values(i)[0]+table.row_values(i)[1]+table.row_values(i)[2])
    print(key)   #province city and county
    value[0]=float(table.row_values(i)[3])
    value[1]=float(table.row_values(i)[4])
    print(value)
    dict[key]=value[0:2]
   
   
    
lon=float(input('please input longitude :'))
lat=float(input('please input  latitude:'))
dist={}

def find_city(a,b):
    e=[10000,1]
    for key in dict.keys():
        dist[key]=np.sqrt((a-dict[key][0])*(a-dict[key][0])+(b-dict[key][1])*(b-dict[key][1]))
       
       #print(dist)
        if float(e[0])>dist[key]:
            e[0]=dist[key]
            e[1]=key
    print(a)
    return e
print('the driver is at:',find_city(lon,lat))