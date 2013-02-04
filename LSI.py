from __future__ import division

import invert
import math
import re

# 3rd party libraries. To find out installation steps, check README
import numpy
from scipy import linalg, mat, dot

numpy.set_printoptions(threshold='nan')

print "Start :: Traversing data"
invert.traverseDataSet()
print "End :: Traversing data"

tokens = invert.tokens
tokenDict = {}
x = raw_input("Enter stopping criteria: ")
y = int(float(x))

if y<0:
    print "Stopping criteria must be greater than 0 and less than 1"
    exit(0)
elif y:
    x=1
    print "As the stopping criteria mentioned by user is more than 1, taking stopping criteria as 1"

stopList = invert.stopList(x)

def totalTokens():
    count = 0
    for token in sorted(tokens.iterkeys()):
        if not (token in stopList):
            tokenDict[token] = count
            count = count + 1
    return count

tokenCount = totalTokens()
fileCount = 40

matrix = numpy.zeros(shape=(tokenCount,fileCount)) 
    
def calculateIDF(word):
    """ Calculate idf using the word """
    if word in tokens:
        tempDict = tokens[word]
        
        df = 0
        for fileName in tempDict:
            df = df + 1
            
    return math.log10(40/df)
    
def calculateTFIDF():
    for key in tokens:
        if not (key in stopList):
            row = int(tokenDict[key])
            idf = calculateIDF(key)
            tempDict = tokens[key]
            for key1 in tempDict:
                col = int(key1)
                tf = (1 + math.log10(len(tempDict[key1])))
                matrix[row][col] = tf*idf
         
def readFromFile(key):
    if(key<10):
        fileName = "0" + str(key)
    else:
        fileName =  str(key)
    fileContent = invert.readFile(fileName)
    fileContentList = fileContent.split(" ")
    
    for i in range(0,15):
        print fileContentList[i]+ " ", 
    
           
while(1):
    query = raw_input("\nEnter a search query : ")
    query = query.lower()
    if re.match("zzz",query):
        print "Found ZZZ. Exiting"
        exit(0)
    elif len(query)<1:
        print "No search query found. Exiting"
        exit(0)
    
    queryList = query.split(" ")

    kquery = raw_input("Enter rank :")
    try:
        k = int(kquery)
    except:
        print "Invalid rank"
        continue
    
    queryVector = numpy.zeros(shape=(tokenCount,1))         # (4875,1)
    
    flagStopList = 0
    print "Search query after taking in account the stoplist : ",
    for word in queryList:
        if word in tokenDict:
            print word,
            row = int(tokenDict[word])
            queryVector[row][0]=1
            flagStopList = 1
           
    if not (flagStopList):
        print "None"
        continue
    
    if k<1:
        print "\nRank should be greater than or equal to 1"
        continue
        
    print ""         
    calculateTFIDF()
    
    U, s, VT = linalg.svd(matrix)                           #(4875, 4875) (40,) (40, 40) = (4875, 40) 
    
    rows,cols = matrix.shape
    for index in xrange(k, cols):
        s[index]=0
    
    matrixk= dot(dot(U,linalg.diagsvd(s,len(matrix),len(VT))),VT)
    
    Uk, sk, VTk = linalg.svd(matrixk) 
    
    aligned_sk = linalg.diagsvd(sk,len(matrixk),len(VTk))   # (4875, 40)
    
    aligned_sk_inverse = linalg.pinv(aligned_sk)            # (40,4875)
    Uk_transpose = numpy.transpose(Uk)                      # (4875,4875)
    
    queryVectork1 = dot(dot(aligned_sk_inverse,Uk_transpose),queryVector)    #(40,1)
    
    queryVectork = dot(aligned_sk,queryVectork1)
    
    row,col = queryVectork.shape
    
    qVkSum = 0
    for i in range(0,row):
        qVkSum =  qVkSum + math.pow(queryVectork[i][0],2)
        
    qVkSum = math.sqrt(qVkSum)
    
    similarityMatrix = {}
    for i in range(0,len(VTk)):
        docMatrix1 = numpy.zeros(shape=(fileCount,1))
        docMatrix1 = VTk[:,i:i+1]                            # (40,1)
        docMatrix = dot(aligned_sk,docMatrix1)
        count = 0
        docSum = 0
        for j in range(0,40):
            count = count + (float(queryVectork[j][0]) * float(docMatrix[j][0]))
            docSum = docSum + math.pow(docMatrix[j][0],2)
        docSum = math.sqrt(docSum)
        similarityMatrix[i] = count/(docSum*qVkSum)
        
    count = 0
    for key in sorted(similarityMatrix, key=similarityMatrix.get, reverse=True):
        if count>4:
            break
        print "Document ",key
	print "Similarity score : ",similarityMatrix[key]
        readFromFile(key)
        print ""
        count = count + 1
