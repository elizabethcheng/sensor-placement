#!/usr/bin/python


#BEST_algorithms.py
import random as random
import numpy as np
import math
import scipy.linalg
import itertools


class Parsing(object):

	def __init__(self, filename):
		self.filename=filename

	def textParsing(self):
		data=dict()
		epochVec=dict()
		locations=dict()
		finalEpochVec=dict()
		comparison=dict()
		finalData=dict()
		parsedEpochFile=open('parsedEpochFile.txt', 'w')
		parsedMoteFile=open('parsedMoteFile.txt','w')

		#specify filename strings ('motelocs.txt')

		#for fyle in filenames:
		rawData=open(self.filename) 
		for line in rawData:
			lineAsList=line.split() #make list of individual strings in line
			#print lineAsList
			if self.filename=='data.txt':
				#date=lineAsList[0]
				#time=lineAsList[1]
				epoch=int(lineAsList[2]) #need int for indexing
				moteNum=int(lineAsList[3])-1 #need int for indexing
				moteReading=lineAsList[4]
				if moteNum not in data:
					data[moteNum]=[]
				if epoch not in epochVec:
					epochVec[epoch]=[]
				epochVec[epoch].append(moteReading+' ') #data from all sensors per epoch
				data[moteNum].append(moteReading+' ') #data from each mote, indexed by moteid
			if self.filename=='mote_locs.txt':
				moteNum=int(lineAsList[0].strip())-1
				moteX=float(lineAsList[1].strip())
				moteY=float(lineAsList[2].strip())
				moteXY=np.array([moteX,moteY])
				if moteNum not in locations:
					locations[moteNum]=[]
				locations[moteNum]=moteXY
		return (epochVec,data,locations)


	def distanceMatrix(self):	
		parsedText=self.textParsing()
		locations=parsedText[2]

		for i in range(54):
			tempDistVec=[]
			for j in range(54):
				squareNorm = (np.linalg.norm(locations[i]-locations[j]))**2
				tempDistVec.append(squareNorm)
			if i==0:
				dists=np.array([tempDistVec])
			else:
				dists=np.insert(dists,i,tempDistVec,axis=0)
		return dists/466.



#i'm going to stop working on getting the same data vector sizes for each mote to do "analysis" and 
#start working on finding the distance to create a cov. matrix.


'''
	for i in range(7000):#fix this WRITING!!!
		if len(epochVec[i])==20:
			epochVec[i]=str(epochVec[i])
			timePoints.append(i)
			finalEpochVec[i]=epochVec[i]
			parsedEpochFile.write(epochVec[i]+'\n')
	print timePoints

	for i in range(58):
		for j in timePoints:
			if i not in finalData:
				finalData[i]=[]
			print data[i][j]
			finalData[i].append(data[i][j])
		parsedMoteFile.write(finalData[i]+'\n')
'''


class Kernel(object):

    matrix= np.matrix(float)
    Lambda = 1

    def __init__(self, filename, Lambda = None):
    	self.filename=filename
        if Lambda is None:
            Lambda =1
        else:
            self.Lambda = Lambda
        parsed=Parsing(self.filename)
        self.squaredNorms=parsed.distanceMatrix()

    def compute_kernel_matrix(self):
        """Computes the kernel matrix between two given x vectors, using a gaussian RBF kernal"""
        #d = np.matrix(np.sum(np.abs(self.locs)**2,axis=1))
        #d2 = np.matrix(np.sum(np.abs(self.locs)**2,axis=1))
        #ones_d = np.ones_like(d)
        #ones_d2 = np.ones_like(d2)
        #X = np.matrix(self.locs)
        #X2 = np.matrix(self.locs)
        #sq_norms=d.transpose()*(ones_d) + ones_d.transpose()*d - 2*X*X.transpose()
        #sq_norms=d.transpose()*(ones_d) + ones_d2.transpose()*d2 - 2*X*X2.transpose()
        return np.exp(-self.squaredNorms/(self.Lambda**2))   # Using Gaussian RBF Kernal

    def get_kernel_partitions(self,partition_number,partition_indices,remainder=False):
        """Returns a set of partitions of the kernel matrix.
           Given a partition of indices P*, this function returns 4 matrices:
           K(P*,P*),K(P*,P),K(P,P*),K(P,P)
           where P represents the remaining partitions
        """
        #partion_indices = self.partion_indices  # copy the partions
        pi = list(partition_indices)
        target_inds = pi[partition_number]
        pi.pop(partition_number)
        # Now add the rest of the indices together
        remain_inds = list(itertools.chain(*pi))
        ix1 = np.ix_(target_inds,target_inds)
        ix2 = np.ix_(target_inds,remain_inds)
        ix3 = np.ix_(remain_inds,target_inds)
        ix4 = np.ix_(remain_inds,remain_inds)
        K = self.matrix
        return (K[ix1],K[ix2],K[ix3],K[ix4])

    def store_kernel_matrix(self,x,x2):
        self.matrix = self.compute_kernel_matrix(x,x2)
    
    def get_kernel_matrix(self):
        return self.matrix

    def set_Lambda(self,s):
        self.Lambda = s




class MI:

	def __init__(self,k, V):
		

	def maxMutualInformation(self,covMatrix):


'''
