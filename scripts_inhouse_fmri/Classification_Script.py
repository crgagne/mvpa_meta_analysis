# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 11:27:27 2014

@author: cgagne
"""



# deprecated
#import os
#os.system(")



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("server",help="if on ccv press 1")
parser.add_argument("subject", help="enter a number like 01")
parser.add_argument("analysis_number", help="which analysis number 1-8",type=int)
parser.add_argument("roi_number", help="analysis_number",type=int)
parser.add_argument("stim",help="rule,resp,type")
parser.add_argument("onset",help="+0,+1,+2,+3")
parser.add_argument("fs",type=float)
args=parser.parse_args()

print "server"+args.server
print "subject "+args.subject
print "analysis number "+str(args.analysis_number)
print "roi number "+str(args.roi_number)
print "stim "+args.stim
print "onset "+args.onset
print "fs "+str(args.fs)

server = args.server
subject = args.subject
analysis_number = args.analysis_number
roi_number = args.roi_number
stim = args.stim
onset = args.onset
fs = args.fs

import sys
#sys.path.append("/home/cgagne/Dropbox/python/PyMVPA-master")
#sys.path.append("/gpfs/home/cgagne/python/PyMVPA-master")
#sys.path.append("/gpfs/home/cgagne/python/PyMVPA-master/mvpa2/lib/python2.6/site-packages")
sys.path.append("/home/bishop/cgagne/MVPA_Analysis/CCV_Files/PyMVPA/mvpa2")
#import mdp # so PCAmapper can work
rois = ['Occipital_Lobe','Parietal_Lobe','Temporal_Lobe','Frontal_Lobe','Medial_Frontal','Lateral_Frontal']
from mvpa2.tutorial_suite import *
import numpy as np
import pdb

#server = 'ccv'
#subject = '01'
#roi_number = 0
#time = 'rule'
#onset = '+4'
#fs = 500
#stim = 'rule'


basefolder='/home/bishop/cgagne/MVPA_Analysis/CCV_Files'
subjectfolder = basefolder+"/data_mri/fmri_"+subject+"_TRI/bold/"
subjectbehfolder = basefolder+"/data_behavioral/fmri_"+subject+"_TRI/tribehav/"
roifolder = basefolder+"/data_rois/"
resultsfolder = basefolder+"/results/"


attr = SampleAttributes(os.path.join(subjectbehfolder,'TR'+stim+'onsets'+onset+'.txt')) # order 2 has exact timings...
np.unique(attr.chunks)
len(attr.chunks)
np.unique(attr.targets)

# Get data
print("Loading Data...")
ds = fmri_dataset(os.path.join(subjectfolder,'wraf.nii.gz'),targets=attr.targets,chunks=attr.chunks,mask=os.path.join(roifolder,rois[roi_number]+'.nii.gz')) # can add mask here. 
ds.sa.time_indices
ds.fa.voxel_indices
ds.shape
print ds.a.mapper
#my time_coords  are 0.

#null distribution plot
def make_null_dist_plot(dist_samples,empirical):
    pl.hist(dist_samples,bins=20,normed=True,alpha=0.8)
    pl.axvline(empirical,color='red')
    pl.axvline(0.5,color='black',ls='--')
    pl.xlim(0,1)
    pl.xlabel('Average cross-validated classification error')



# Nested Cross Validation
def select_best_clf(dataset_,clfs):
    "select best model according to CVTE, used in nested cross validtion"
    best_acc = None
    for clf in clfs:
        cv = CrossValidation(clf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t))
        try:
            accuracy = np.mean(cv(dataset_))
        except LearnerError, e:
            continue
        if best_acc is None or accuracy > best_acc:
            best_clf = clf
            best_acc = accuracy
        verbose(4,"Classifier %s cv error=%.2f" % (clf.descr,accuracy))
    return best_clf,best_acc

def explainable_variance(ds):
    m=len(np.unique(ds.sa.targets))
    print m
    mtypes = np.unique(ds.sa.targets)
    n=len(ds.sa.targets)/m
    print n
    size = ds.shape
    EV = np.empty([size[1],1]) 
    F =  np.empty([size[1],1]) 
    MSbetstore = np.empty([size[1],1])
    MSwitstore = np.empty([size[1],1])
    for voxel in range(size[1]):
        voxeldata = ds[:,voxel]
        globalmean = np.mean(voxeldata)  
        sb = np.empty([m,1])
        sww=np.empty([m,1])
        for idx, targettype in enumerate(mtypes):
            targetdata = voxeldata[ds.sa.targets==[targettype]]
            targetmean = np.mean(targetdata)
            targetstd = np.std(targetdata)
            sb[idx]=n*(math.pow((targetmean-globalmean),2))
            sww[idx]=(n-1)*math.pow(targetstd,2)
        MSbet=np.sum(sb)/(m-1)
        MSwit=np.sum(sww)/(len(ds.sa.targets)-m) # divide by big N - number of groups
        F[voxel] = MSbet/MSwit
        Ftemp = MSbet/MSwit
        EV[voxel] = 1-1/Ftemp
    return F,EV


# WHICH ANALYSIS


print("Starting Analysis")
noskip = 1
if analysis_number==1:
    
    # show data before and after preprocessing
    ds.samples = ds.samples.astype('float')
    pl.figure()
    pl.subplot(121)
    plot_samples_distance(ds,sortbyattr='chunks')
    pl.title('Sample Distance (sorted by runs)')
    pl.subplot(122)
    plot_samples_distance(ds,sortbyattr='targets')
    pl.title('Sample Distance (sorted by targets)')
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    pl.figure()
    pl.subplot(121)
    plot_samples_distance(ds,sortbyattr='chunks')
    pl.title('Distances: zscored,detrended (sorted by chunks)')
    pl.subplot(122)
    plot_samples_distance(ds,sortbyattr='targets')
    pl.title('Distances: zscored,detrended (sorted by targets)')
elif analysis_number==2:
    
    # SVM Classification   
    analysisfolder = 'Knn20_Corr_kfold'
    print analysisfolder 
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #detrend by run. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    
    #add in a different attribute #
    ds.sa['trials'] = range(ds.shape[0])
    
    baseclf = LinearCSVMC()
    baseclf = kNN(k=20,dfx=one_minus_correlation)
    #fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FractionTailSelector(fs,mode='select',tail='upper'))
    #fselclf = FeatureSelectionClassifier(baseclf,fsel)  
          
    cvte = CrossValidation(baseclf,NFoldPartitioner(attr='trials'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    #confusion matrix
    print cvte.ca.stats.as_string(description=True)
    print cvte.ca.stats.matrix # reduced form
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    out = mat #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'fs'+str(fs)+'out_ACC.txt',out,fmt='%.8f',delimiter='\t')     
   
    #EV2

elif analysis_number==3:
    
    # SVM Classification with feature selection  
    analysisfolder = 'fs_and_onsets'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    baseclf = LinearCSVMC() 
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    #confusion matrix
    print cvte.ca.stats.as_string(description=True)
    print cvte.ca.stats.matrix # reduced form
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'out.txt',mat,fmt='%.3f',delimiter='\t')





elif analysis_number==4:
    
    # SVM Classification  with permutation   
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    analysisfolder = 'permutations'
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    baseclf = LinearCSVMC()
    #permutation # Actually shouldnt do it on test and train data... so need to add in here.. 
    perm = 1000
    permutator = AttributePermutator('targets',count=perm)
    distr_est = MCNullDist(permutator,tail='left',enable_ca=['dist_samples'])
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    #cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);    
    cv_mc = CrossValidation(fselclf,NFoldPartitioner(),postproc=mean_sample(),null_dist=distr_est,enable_ca=['stats'])
    cv_mc_results = cv_mc(ds)
    p = cv_mc.ca.null_prob
    #_=pl.figure()  # underscore=   who knows why?
    #make_null_dist_plot(np.ravel(cv_mc.null_dist.ca.dist_samples),np.asscalar(cv_mc_results))
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    
    mat = np.empty([perm+1,1])
    mat = np.transpose(mat)
    mat[0,0] = np.asscalar(cv_mc_results)
    mat[0,1:] = (cv_mc.null_dist.ca.dist_samples)
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'out.txt',mat,fmt='%.3f',delimiter='\t')


elif analysis_number==5:
    
    # SVM Classification with Meta Param Fitting 
    analysisfolder = 'SVM_nested3'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    clfgroup=[]
    #params = [3,-8,-7,-6,-5,-4,-3,-9,-10,-11,-2,-1,1,2,4,5,6,7]
    params = [-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5]    
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    for r in params:  # make the classifier list #
        baseclf = LinearCSVMC(C=math.pow(10,r))
        baseclf = FeatureSelectionClassifier(baseclf,fsel) # probably dont need fsel if listen to thomas.
        clfgroup.append(baseclf)
    best_clfs = {}
    
    alt_clf = RbfCSVMC()
    confusion = ConfusionMatrix()
    verbose(1, "Estimating error using nested CV for model selection")
    partitioner = NFoldPartitioner()
    splitter = Splitter('partitions')
    cross_acc = []
    best_list = []
    cross_acc2 = []
    s = 0
    for isplit, partitions in enumerate(partitioner.generate(ds)):
        # partitions becomes a dataset with partitions.sa.partitions. (11111111,222) for that cross valid
        verbose(2, "Processing split #%i" % isplit)
        print s
        dstrain, dstest = list(splitter.generate(partitions))
        best_clf, best_error = select_best_clf(dstrain, clfgroup) # feed it dstrain which has 1-14 runs excluding 15. It cross validates and selects best param)
        print best_error
        best_list.append(best_clf.summary)
        best_clfs[best_clf.descr] = best_clfs.get(best_clf.descr, 0) + 1
        best_clf.train(dstrain)
        pred = best_clf.predict(dstest)
        acc = np.mean(pred==dstest.sa.targets)
        cross_acc2.append(acc)        
        tm = TransferMeasure(best_clf, splitter,postproc=BinaryFxNode(mean_mismatch_error,space='targets'),enable_ca=['stats'])
        result = tm(partitions) # probably need to do equals...
        #print result.ca
        confusion += tm.ca.stats
        print confusion   
        cross_acc.append(confusion.stats['mean(ACC)'])
        s = s + 1
    #cross_acc.append(np.mean(cross_acc))    
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    #mat = np.empty([len(cv_results)+1,1])
    #mat[0] = np.mean(cv_results)
    #mat[1:] = cv_results.samples
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+'ACC_out.txt',cross_acc,fmt='%.3f',delimiter='\t')
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+'ACC2_out.txt',cross_acc,fmt='%.3f',delimiter='\t')
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+'Param_out.txt',best_list,fmt='%s',delimiter='\t')


elif analysis_number==6:
    
    # PCA 
    numPCA = 2; 
    analysisfolder = 'PCA'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    print len(ds)
    baseclf = LinearCSVMC() # make a classifier class with normal svm
    ssel = StaticFeatureSelection(slice(None,2))
    #baseclf = FeatureSelectionClassifier(baseclf,ssel)
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    #mapper = ChainMapper([StaticFeatureSelection(slice(None,len(ds))),fsel,SVDMapper()])
    # mapper = ChainMapper([StaticFeatureSelection(slice(None,ds.shape[1])),fsel,PCAMapper(reduce=True),StaticFeatureSelection(slice(None,2))])
    mapper = ChainMapper([StaticFeatureSelection(slice(None,ds.shape[1])),fsel,SVDMapper(),StaticFeatureSelection(slice(None,numPCA))])
    
    
    #mapper = ChainMapper([PCAMapper(reduce=True),StaticFeatureSelection(slice(None,numPCA))]) # perform singular value decomposition then select first 2 dimensions.
    print mapper
    metaclf = MappedClassifier(baseclf,mapper) # makes a more complex classifier using SVM on data that has gone through mapping... 
    sensitivities = []
    #indexes = []    
    def store_me(data,node,result): # node here is transfer measure... could get traiing error? 
        sens = node.measure.get_sensitivity_analyzer(force_train=False)(data)
        #index = node.measure.mapper.slicearg
        sensitivities.append(sens)
        #indexes.append(index)
    
    cvte = CrossValidation(metaclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats'],callback=store_me);
    cv_results = cvte(ds)
    sen = sensitivities[0].samples    
    for c in range(len(sensitivities)-1):
        sent = sensitivities[c+1].samples # extract numpy array
        sen = np.concatenate((sen,sent),axis=0)
    sen1= sen
    sen = sen<>0
    numb_runs_used = np.sum(sen,0)
    limiteddata = ds[:,sen[0,:]]
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    import scipy.io as io
    #np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'fs_voxel.txt',limiteddata.samples,fmt='%.5f',delimiter='\t')
    io.savemat(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+str(numPCA)+'fs_voxel.mat',dict(d=limiteddata.samples))
    io.savemat(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+str(numPCA)+'fs_svm.mat',dict(svm=sen1[0,sen[0,:]]))
    io.savemat(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+str(numPCA)+'fs_svm_bias.mat',dict(bias=sensitivities[0].sa.biases))
    a = np.empty([len(ds.sa.targets),1])    
    a = ds.sa.targets #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    out = a.astype(np.float)    
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'out_targets.txt',out,fmt='%.5f',delimiter='\t')  
    np.mean(cv_results)
    print ds.shape
    #confusion matrix
    print cvte.ca.stats.as_string(description=True)
    print cvte.ca.stats.matrix # reduced form
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+'out'+str(numPCA)+'.txt',mat,fmt='%.3f',delimiter='\t')
    print ds.sa.chunks
    
elif analysis_number==7:    

    # Event Related
    analysisfolder = 'EventModel'
    events = find_events(targets=ds.sa.targets,chunks=ds.sa.chunks) #finds where target value changes..
    print len(events)
    TR = 1 # time coords didnt work.. so manually enter in.. 
    for ev in events:
        ev['onset'] = (ev['onset'] * TR) #converts to seconds
        ev['duration'] =1 #correct the duration.. because find_events didnt work that well...  
    events = [ev for ev in events if ev['targets'] in ['1','2']] # remove rest from events list
    ds = eventrelated_dataset(ds,events)
    zscore(ds,chunks_attr=None)
    baseclf = LinearCSVMC()
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);   
    cv_results = cvte(ds)
    print np.mean(cv_results)
    #confusion matrix
    print cvte.ca.stats.as_string(description=True)
    print cvte.ca.stats.matrix # reduced form
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'out.txt',mat,fmt='%.3f',delimiter='\t')

    

elif analysis_number==8:  

    analysisfolder = 'fs_stability'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    baseclf = LinearCSVMC() 
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    #cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    #cv_results = cvte(ds)
    #sensitivity analysis
    #sensana = fselclf.get_sensitivity_analyzer(postproc=maxofabs_sample())
    #type(sensana)
    #cv_sensana = RepeatedMeasure(sensana,NFoldPartitioner())
    #sens = cv_sensana(ds) # repeats the feature selection... 
    
    # that didnt work
    # BUT IT DID GIVE DIFFERENT SENSITIVITIES when i did each chunk individually
    sensitivities = []
    indexes = []
    def store_me(data,node,result):
        sens = node.measure.get_sensitivity_analyzer(force_train=False)(data)
        index = node.measure.mapper.slicearg
        sensitivities.append(sens)
        indexes.append(index)
        
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats'],callback=store_me);
    cv_results = cvte(ds)
    
    sen = sensitivities[0].samples    
    for c in range(len(sensitivities)-1):
        sent = sensitivities[c+1].samples # extract numpy array
        sen = np.concatenate((sen,sent),axis=0)
    sen1= sen
    sen = sen<>0
    numb_runs_used = np.sum(sen,0)
    match = np.sum(sen[0,indexes[0]])
    
    # I believe this is getting the SVM values... 
    
    #sens_comb = sens.get_mapped(maxofabs_sample())
    #type(sens)
    #print sens.shape # if there are more than 1 binary classification, like 8-way.. will get different number of subproblems maps.. from SVM
    #len(sens_comb.samples[sens_comb.samples>0])
    #overlap from each partition.. 
    #ov = MapOverlap()
    #overlap_fraction = ov(sens.samples>0)
    #print overlap_fraction
    #back to nifti.
    #nimg = map2nifti(ds,sens)
    #nimg.to_filename('sensitivity.nii.gz')
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    mat = numb_runs_used
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'fs_voxel.txt',mat,fmt='%.10f',delimiter='\t')
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'fs_voxel15.txt',sen1,fmt='%.10f',delimiter='\t')
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'fs_index.txt',indexes,fmt='%.10f',delimiter='\t')
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'fs_match.txt',match,fmt='%.10f',delimiter='\t')

   # np.savetxt(resultsfolder+analysisfolder+'/test.txt',sen1,fmt='%.10f',delimiter='\t')
    


elif analysis_number==9:
    
    ### DO AVERAGING... 
    analysisfolder = 'Averaged'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    run_averager = mean_group_sample(['targets','chunks'])
    ds = ds.get_mapped(run_averager)
    baseclf = LinearCSVMC() 
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    #confusion matrix
    print cvte.ca.stats.as_string(description=True)
    print cvte.ca.stats.matrix # reduced form
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    perm = 1000
    permutator = AttributePermutator('targets',count=perm)
    distr_est = MCNullDist(permutator,tail='left',enable_ca=['dist_samples']) 
    #cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);    
    cv_mc = CrossValidation(fselclf,NFoldPartitioner(),postproc=mean_sample(),null_dist=distr_est,enable_ca=['stats'])
    cv_mc_results = cv_mc(ds)
    p = cv_mc.ca.null_prob
    #_=pl.figure()  # underscore=   who knows why?
    #make_null_dist_plot(np.ravel(cv_mc.null_dist.ca.dist_samples),np.asscalar(cv_mc_results))
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    
    mat = np.empty([perm+1,1])
    mat = np.transpose(mat)
    mat[0,0] = np.asscalar(cv_mc_results)
    mat[0,1:] = (cv_mc.null_dist.ca.dist_samples)
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'out.txt',mat,fmt='%.3f',delimiter='\t')



elif analysis_number==10:
    # Explainable Variance
    
    analysisfolder = 'Explainable_Variance_RecalcLast1000'
    
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    dsorig = ds
    baseclf = LinearCSVMC()
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    
#    #total explainable Variance
#    EV2 = np.empty([len(dsorig.samples[0]),1]) 
#    Fbet2 = np.empty([len(dsorig.samples[0]),1])
#    Fwit2 = np.empty([len(dsorig.samples[0]),1])
#    m=len(np.unique(dsorig.sa.targets))
#    mtypes = np.unique(dsorig.sa.targets)
#    n=len(ds.sa.targets)/m
#
#   
#    
#    for voxel in range(len(dsorig.samples[0])):
#        voxeldata = ds.samples[:,voxel]
#                
#        
#        globalmean = np.mean(voxeldata)
#        # Classic Anova way (ignoring correlated variance) - F ratio
#        t=0    
#        sb = np.empty([len(np.unique(dsorig.sa.targets)),1])
#        sww=np.empty([len(np.unique(dsorig.sa.targets)),1])
#        for targettype in np.unique(dsorig.sa.targets):
#            targetdata = voxeldata[dsorig.sa.targets==[targettype]]
#            targetmean = np.mean(targetdata)
#            sb[t]=len(targetdata)*math.pow((targetmean-globalmean),2)
#            sw=np.empty([len(targetdata),1])
#            for targetnumber in range(len(targetdata)):
#                sw[targetnumber] = math.pow((targetdata[targetnumber]-targetmean),2)
#            sww[t] = np.sum(sw)
#            t=t+1
#        Fbet2[voxel]=np.sum(sb)/(m-1)
#        Fwit2[voxel]=np.sum(sww)/(n-m)
#        EV2[voxel] = 1-(1/(Fbet2[voxel]/Fwit2[voxel]))    
#    
    
    
    # get sensitive voxels    
    sensana = baseclf.get_sensitivity_analyzer(postproc=maxofabs_sample())
    cv_sensana = RepeatedMeasure(sensana,NFoldPartitioner())
    sens = cv_sensana(ds)
    sensmean = np.mean(sens,0) # mean across cross validation
    np.max(sensmean)
    len(sensmean[sensmean>0])
    sensmeansort = np.sort(sensmean)
    sensmeansort = sensmeansort[::-1]
    cutoff = sensmeansort[1000]
    
    ds.samples = ds.samples[:,sensmean>cutoff]
    print ds.samples.shape
#    EV2 = np.transpose(EV2)
#    #cor = np.corrcoef(EV2[~np.isnan(EV2)],sensmean[~np.isnan(EV2[:,0])])    
#    
#    
#    
#    #
#    EVvoxel = np.empty([len(ds.samples[0]),1])
#    EV = np.empty([len(ds.samples[0]),1])
#    MSbet = np.empty([len(ds.samples[0]),1])
#    Fbet = np.empty([len(ds.samples[0]),1])
#    Fwit = np.empty([len(ds.samples[0]),1])
#    MSbetalt = np.empty([len(ds.samples[0]),1])
#    MSwit = np.empty([len(ds.samples[0]),1])
#    m=len(np.unique(ds.sa.targets))
#    mtypes = np.unique(ds.sa.targets)
#    n=len(ds.sa.targets)/m
#    
#    voxeldataout = np.empty([len(ds.samples[0]), len(ds.sa.targets)])
#    
#    for voxel in range(len(ds.samples[0])):
#        voxeldata = ds.samples[:,voxel]
#        voxeldataout[voxel,:] = voxeldata
#        globalmean = np.mean(voxeldata)
#        # Classic Anova way (ignoring correlated variance) - F ratio
#        t=0    
#        sb = np.empty([len(np.unique(ds.sa.targets)),1])
#        sww=np.empty([len(np.unique(ds.sa.targets)),1])
#        for targettype in np.unique(ds.sa.targets):
#            targetdata = voxeldata[ds.sa.targets==[targettype]]
#            targetmean = np.mean(targetdata)
#            sb[t]=len(targetdata)*math.pow((targetmean-globalmean),2)
#            sw=np.empty([len(targetdata),1])
#            for targetnumber in range(len(targetdata)):
#                sw[targetnumber] = math.pow((targetdata[targetnumber]-targetmean),2)
#            sww[t] = np.sum(sw)
#            t=t+1
#        Fbet[voxel]=np.sum(sb)/(m-1)
#        Fwit[voxel]=np.sum(sww)/(n-m)
#        EV[voxel] = 1-(1/(Fbet[voxel]/Fwit[voxel]))
#        
#        
#        # Estimate Between Type Variance (mean squared error MS)
#        #     For each block, get each stim type. 
#        #     average each stim type, and form grand average
#        #     between type variance is defined as stim type average - grand average, summed. 
#        
#        #squarederror = np.empty([len(voxeldata),1])      
#        #for targnumber in range(len(ds.samples)):
#        #    squarederror[targnumber]=math.pow((voxeldata[targnumber]-globalmean),2)
#        #MSbet[voxel] = np.sum(squarederror)*(1/((m-1)*n))#this might be the wrong divider... 
#        
#        # Estimate Within Subject Variance 
#       
#        t=0
#        sumwithintypes = np.empty([m,1])
#        squaredmeanerror = np.empty([m,1])
#        for targettype in np.unique(ds.sa.targets):
#            targetdata = voxeldata[ds.sa.targets==targettype]
#            targetmean = np.mean(targetdata)          
#            squaredmeanerror[t]=math.pow((targetmean-globalmean),2)            
#            squarederror2 = np.empty([len(targetdata),1])  
#            for targetnumber in range(len(targetdata)):
#                squarederror2[targetnumber]=math.pow((targetdata[targetnumber]-targetmean),2)
#            sumwithintypes[t]=np.sum(squarederror2)
#            t=t+1
#        sumbetweentypesalt = np.sum(squaredmeanerror)
#        MSbetalt[voxel] = sumbetweentypesalt*(1/(m-1))
#        sumbetweentypes = np.sum(sumwithintypes)
#        MSwit[voxel] = sumbetweentypes*(1/((m-1)*n))
#        
#        # Shuffle Estimate 
#        #     shuffle the types within each block. 
#        #     recalculate the between type variance. 
#        #
#        #     subtract Shuffled variance with nonshuffled variance
#        #
#        #
#        #     Form the ratio of ... explained variance.. 
#
#        # Calculate Explainable Variance
#    
#       
#            #EVvoxel[voxel] = (MSbet[voxel]-(MSwit[voxel])/MSbet[voxel])
#        EVvoxel[voxel]= (MSbetalt[voxel]-((MSwit[voxel]/n)))/MSbetalt[voxel]
#        #print EVvoxel
#    print np.mean(EVvoxel)
    # Save
    

    ## Explainablel Variance and F ratio from Outside Source
        #total explainable Variance
    EV = np.empty([len(ds.samples[0]),1]) 
    F =  np.empty([len(ds.samples[0]),1]) 
    MSbetstore = np.empty([len(ds.samples[0]),1])
    MSwitstore = np.empty([len(ds.samples[0]),1])
    m=len(np.unique(ds.sa.targets))
    print m
    mtypes = np.unique(ds.sa.targets)
    n=len(ds.sa.targets)/m
    print n
    for voxel in range(len(ds.samples[0])):
        voxeldata = ds.samples[:,voxel]
        
        globalmean = np.mean(voxeldata)  
        sb = np.empty([m,1])
        sww=np.empty([m,1])
        for idx, targettype in enumerate(mtypes):
            targetdata = voxeldata[ds.sa.targets==[targettype]]
            targetmean = np.mean(targetdata)
            targetstd = np.std(targetdata)
            sb[idx]=n*(math.pow((targetmean-globalmean),2))
            sww[idx]=(n-1)*math.pow(targetstd,2)

        MSbet=np.sum(sb)/(m-1)
        MSwit=np.sum(sww)/(len(ds.sa.targets)-m) # divide by big N - number of groups
        F[voxel] = MSbet/MSwit
        Ftemp = MSbet/MSwit
        EV[voxel] = 1-1/Ftemp

    
    
    
    
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    out = EV #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'fs'+str(fs)+'out_EV.txt',out,fmt='%.8f',delimiter='\t')     
    out = F #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'fs'+str(fs)+'out_F.txt',out,fmt='%.8f',delimiter='\t')     
    
    
    
#    out = [np.mean(cv_results)]  
#    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'fs'+str(fs)+'out_acc.txt',out,fmt='%.8f',delimiter='\t')     
#    noskip = 0
#    out = EV2 #np.concatenate((EVvoxel,MSbet,MSwit),1)  
#    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'out_EVall.txt',out,fmt='%.8f',delimiter='\t')     
#    out = voxeldataout #np.concatenate((EVvoxel,MSbet,MSwit),1)  
#    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'out_voxeldata.txt',out,fmt='%.8f',delimiter='\t')     
#    a = np.empty([len(ds.sa.targets),1])    
#    a = ds.sa.targets #np.concatenate((EVvoxel,MSbet,MSwit),1)  
#    out = a.astype(np.float)    
#    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'out_targets.txt',out,fmt='%.8f',delimiter='\t')     
#    
    # do again with DS original.. 
    

        
elif analysis_number==11:    
    analysisfolder = 'SVM_Weights'
    
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    dsorig = ds
    baseclf = LinearCSVMC()
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    
   # get sensitive voxels    
    sensana = baseclf.get_sensitivity_analyzer(postproc=maxofabs_sample())
    cv_sensana = RepeatedMeasure(sensana,NFoldPartitioner())
    sens = cv_sensana(ds)
    sensmean = np.mean(sens,0) # mean across cross validation
    np.max(sensmean)
    len(sensmean[sensmean>0])
    sensmeansort = np.sort(sensmean)
    sensmeansort = sensmeansort[::-1]
    cutoff = sensmeansort[1000]
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ds.samples = ds.samples[:,sensmean>cutoff]
    print ds.samples.shape
    out = sensmean #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'out_SVMweights.txt',out,fmt='%.8f',delimiter='\t')     
 
 
 
elif analysis_number==12:
    analysisfolder = 'DoubleDipping'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    dsorig = ds
    baseclf = LinearCSVMC()
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fsel.train(ds)   
    ds = fsel(ds)    
    #fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    cvte = CrossValidation(baseclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    out = mat #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'fs'+str(fs)+'out_EV.txt',out,fmt='%.8f',delimiter='\t')     
  
  
elif analysis_number==13:
    
    
    analysisfolder = 'Explainable_Variance_All'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    baseclf = LinearCSVMC()
    cvte = CrossValidation(baseclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples   
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'outACC.txt',mat,fmt='%.3f',delimiter='\t')

     #total explainable Variance
    EV = np.empty([len(ds.samples[0]),1]) 
    MSbetstore = np.empty([len(ds.samples[0]),1])
    MSwitstore = np.empty([len(ds.samples[0]),1])
    m=len(np.unique(ds.sa.targets))
    mtypes = np.unique(ds.sa.targets)
    n=len(ds.sa.targets)/m

    
    
    ## Explainablel Variance from Paper.. 
    for voxel in range(len(ds.samples[0])):
        voxeldata = ds.samples[:,voxel]
        
        globalmean = np.mean(voxeldata)  
        sb = np.empty([m,1])
        sww=np.empty([m,1])
        for idx, targettype in enumerate(mtypes):
            targetdata = voxeldata[ds.sa.targets==[targettype]]
            targetmean = np.mean(targetdata)
            sb[idx]=math.pow((targetmean-globalmean),2)
            sw=np.empty([len(targetdata),1])
            for targetnumber in range(len(targetdata)):
                sw[targetnumber] = math.pow((targetdata[targetnumber]-targetmean),2)
            sww[idx] = np.sum(sw)

        MSbet=np.sum(sb)/(m-1)
        MSwit=np.sum(sww)/((m-1)*n)
        EV[voxel] = (MSbet-MSwit/n)/MSbet  
    
    
    out = EV #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'out_EV.txt',out,fmt='%.8f',delimiter='\t')     
   
     ## Explainablel Variance and F ratio from Outside Source
        #total explainable Variance
    EV = np.empty([len(ds.samples[0]),1]) 
    F =  np.empty([len(ds.samples[0]),1]) 
    MSbetstore = np.empty([len(ds.samples[0]),1])
    MSwitstore = np.empty([len(ds.samples[0]),1])
    m=len(np.unique(ds.sa.targets))
    print m
    mtypes = np.unique(ds.sa.targets)
    n=len(ds.sa.targets)/m
    print n
    for voxel in range(len(ds.samples[0])):
        voxeldata = ds.samples[:,voxel]
        
        globalmean = np.mean(voxeldata)  
        sb = np.empty([m,1])
        sww=np.empty([m,1])
        for idx, targettype in enumerate(mtypes):
            targetdata = voxeldata[ds.sa.targets==[targettype]]
            targetmean = np.mean(targetdata)
            targetstd = np.std(targetdata)
            sb[idx]=n*(math.pow((targetmean-globalmean),2))
            sww[idx]=(n-1)*math.pow(targetstd,2)

        MSbet=np.sum(sb)/(m-1)
        MSwit=np.sum(sww)/(len(ds.sa.targets)-m) # divide by big N - number of groups
        F[voxel] = MSbet/MSwit
        Ftemp = MSbet/MSwit
        EV[voxel] = 1-1/Ftemp
    

    out = EV #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'out_FEV.txt',out,fmt='%.8f',delimiter='\t')     
    out = F #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'out_F.txt',out,fmt='%.8f',delimiter='\t')     
    
    
   
    # Delete top 100 voxels 
    indices = np.argsort(np.transpose(F))[0,::-1]
    print ds.shape
    #ds.samples = np.delete(ds.samples,indices[0:1000],axis=1)
    #ds.fa.voxel_indices = np.delete(ds.fa.voxel_indices,indices[0:1000],axis=0)
    #fs0 = StaticFeatureSelection(indices)
    #fs0(ds).samples
    ds = ds[:,indices[700::]]    
    print ds.shape
    baseclf = LinearCSVMC()
    cvte = CrossValidation(baseclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples   
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'outACC-700.txt',mat,fmt='%.3f',delimiter='\t')


elif analysis_number==14:
    
    
    analysisfolder = 'Explainable_Variance_CrossVal'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    baseclf = LinearCSVMC()
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)      
    
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples   
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'outACC.txt',mat,fmt='%.3f',delimiter='\t')


    # Feature Selection Voxels. 
    sensana = fselclf.get_sensitivity_analyzer(postproc=maxofabs_sample())
    cv_sensana = RepeatedMeasure(sensana,NFoldPartitioner())
    sens = cv_sensana(ds)
    ov = MapOverlap()
    overlap_fraction = ov(sens.samples>0) # Look for stability in feature selection
    print overlap_fraction
   
    # another way to call back..    
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats'],callback=store_me);
    
     ## Explainablel Variance and F ratio from Outside Source
        #total explainable Variance
    
    m=len(np.unique(ds.sa.targets))
    print m
    mtypes = np.unique(ds.sa.targets)
    n=len(ds.sa.targets)/m
    print n
    for cross in range(len(sens)):
        dscross = ds.samples[:,sens.samples[cross,:]>0]
        EV = np.empty([len(dscross[0]),1]) 
        F =  np.empty([len(dscross[0]),1]) 
        MSbetstore = np.empty([len(dscross[0]),1])
        MSwitstore = np.empty([len(dscross[0]),1])
        for voxel in range(len(dscross[0])):
            voxeldata = dscross[:,voxel]
            
            globalmean = np.mean(voxeldata)  
            sb = np.empty([m,1])
            sww=np.empty([m,1])
            for idx, targettype in enumerate(mtypes):
                targetdata = voxeldata[ds.sa.targets==[targettype]]
                targetmean = np.mean(targetdata)
                targetstd = np.std(targetdata)
                sb[idx]=n*(math.pow((targetmean-globalmean),2))
                sww[idx]=(n-1)*math.pow(targetstd,2)
    
            MSbet=np.sum(sb)/(m-1)
            MSwit=np.sum(sww)/(len(ds.sa.targets)-m) # divide by big N - number of groups
            F[voxel] = MSbet/MSwit
            Ftemp = MSbet/MSwit
            EV[voxel] = 1-1/Ftemp
        out = EV #np.concatenate((EVvoxel,MSbet,MSwit),1)  
        np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'out_FEV'+str(cross)+'.txt',out,fmt='%.8f',delimiter='\t')     
        out = F #np.concatenate((EVvoxel,MSbet,MSwit),1)  
        np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'out_F'+str(cross)+'.txt',out,fmt='%.8f',delimiter='\t')     
    out = overlap_fraction
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'outOverlap.txt',out,fmt='%.8f',delimiter='\t')     


elif analysis_number==15:
    
    
    analysisfolder = 'RecursiveFE'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    #rfesvm_split = SplitClassifier(LinearCSVMC())
    baseclf = LinearCSVMC(); 
    
    #fsel=RFE(sensitivity_analyzer=rfesvm_split.get_sensitivity_analyzer(),transfer_error=ConfusionBasedError(rfesvm_split,confusion_state="stats"),Repeater(2),feature_selector=FractionTailSelector(0.2,mode='discard',tail='lower'),update_sensitivity=True,descr='LinSVM+RFE') 
    rfesvm_split = SplitClassifier(LinearCSVMC(), OddEvenPartitioner())
    rfe = RFE(rfesvm_split.get_sensitivity_analyzer( \
          postproc=ChainMapper([ FxMapper('features', l2_normed),FxMapper('samples', np.mean),FxMapper('samples', np.abs)])), \
          ConfusionBasedError(rfesvm_split, confusion_state='stats'), \
          Repeater(2),\
          fselector=FractionTailSelector(0.20,mode='select', tail='upper'), \
          stopping_criterion=NBackHistoryStopCrit(BestDetector(), 10),train_pmeasure=False,update_sensitivity=True)    
    
    clf = FeatureSelectionClassifier(baseclf,rfe)
    cvte = CrossValidation(clf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print ds.shape
    print np.mean(cv_results)
    
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    out = mat #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'fs'+str(fs)+'out_acc.txt',out,fmt='%.8f',delimiter='\t')     
  
elif analysis_number==16:
    
    
    analysisfolder = 'OtherClassifiers'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    
    #classifierlist = []    
    
    baseclf = clfswh['random-forest']
    print baseclf
    baseclf = baseclf[0]
    print baseclf
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    #confusion matrix
    print cvte.ca.stats.as_string(description=True)
    print cvte.ca.stats.matrix # reduced form
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'class'+baseclf.descr+'out_acc.txt',mat,fmt='%.3f',delimiter='\t')

elif analysis_number==17:
    
    
    analysisfolder = 'OtherFS'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    
    #classifierlist = []    
    
    baseclf =  LinearCSVMC()
    fsel = SensitivityBasedFeatureSelection(CorrStability(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    print np.mean(cv_results)
    #confusion matrix
    print cvte.ca.stats.as_string(description=True)
    print cvte.ca.stats.matrix # reduced form
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'out_acc.txt',mat,fmt='%.3f',delimiter='\t')

elif analysis_number==18:
    
    # SVM Classification with Meta Param Fitting 
    analysisfolder = 'Nested_AllCLF'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    clfgroup=[]
    params = [3,-8,-7,-6,-5,-4,-3,-9,-10,-11,-2,-1,1,2,4,5,6,7]
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    for r in params:  # make the classifier list #
        baseclf = LinearCSVMC(C=math.pow(10,r))
        baseclf = FeatureSelectionClassifier(baseclf,fsel) # probably dont need fsel if listen to thomas.
        clfgroup.append(baseclf)
    best_clfs = {}
    confusion = ConfusionMatrix()
    verbose(1, "Estimating error using nested CV for model selection")
    partitioner = NFoldPartitioner()
    splitter = Splitter('partitions')
    cross_acc = []
    best_list = []
    s = 0
    for isplit, partitions in enumerate(partitioner.generate(ds)):
        verbose(2, "Processing split #%i" % isplit)
        print s
        dstrain, dstest = list(splitter.generate(partitions))
        best_clf, best_error = select_best_clf(dstrain, clfswh['!gnpp'])
        print best_error
        best_list.append(best_clf.summary)
        best_clfs[best_clf.descr] = best_clfs.get(best_clf.descr, 0) + 1
        tm = TransferMeasure(best_clf, splitter,postproc=BinaryFxNode(mean_mismatch_error,space='targets'),enable_ca=['stats'])
        result = tm(partitions) # probably need to do equals...
        #print result.ca
        confusion += tm.ca.stats
        print confusion   
        cross_acc.append(confusion.stats['mean(ACC)'])
        s = s + 1
        
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    #mat = np.empty([len(cv_results)+1,1])
    #mat[0] = np.mean(cv_results)
    #mat[1:] = cv_results.samples
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+'ACC_out.txt',cross_acc,fmt='%.3f',delimiter='\t')
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+'Param_out.txt',best_list,fmt='%s',delimiter='\t')

elif analysis_number==19:
    
    # SVM Classification with Meta Param Fitting 
    analysisfolder = 'KNN_nested3'
    analysisfolder = 'PLR nested'
    analysisfolder = 'SMLR_nested_nofs'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    clfgroup=[]
    #params = [3,-8,-7,-6,-5,-4,-3,-9,-10,-11,-2,-1,1,2,4,5,6,7]
    params = [2,3,4,5,6,7,8,9,10,15,20,25,30,40,50]   
    params = [-2,-1,1,2,3]
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
  
    baseclf = clfswh.get_by_descr('libsvm.LinSVM(C=1)')
    baseclf = clfswh.get_by_descr('kNN(k=5)') # this is annoying once created can change k
   
    for r in params:  # make the classifier list #
        baseclf = kNN(k=r, voting='majority', descr="kNN(k="+str(r))
        baseclf = PLR(lm=math.pow(10,r)) # also can specifiy reduced... which is different
        baseclf = SMLR(lm=math.pow(10,r))
        #baseclf = FeatureSelectionClassifier(baseclf,fsel) # probably dont need fsel if listen to thomas.
        clfgroup.append(baseclf)
    best_clfs = {}
    
    alt_clf = RbfCSVMC()
    confusion = ConfusionMatrix()
    verbose(1, "Estimating error using nested CV for model selection")
    partitioner = NFoldPartitioner()
    splitter = Splitter('partitions')
    cross_acc = []
    best_list = []
    cross_acc2 = []
    s = 0
    for isplit, partitions in enumerate(partitioner.generate(ds)):
        # partitions becomes a dataset with partitions.sa.partitions. (11111111,222) for that cross valid
        verbose(2, "Processing split #%i" % isplit)
        print s
        dstrain, dstest = list(splitter.generate(partitions))
        best_clf, best_error = select_best_clf(dstrain, clfgroup) # feed it dstrain which has 1-14 runs excluding 15. It cross validates and selects best param)
        print best_error
        best_list.append(best_clf.summary)
        best_clfs[best_clf.descr] = best_clfs.get(best_clf.descr, 0) + 1
        best_clf.train(dstrain)
        pred = best_clf.predict(dstest)
        acc = np.mean(pred==dstest.sa.targets)
        cross_acc2.append(acc)        
        tm = TransferMeasure(best_clf, splitter,postproc=BinaryFxNode(mean_mismatch_error,space='targets'),enable_ca=['stats'])
        result = tm(partitions) # probably need to do equals...
        #print result.ca
        confusion += tm.ca.stats
        print confusion   
        cross_acc.append(confusion.stats['mean(ACC)'])
        s = s + 1
    #cross_acc.append(np.mean(cross_acc))    
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    #mat = np.empty([len(cv_results)+1,1])
    #mat[0] = np.mean(cv_results)
    #mat[1:] = cv_results.samples
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+'ACC_out.txt',cross_acc,fmt='%.3f',delimiter='\t')
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+'ACC2_out.txt',cross_acc,fmt='%.3f',delimiter='\t')
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'onset'+onset+'Param_out.txt',best_list,fmt='%s',delimiter='\t')


elif analysis_number==20:
    
    c = 1000; 
    # Training Vs Testing** (Loop Through C values)  
    analysisfolder = 'training_v_testing'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    baseclf = LinearCSVMC(C=c, enable_ca=['training_stats'])
    
    rs = []
    errors, training_errors = [], []    
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    
    cvte = CrossValidation(baseclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    sana = baseclf.get_sensitivity_analyzer(postproc=None)    
    sa = sana(ds)
    w = sa.samples[0]
    b = np.asscalar(sa.sa.biases)
    # width each way
    r = 1./np.linalg.norm(w)
    training_error = baseclf.ca.training_stats.stats['ACC']
    print training_error
    print np.mean(cv_results)
    #confusion matrix
    print cvte.ca.stats.as_string(description=True)
    print cvte.ca.stats.matrix # reduced form
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'c'+str(c)+'test.txt',mat,fmt='%.3f',delimiter='\t')
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = training_error
    mat[1] = r
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'c'+str(c)+'train.txt',mat,fmt='%.3f',delimiter='\t')


elif analysis_number==21:  

    analysisfolder = 'voxel_data'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    baseclf = LinearCSVMC(C=0.01,enable_ca='training_stats') 
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    #cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    #cv_results = cvte(ds)
    #sensitivity analysis
    #sensana = fselclf.get_sensitivity_analyzer(postproc=maxofabs_sample())
    #type(sensana)
    #cv_sensana = RepeatedMeasure(sensana,NFoldPartitioner())
    #sens = cv_sensana(ds) # repeats the feature selection... 
    
    # that didnt work
    # BUT IT DID GIVE DIFFERENT SENSITIVITIES when i did each chunk individually
    sensitivities = []
    indexes = []
    def store_me(data,node,result):
        sens = node.measure.get_sensitivity_analyzer(force_train=False)(data)
        index = node.measure.mapper.slicearg
        sensitivities.append(sens)
        indexes.append(index)
        
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats'],callback=store_me);
    cv_results = cvte(ds)
    
    sen = sensitivities[0].samples    
    for c in range(len(sensitivities)-1):
        sent = sensitivities[c+1].samples # extract numpy array
        sen = np.concatenate((sen,sent),axis=0)
    sen1= sen
    sen = sen<>0
    numb_runs_used = np.sum(sen,0)
    match = np.sum(sen[0,indexes[0]])
    
    limiteddata = ds[:,sen[0,:]]
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    ## SAVE RESULTS
    import scipy.io as io
    #np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'fs_voxel.txt',limiteddata.samples,fmt='%.5f',delimiter='\t')
    io.savemat(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'fs_voxel.mat',dict(d=limiteddata.samples))
    io.savemat(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'fs_svm.mat',dict(svm=sen1[0,sen[0,:]]))
    io.savemat(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'fs_svm_bias.mat',dict(bias=sensitivities[0].sa.biases))
    
    
        
    a = np.empty([len(ds.sa.targets),1])    
    a = ds.sa.targets #np.concatenate((EVvoxel,MSbet,MSwit),1)  
    out = a.astype(np.float)    
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+str(roi_number)+'sub'+subject+'out_targets.txt',out,fmt='%.5f',delimiter='\t')  
    
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'out_acc.txt',mat,fmt='%.3f',delimiter='\t')

    print 'done'
    #EV2 = np.transpose(EV2)
    #cor = np.corrcoef(EV2[~np.isnan(EV2)],sensmean[~np.isnan(EV2[:,0])])    
    #
    #pdata = EVvoxel[~np.isnan(EVvoxel)]
    # pdata = pdata[pdata>-1]
    # hist(pdata)

#baseclf3 = PLR(criterion=0.00001)
    #baseclf2 = kNN(k=1,dfx=one_minus_correlation,voting='majority')

elif analysis_number==22:  
    
    
  
    #### ONE SPLIT (DATA CHARACTERIZATION)
    analysisfolder = 'characterization2!z'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs..
    ds= ds.get_mapped(detrender)
    
    ##
   # ds = ds[ds.sa.targets != ['0']]
   # zscore(ds,param_est=('targets',['1','2'])) # zscore with respect to rest.
   
    zscore(ds,param_est=('targets',['0']))
    ds = ds[ds.sa.targets != ['0']]
    ##    
    ## Parition Data..
    dstrain = ds[ds.sa.chunks != [np.max(ds.sa.chunks)]]
    dstest = ds[ds.sa.chunks == [np.max(ds.sa.chunks)]]
    ## Apply Various Mappers to Data
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fsel.train(dstrain)
    dstrain =dstrain.get_mapped(fsel)# actually changes the dataset
    dstest  =dstest.get_mapped(fsel)
    # Get Characteristics of top 500
    Ftrain,EVtrain = explainable_variance(dstrain)
    Ftest,EVtest = explainable_variance(dstest)
    cls = LinearCSVMC(C=1, enable_ca=['training_stats'])
    cls.train(dstrain)
    training_stats500 = cls.ca.training_stats ##### check to make sure dstrain is smaller... 
    print training_stats500 
    res = cls(dstest)
    stats500 = cls.__summary_class__(
    targets=res.sa[cls.get_space()].value,
    predictions=res.samples[:, 0],
    estimates=cls.ca.get('estimates', None))
    print stats500
    sens_analyzer = cls.get_sensitivity_analyzer()
    sensitivity = sens_analyzer(dstrain)

    data={'trainaccuracy500':training_stats500.stats['ACC'], 'testaccuracy500':stats500.stats['ACC']}
    data['svm_weights500'] = sensitivity.samples
    data['svm_bias500']= sensitivity.sa.biases
    data['EV500train'] = EVtrain
    data['EV500test'] = EVtest
    data['F500train'] = Ftrain
    data['F500test'] = Ftest
    data['voxeltrain'] = dstrain.samples
    data['voxeltest'] = dstest.samples
    
    # Get Characterstics of 2 Top SVD Dimensions
    smap = SVDMapper()
    smap.train(dstrain)
    dstrain = dstrain.get_mapped(smap)
    dstest = dstest.get_mapped(smap)
    stsel = StaticFeatureSelection(slice(None,2))
    dstrain = dstrain.get_mapped(stsel)
    dstest = dstest.get_mapped(stsel)
    cls = LinearCSVMC(C=1, enable_ca=['training_stats'])
    cls.train(dstrain)
    training_stats = cls.ca.training_stats ##### check to make sure dstrain is smaller... 
    print training_stats 
    res = cls(dstest)
    stats = cls.__summary_class__(
    targets=res.sa[cls.get_space()].value,
    predictions=res.samples[:, 0],
    estimates=cls.ca.get('estimates', None))
    print stats
    
    sens_analyzer = cls.get_sensitivity_analyzer()
    sensitivity = sens_analyzer(dstrain)
    print sensitivity.samples
    print sensitivity.sa.biases
    print sensitivity.targets # what does this mean? 
    
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    import scipy.io as io
    data['svm_weights'] = sensitivity.samples
    data['svm_bias']= sensitivity.sa.biases
    data['SVtrain']= dstrain.samples
    data['SVtest']=dstest.samples
    data['traintargs']=dstrain.targets
    data['testtargs']=dstest.targets
    data['trainaccuracySV'] = training_stats.stats['ACC']
    data['testaccuracySV'] = stats.stats['ACC']
    
    attr = SampleAttributes(os.path.join(subjectbehfolder,'TRtypeonsets'+onset+'.txt'))
    data['targs8'] = attr.targets
    io.savemat(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'.mat',data)
    
    
 
elif analysis_number==23:    
    analysisfolder = 'voxel_data_type'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs..
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest.
    ds = ds[ds.sa.targets != ['0']]
    ## Parition Data..
    dstrain = ds[ds.sa.chunks != [np.max(ds.sa.chunks)]]
    dstest = ds[ds.sa.chunks == [np.max(ds.sa.chunks)]]
    ## Apply Various Mappers to Data
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fsel.train(dstrain)
    dstrain =dstrain.get_mapped(fsel)# actually changes the dataset
    dstest  =dstest.get_mapped(fsel)
    # Get Characteristics of top 500
    Ftrain,EVtrain = explainable_variance(dstrain)
    Ftest,EVtest = explainable_variance(dstest)
    cls = LinearCSVMC(C=1, enable_ca=['training_stats'])
    cls.train(dstrain)
    training_stats500 = cls.ca.training_stats ##### check to make sure dstrain is smaller... 
    print training_stats500 
    res = cls(dstest)
    stats500 = cls.__summary_class__(
    targets=res.sa[cls.get_space()].value,
    predictions=res.samples[:, 0],
    estimates=cls.ca.get('estimates', None))
    print stats500
    data={'trainaccuracy500':training_stats500.stats['ACC'], 'testaccuracy500':stats500.stats['ACC']}
    data['EV500train'] = EVtrain
    data['EV500test'] = EVtest
    data['F500train'] = Ftrain
    data['F500test'] = Ftest
    data['voxeltrain'] = dstrain.samples
    data['voxeltest'] = dstest.samples
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    import scipy.io as io
    io.savemat(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'.mat',data)
   
   
elif analysis_number==24:
    
    # Shuffle Correlation Strucutre***
    analysisfolder = 'Shuffle_Correlation2'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    
    
    iteracc = []
    #instead... loop here.. 10 times copy dataset.
    for i in range(10):
        dscopy = ds.copy(deep=True)
        
        partitioner = NFoldPartitioner()
        splitter = Splitter('partitions')
        cross_acc = []
        best_list = []
        cross_acc2 = []
        s = 0
        for isplit, partitions in enumerate(partitioner.generate(dscopy)):
            # partitions becomes a dataset with partitions.sa.partitions. (11111111,222) for that cross valid
            verbose(2, "Processing split #%i" % isplit)
            print s
            dstrain, dstest = list(splitter.generate(partitions))
            dstraincopy = dstrain.copy(deep=True)
            dstestcopy = dstest.copy(deep=True)
            fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
            fsel.train(dstraincopy)
            dstraincopy =dstraincopy.get_mapped(fsel)# actually changes the dataset
            dstestcopy  =dstestcopy.get_mapped(fsel)
            sizee = dstraincopy.shape
            mtypes = np.unique(ds.sa.targets)
            for voxel in range(sizee[1]):
                voxeldata = dstraincopy.samples[:,voxel]
                voxeldata2 = dstestcopy.samples[:,voxel]
                for idx, targettype in enumerate(mtypes):
                    targetdata = voxeldata[dstraincopy.sa.targets==[targettype]]
                    targetdata2 = voxeldata2[dstestcopy.sa.targets==[targettype]]
                    
                     #Shuffle #\
                    np.random.shuffle(targetdata) # oddly you dont set it equal to anything.
                    np.random.shuffle(targetdata2)
                    voxeldata[dstraincopy.sa.targets==[targettype]] = targetdata  
                    voxeldata2[dstestcopy.sa.targets==[targettype]] = targetdata2
                dstraincopy.samples[:,voxel] = voxeldata # why doesnt this work***
                dstestcopy.samples[:,voxel] = voxeldata2 
                
                # check if unalter DS train, is it same as putting target data out and in.. 
            cls = LinearCSVMC() 
            cls.train(dstraincopy)
            res = cls(dstestcopy)
            stats = cls.__summary_class__(
            targets=res.sa[cls.get_space()].value,
            predictions=res.samples[:, 0],
            estimates=cls.ca.get('estimates', None))
            print stats
            cross_acc2.append(stats.stats['ACC'])
            s = s + 1
        iteracc.append(np.mean(cross_acc2))
        
    
    cls = LinearCSVMC() 
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(cls,fsel) 
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats']);
    cv_results = cvte(ds)
    mat = np.empty([len(cv_results)+1,1])
    mat[0] = np.mean(cv_results)
    mat[1:] = cv_results.samples
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    data={'caccuracy':iteracc}
    data['caccuracy_originalmethod'] = mat
    import scipy.io as io
    io.savemat(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'.mat',data)
   
   
elif analysis_number==25:  

   
    # Get Rule Data
    analysisfolder = 'VoxelOut'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]
    baseclf = LinearCSVMC() 
    fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
    fselclf = FeatureSelectionClassifier(baseclf,fsel)    
    sensitivities = []
    indexes = []
    def store_me(data,node,result):
        sens = node.measure.get_sensitivity_analyzer(force_train=False)(data)
        index = node.measure.mapper.slicearg
        sensitivities.append(sens)
        indexes.append(index)
    cvte = CrossValidation(fselclf,NFoldPartitioner(attr='chunks'),errorfx=lambda p,t: np.mean(p==t),enable_ca=['stats'],callback=store_me);
    cv_results = cvte(ds)    
    indexes = np.asarray(indexes)
    allindexes = np.unique(indexes)
    dataout = ds.samples[:,allindexes]
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    mat = dataout
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'voxels_rule.txt',mat,fmt='%.10f',delimiter='\t')
    l = ds.sa.targets
    l = l.astype(np.float)
    r = ds.sa.chunks
    r = r.astype(np.float)
    mat = np.vstack((r,l))
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'rule.txt',mat,fmt='%.10f',delimiter='\t')
    mat[0] = np.mean(cv_results)
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'acc_rule.txt',mat,fmt='%.10f',delimiter='\t')
    
    # Get Type Data
    attr = SampleAttributes(os.path.join(subjectbehfolder,'TRtypeonsets'+onset+'.txt')) # order 2 has exact timings...
    ds2 = fmri_dataset(os.path.join(subjectfolder,'wraf.nii.gz'),targets=attr.targets,chunks=attr.chunks,mask=os.path.join(roifolder,rois[roi_number]+'.nii.gz')) # can add mask here. 
    zscore(ds2,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds2 = ds2[ds2.sa.targets != ['0']]
    dataout = ds2.samples[:,allindexes] # reuse the rule indexes*** measuring type data within rule selection**
    mat = dataout
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'voxels_type.txt',mat,fmt='%.10f',delimiter='\t')
    l = ds2.sa.targets
    l = l.astype(np.float)
    r = ds2.sa.chunks
    r = r.astype(np.float)
    print l
    mat = np.vstack((r,l))
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'type.txt',mat,fmt='%.10f',delimiter='\t')
    ##mat[0] = np.mean(cv_results)
    ##np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'acc_type.txt',mat,fmt='%.10f',delimiter='\t')
    

elif analysis_number==26:  

    # Get Rule Data
    analysisfolder = 'Correlation_BySplit_Unbiased'
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]   
    
    # Choose a partition   
    partitiontype = 'nfold'
    partitioner = NFoldPartitioner()
    #partitioner = OddEvenPartitioner()
    splitter = Splitter('partitions')
    within_rule_eo = np.empty(0) 
    between_rule_eo=np.empty(0)
    disimilarity = np.empty(0)
    cormatrix = np.zeros((2,2))
    acc = np.empty(0)
    dscopy = ds.copy(deep=True)
    sensitivities = []
    indexes = []
    for isplit, partitions in enumerate(partitioner.generate(dscopy)):
        dstrain, dstest = list(splitter.generate(partitions))
        dstraincopy = dstrain.copy(deep=True)
        dstestcopy = dstest.copy(deep=True)
        # feature select on training patterns (one split)
        fsel = SensitivityBasedFeatureSelection(OneWayAnova(),FixedNElementTailSelector(fs,mode='select',tail='upper'))
        fsel.train(dstraincopy)
        dstraincopy =dstraincopy.get_mapped(fsel)# actually changes the dataset
        dstestcopy  =dstestcopy.get_mapped(fsel)
        
        # correlate testing and training patterns
        rule1e = np.mean(dstraincopy.samples[dstraincopy.targets=='1',:],axis=0) 
        rule1o = np.mean(dstestcopy.samples[dstestcopy.targets=='1',:],axis=0) 
        rule2e = np.mean(dstraincopy.samples[dstraincopy.targets=='2',:],axis=0) 
        rule2o = np.mean(dstestcopy.samples[dstestcopy.targets=='2',:],axis=0) 
        recombined = np.vstack((rule1e,rule1o,rule2e,rule2o))
        temp_corr = np.zeros((2,2))
        tc = np.corrcoef(rule1e,rule1o)
        temp_corr[0,0] = tc[1,0]
        tc = np.corrcoef(rule2e,rule1o)
        temp_corr[0,1] = tc[1,0]
        tc = np.corrcoef(rule1e,rule2o)
        temp_corr[1,0] = tc[1,0]
        tc = np.corrcoef(rule2e,rule2o)
        temp_corr[1,1] = tc[1,0]
        
        cormatrix = np.vstack((cormatrix,temp_corr)) # add the correlations up.. divide at end
        #within_rule_eo = np.append(within_rule_eo, (cormatrix[1,0]+cormatrix[3,2])/2)
        #between_rule_eo = np.append(between_rule_eo,((cormatrix[2,0]+cormatrix[3,0])/2+(cormatrix[2,1]+cormatrix[3,1])/2)/2)
        #disimilarity = np.append(disimilarity,np.sqrt((np.square(cormatrix[1,0])+np.square(cormatrix[3,2]))/2)-between_rule_eo[isplit])
       
        # get voxels for type calculation*
        index = fsel.slicearg
        indexes.append(index)
        
        #classify
        recombined = np.vstack((rule1e,rule2e,rule1o,rule2o)) # odds are testing patterns..
        cormatrix2 = np.corrcoef(recombined)
        cormatrix2 = np.tril(cormatrix2,-1)
        if cormatrix2[2,0]>cormatrix2[2,1]:
            guess_rule1 = 1
        else: 
            guess_rule1 = 0
        if cormatrix2[3,0]>cormatrix2[3,1]:
            guess_rule2 = 0
        else: 
            guess_rule2 = 1
        acc = np.append(acc,float(guess_rule1+guess_rule2)/2)

    indexes = np.asarray(indexes)
    if not os.path.exists(resultsfolder+analysisfolder):
        os.makedirs(resultsfolder+analysisfolder)
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'rule_'+partitiontype+'_acc.txt',acc,fmt='%.10f',delimiter='\t')
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'rule_'+partitiontype+'_cormat.txt',cormatrix,fmt='%.10f',delimiter='\t')
    
    ##### REDO FOR TYPE 
    attr = SampleAttributes(os.path.join(subjectbehfolder,'TRtypeonsets'+onset+'.txt')) # order 2 has exact timings...
    print("Loading Data...")
    ds = fmri_dataset(os.path.join(subjectfolder,'wraf.nii.gz'),targets=attr.targets,chunks=attr.chunks,mask=os.path.join(roifolder,rois[roi_number]+'.nii.gz')) # can add mask here. 
    detrender = PolyDetrendMapper(polyord=1,chunks_attr='chunks') #does so by runs.. 
    ds= ds.get_mapped(detrender)
    zscore(ds,param_est=('targets',['0'])) # zscore with respect to rest. 
    ds = ds[ds.sa.targets != ['0']]        
    dscopy = ds.copy(deep=True)
    within_type_eo_avg = np.empty((0,8))
    for isplit, partitions in enumerate(partitioner.generate(dscopy)):
        dstrain, dstest = list(splitter.generate(partitions))
        dstraincopy = dstrain.copy(deep=True)
        dstestcopy = dstest.copy(deep=True)
        dstraincopy =dstraincopy[:,indexes[isplit]]# actually changes the dataset
        dstestcopy  =dstestcopy[:,indexes[isplit]]
        within_type_eo_t = np.zeros((1,8))
        for typee in [0,1,2,3,4,5,6,7]:
            mean_even = np.nanmean(dstraincopy.samples[dstraincopy.targets==str(typee),:],axis=0)
            mean_odd =  np.nanmean(dstestcopy.samples[dstestcopy.targets==str(typee),:],axis=0)
            cormatrixT = np.corrcoef(np.vstack((mean_even,mean_odd)))
            within_type_eo_t[0,typee] = cormatrixT[1,0]
        within_type_eo_avg = np.vstack((within_type_eo_avg,within_type_eo_t))
        
    mat = within_type_eo_avg
    np.savetxt(resultsfolder+analysisfolder+'/'+'roi'+rois[roi_number]+'sub'+subject+'fs'+str(fs)+'onset'+onset+'type_'+partitiontype+'.txt',mat,fmt='%.10f',delimiter='\t')
    
    






    
    

    


