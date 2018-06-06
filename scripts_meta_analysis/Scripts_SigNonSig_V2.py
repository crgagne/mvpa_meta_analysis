

import pandas as pd
import numpy as np

##### DATA IMPORT FUNCTIONS ########
##### V1-->V2 was done for JOCN Comments. Using version v8b dataset. 5/10/18


def get_data_coutanche():
	df_coutanche = pd.read_csv('../data_meta_analysis/coutanche_data_flat.csv')

	# preprocess a bit
	df_coutanche['Accuracy.50']=df_coutanche['accuracy']/100.0
	df_coutanche.loc[df_coutanche['chance']!=50.0,'Accuracy.50']=np.nan


	df_coutanche_50 = df_coutanche.loc[~np.isnan(df_coutanche['Accuracy.50']),]
	df_coutanche_50.head()
	return(df_coutanche_50)


def average_data_within_a_region(data_simple_one_region,R1names):
	# this is slow
	started = 0
	for PID in data_simple_one_region.PID.unique():
	    study_data = data_simple_one_region.loc[data_simple_one_region.PID==PID,]

	    # for a single study #
	    for region in list(R1names):
		# get the region #
		region_data = study_data.loc[study_data[region]==1,]
		if len(region_data)>0:

		    # create new region data.
		    region_data_single_row =region_data.iloc[0]
		    # average the accuracy over the other observations #
		    region_data_single_row['Accuracy.50'] = np.mean(region_data['Accuracy.50'])

		    # append the rows to a big data frame
		    if started==0:
		        new_df = region_data_single_row.as_matrix()
		        started = 1
		    else:
		        new_df = np.vstack((new_df,region_data_single_row.as_matrix()))

	new_df = pd.DataFrame(new_df)

	return(new_df)


def average_data_within_a_region_nonsingle_regions(df,R1names):

	started = 0
	for PID in df.PID.unique():
	    # get study data
	    study_data = df.loc[df.PID==PID,]

	    # get regions part of data frame as a matrix
	    r = study_data[R1names].as_matrix()

	    # get unique rows (ie. unqiue combinations of 0,1,0,1) for the regions
	    unique_r = np.vstack({tuple(row) for row in r})

	    # loop through all those
	    for rr in unique_r:
    		index = (rr==r).all(1) # shows which in r, match unique r
	 	region_data = study_data[index]

		 # create new region data.
		region_data_single_row =region_data.iloc[0]
	         # average the accuracy over the other observations #
	        region_data_single_row['Accuracy.50'] = np.mean(region_data['Accuracy.50'])

		# append the rows to a big data frame
		if started==0:
		        new_df = region_data_single_row.as_matrix()
		        started = 1
		else:
			new_df = np.vstack((new_df,region_data_single_row.as_matrix()))

	new_df = pd.DataFrame(new_df)
	return(new_df)



def get_data_for_sig_nonsig_analyses(average_within_region=False,version='vb8'):

	out = {}
	df = pd.read_csv('../data_meta_analysis/Data_Classifications_'+version+'.csv')

	# add regions #
	R1 = pd.read_csv("../data_meta_analysis/data_derived_meta_analysis/X_region1_before_clustering_"+version+".csv")
	R1names = list(R1.columns)
	df = pd.concat((df,R1),axis=1)


	# get just 50% accuracies
	df_50 = df.loc[np.logical_not(np.isnan(df.loc[:,'Accuracy.50']))] # and at ROI's.. (searchlights take maximum )



	# Get a data frame with only observations from one region #
	for row_num in range(R1.shape[0]):
	    R1.loc[row_num,:]==1
	    r = R1.loc[row_num,:]==1
	    regions_for_observation = list(r.index[R1.loc[row_num,:]==1])

	    # if observation has just one region
	    if len(regions_for_observation)==1:
		region = regions_for_observation[0]
		df.loc[row_num,'oneregion']=True
	    else:
		df.loc[row_num,'oneregion']=False
	data_simple_one_region = df.loc[df.oneregion==True,]

	# get just 50% again for this
	data_simple_one_region = data_simple_one_region[np.logical_not(np.isnan(df.loc[:,'Accuracy.50']))]


	# split sig / non sig
	df_50_sig = df_50[df_50['Significance']=='1']
	df_50_nonsig = df_50[df_50['Significance']=='0']

	if average_within_region:
	# get average sig and non-sig in a region #
		data_simple_one_region_sig = data_simple_one_region[data_simple_one_region['Significance']=='1']
		data_simple_one_region_nonsig = data_simple_one_region[data_simple_one_region['Significance']=='0']

		data_simple_one_region_average_within_region_sig = average_data_within_a_region(data_simple_one_region_sig,R1names)
		data_simple_one_region_average_within_region_sig.columns = data_simple_one_region.columns

		data_simple_one_region_average_within_region_nonsig = average_data_within_a_region(data_simple_one_region_nonsig,R1names)
		data_simple_one_region_average_within_region_nonsig.columns = data_simple_one_region.columns

		out['data_simple_one_region_average_within_region_sig'] = data_simple_one_region_average_within_region_sig
		out['data_simple_one_region_average_within_region_nonsig'] = data_simple_one_region_average_within_region_nonsig



	# get average sig / non-sig in all sets of unique regions
	data_average_within_region_sig = average_data_within_a_region_nonsingle_regions(df_50_sig,R1names)
	data_average_within_region_sig .columns = df_50_sig.columns

	data_average_within_region_nonsig = average_data_within_a_region_nonsingle_regions(df_50_nonsig,R1names)
	data_average_within_region_nonsig .columns = df_50_nonsig.columns

	out['data_average_within_region_sig'] = data_average_within_region_sig
	out['data_average_within_region_nonsig'] = data_average_within_region_nonsig



	df_50_nonsig = df_50[df_50['Significance']=='0']
	# dropping non-representive values for no significant..
	df_50_nonsig = df_50_nonsig.loc[~(df_50_nonsig['PID']==31)] # these ones use .0001 pvalue.

	## create design matrix  (study indicator )
	df_50_sig = df_50_sig.sort_values(by='PID') ## careful with the soring #
	uni_sig = df_50_sig['PID'].unique()
	X_sig = np.zeros((len(df_50_sig),len(uni_sig)))
	for pp,pid in enumerate(uni_sig):
	    X_sig[:,pp] = (df_50_sig['PID'].as_matrix()==pid).astype('float')
	    df_50_sig.loc[df_50_sig['PID'].as_matrix()==pid,'Xid']=pp
	    #data_50_paper.loc[data_50_paper['PID'].as_matrix()==pid,'Xid']=pp

	df_50_nonsig = df_50_nonsig.sort_values(by='PID') ## Careful with the sorting ##
	uni_nonsig = df_50_nonsig['PID'].unique()
	X_nonsig = np.zeros((len(df_50_nonsig),len(uni_nonsig)))
	for pp,pid in enumerate(uni_nonsig):
	    X_nonsig[:,pp] = (df_50_nonsig['PID'].as_matrix()==pid).astype('float')
	    df_50_nonsig.loc[df_50_nonsig['PID'].as_matrix()==pid,'Xid']=pp


	### change X to be indicator flatted
	Xflat_sig = X_sig.copy()
	for col in range(np.shape(Xflat_sig)[1]):
	    Xflat_sig[:,col]=Xflat_sig[:,col]*(col+1)
	Xflat_sig = Xflat_sig.sum(1)

	Xflat_nonsig = X_nonsig.copy()
	for col in range(np.shape(Xflat_nonsig)[1]):
	    Xflat_nonsig[:,col]=Xflat_nonsig[:,col]*(col+1)
	Xflat_nonsig = Xflat_nonsig.sum(1)

	# re-label the paper ideas so they count from 1-50
	for pid in df_50_sig['PID'].unique():
	    xid = df_50_sig.loc[df_50_sig['PID']==pid,'Xid'].as_matrix()[0]
	    df_50_sig.loc[df_50_sig['PID']==pid,'XID2']=xid
	    df_50_nonsig.loc[df_50_nonsig['PID']==pid,'XID2']=xid




	out['df_50']=df_50
	out['df_50_nonsig'] = df_50_nonsig
	out['df_50_sig'] = df_50_sig
	out['X_sig']=X_sig
	out['X_nonsig']=X_nonsig
	out['data_simple_one_region']=data_simple_one_region
	out['R1'] = R1
	out['R1names'] = R1names

	return(out)



def collapse_study_data(df_50):

	# either put in DF 50_sig or not

	# for PID in PID
	pids = df_50.PID.unique()
	for pidn,pid in enumerate(pids):
		dftmp = df_50.loc[df_50['PID']==pid,:]
		avg_acc = dftmp['Accuracy.50'].mean() #get mean accuracy
		dftmp = pd.DataFrame(dftmp.iloc[0,:]) # get first row.
		dftmp = dftmp.transpose()
		# to-do: if non unique other values (e.g. info) put nan

		if pidn==0:
			df = dftmp
		else:
			df = pd.concat((df,dftmp))


	return(df)
