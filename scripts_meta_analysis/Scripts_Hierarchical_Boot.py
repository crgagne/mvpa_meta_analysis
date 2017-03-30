import numpy as np
import statsmodels as sm


####### Kernel density Estimate  ########### 

def kde_scipy(x, x_grid, bw_method,**kwargs):

    from sklearn.neighbors import KernelDensity
    from scipy.stats import gaussian_kde

    """Kernel Density Estimation with Scipy"""
    # Note that scipy weights its bandwidth by the covariance of the
    # input data.  To make the results comparable to the other methods,
    # we divide the bandwidth by the sample standard deviation here.
    kde = gaussian_kde(x, bw_method=bw_method, **kwargs)
    return kde.evaluate(x_grid)


def hierarchical_boot(df,savetitle,save=True,num_boots=2000):


    # pre-allocate 
    meann = np.array([])
    mediann = np.array([])
    meann_study = np.array([])

    prob_less_than_thresh1 = np.array([])
    prob_less_than_thresh2 = np.array([])

    x = np.linspace(0,1,100) # what range for cdf # 
    epcdf_store = np.empty((num_boots,len(x)))
    eppdf_store= np.empty((num_boots,len(x)))

    prob_less_than_62 = np.array([])
    prob_less_than_perc95 = np.array([])
    prob_less_than_mean_max = np.array([])
    mean_maxes = np.array([])
    perc95 = np.array([])
    perc75 = np.array([])
    perc25 = np.array([])

    for boot in range(num_boots):

        # sample studies with replacement
        studies_boot = np.random.choice(df['PID'].unique(),len(df['PID'].unique()),replace=True)
        study_accs_boot = np.array([])
        maxes = []
        means_study = []

        # sample accuracies within each new study 
        for pid in studies_boot:
             study_acc_b_single = np.random.choice(df.loc[df['PID']==pid,'Accuracy.50'],
                                                   size=len(df.loc[df['PID']==pid,'Accuracy.50']),replace=True)
             study_accs_boot= np.append(study_accs_boot,study_acc_b_single)
             maxes.append(np.max(study_acc_b_single)) # grab the max for each study for nonsig
             means_study.append(np.mean(study_acc_b_single))
        # store empirical cdf for each boot 
        ecdf = sm.distributions.empirical_distribution.ECDF(study_accs_boot)
        epcdf_store[boot,:]=ecdf(x)

        # store pdf for each boot
        kd = kde_scipy(study_accs_boot,x,bw_method='scott')
        eppdf_store[boot,:]=kd

        # store mean of maxes within each study
        maxes = np.array(maxes)
        mean_maxes = np.append(mean_maxes,maxes.mean())

        # store the 95% percentile 
        perc95 = np.append(perc95,np.percentile(study_accs_boot,95))
	perc75 = np.append(perc75,np.percentile(study_accs_boot,75))
	perc25 = np.append(perc25,np.percentile(study_accs_boot,25))

        meann = np.append(meann,np.mean(study_accs_boot))
        mediann = np.append(mediann,np.median(study_accs_boot))
        
        # study means 
        meann_study = np.append(meann_study,np.mean(means_study))


        prob_less_than_62 = np.append(prob_less_than_62,np.sum(study_accs_boot<.62)/float(len(study_accs_boot)))
        prob_less_than_perc95 = np.append(prob_less_than_perc95,np.sum(study_accs_boot<perc95[boot])/float(len(study_accs_boot)))
        prob_less_than_mean_max = np.append(prob_less_than_mean_max,np.sum(study_accs_boot<mean_maxes[boot])/float(len(study_accs_boot)))

    if save:
        
        np.savez('../data_meta_analysis/data_derived_meta_analysis/bootstrap_results_'+savetitle,
                 epcdf_store=epcdf_store,
                   eppdf_store= eppdf_store,
                  meann= meann,
                 mediann = mediann,
                 meann_study=meann_study,
                 prob_less_than_62 =prob_less_than_62,
                 perc95=perc95,
		 perc75=perc75,
		 perc25=perc25,
                 mean_max=mean_maxes)
    out = {}
    out['mean'] = meann
    out['perc95']=perc95
    return(out)
