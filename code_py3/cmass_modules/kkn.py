#!/usr/bin/env python
import sys
from time import time
import os
import numpy as np
from cmass_modules import io, DES_to_SDSS, im3shape, Cuts

class knn(object):





    def _SDSS_cmass_criteria(self, sdss):
        
        modelmag_r = sdss['MODELMAG_R'] - sdss['EXTINCTION_R']
        modelmag_i = sdss['MODELMAG_I'] - sdss['EXTINCTION_I']
        modelmag_g = sdss['MODELMAG_G'] - sdss['EXTINCTION_G']
        cmodelmag_i = sdss['CMODELMAG_I'] - sdss['EXTINCTION_I']
        
        dperp = (modelmag_r - modelmag_i) - (modelmag_g - modelmag_r)/8.0
        fib2mag = sdss['FIBER2MAG_I']
        cmass = ((dperp > 0.55) &
                 (cmodelmag_i < (19.86 + 1.6*(dperp - 0.8))) &
                 (cmodelmag_i > 17.5 ) &
                 (cmodelmag_i < 19.9) &
                 ((modelmag_r - modelmag_i ) < 2.0 )&
                 (fib2mag < 21.5 )  #&
                 #(modelmag_i < 21.3)
                 )
        star = (((sdss['PSFMAG_I'] - sdss['EXPMAG_I']) > (0.2 + 0.2*(20.0 - sdss['EXPMAG_I']))) &
                ((sdss['PSFMAG_Z'] - sdss['EXPMAG_Z']) > (9.125 - 0.46 * sdss['EXPMAG_Z'])))
                 
        return cmass & star #sdss[cmass]
    
    def _arrange_data_for_fitting(self, data, tags=None):
        x = np.zeros((len(data),len(tags)))
        for i,tag in enumerate(tags):
            x[:,i] = data[tag]
        return x


    def classifier(self, sdss, des, des_tags=None, sdss_tags = None, train=None, train_size=60000):

        print('after prior cut (s/d) :', len(sdss), len(des))
        print('train_size ', train_size)
        """
        dperp_sdss = ( sdss['MODELMAG_R'] - sdss['MODELMAG_I']) - (sdss['MODELMAG_G'] - sdss['MODELMAG_R'])/8.
        sdss_cuts = ( ( dperp_sdss > 0.55 ) &
                     (sdss['CMODELMAG_I'] < (19.86 + 1.6*(dperp_sdss - 0.8))) &
                     (sdss['CMODELMAG_I'] < 19.9) &
                     (sdss['CMODELMAG_I'] > 17.5) &
                     (sdss['FIBER2MAG_I'] < 21.5) &
                     ((sdss['MODELMAG_R'] - sdss['MODELMAG_I']) < 2.) )
        """
        sdss_matched, des_matched = sdss, des
        
        #sdss_matched, des_matched = DES_to_SDSS.match(sdss, des)
        #sdss, des = DES_to_SDSS.match(sdss, des)
        #sdss_cuts = SDSS_cmass_criteria(sdss)
        #sdss, des = DES_to_SDSS.match(sdss, des)
        #if train is None:

        sdss_cuts = self._SDSS_cmass_criteria(sdss_matched)
        train = np.random.choice(np.arange(des_matched.size), size=train_size, replace=False)
        train_mask = np.zeros(des_matched.size, dtype=bool)
        train_mask[train] = 1
        test = np.where(~train_mask)[0]
        test_size = np.sum((~train_mask))

        x = self._arrange_data_for_fitting(des_matched[train],tags=des_tags)
        y = np.zeros(train_size)
        y[sdss_cuts[train]] = 1

        x_test = self._arrange_data_for_fitting(des_matched[test],tags=des_tags)
        y_test = np.zeros(test_size)
        y_test[sdss_cuts[test]] = 1
    
        #x_all = self._arrange_data_for_fitting(des,tags=des_tags)
        #y_all = np.zeros(des.size)
        #y_all[sdss_cuts] = 1

        """
        elif train is not None:
            
            train_sample = train.copy()
            sdss_matched, train_sample = DES_to_SDSS.match(sdss, train_sample)
            sdss_cuts = self._SDSS_cmass_criteria(sdss_matched)
            
            train = np.random.choice(np.arange(train_sample.size), size=train_size, replace=False)
            train_mask = np.zeros(train_sample.size, dtype=bool)
            train_mask[train] = 1
            test = np.where(~train_mask)[0]
            test_size = np.sum((~train_mask))

            x = self_.arrange_data_for_fitting(train_sample[train],tags=des_tags)
            y = np.zeros(train_size)
            y[sdss_cuts[train]] = 1
        
            x_test = self_.arrange_data_for_fitting(train_sample[test],tags=des_tags)
            y_test = np.zeros(test_size)
            y_test[sdss_cuts[test]] = 1

            #x_all = self_.arrange_data_for_fitting(des,tags=des_tags)
            #y_all = np.zeros(des.size)
            #y_all[sdss_cuts] = 1
        """
        
        print('train set ', np.sum(train_mask), ' test set ', np.sum((~train_mask)))
        print('cmass/train', np.sum(sdss_cuts[train]), ' cmass/test', np.sum(sdss_cuts[test]), ' total', np.sum(sdss_cuts))


        #from sklearn.ensemble import RandomForestClassifier as rfc
        #from sklearn.ensemble import AdaBoostClassifier as rfc
        #from sklearn.ensemble import GradientBoostingClassifier as rfc
        #pl = rfc(n_estimators=1000)
        from sklearn.neighbors import KNeighborsClassifier as kNN
        
        n_neighbors = 100 #   int(train_size * 0.02)
        print('n_neighbors', n_neighbors)
        pl = kNN(n_neighbors=n_neighbors,weights='distance',p=2,n_jobs=-1)
        pl.fit(x,y)
        predict_test = pl.predict(x_test)
        truth_test = y_test == 1
        
        predict_test_all = pl.predict(x_all)
        #truth_test_all = y_all == 1
        
        print("Classifier completeness:", np.sum(predict_test * truth_test) *1. / np.sum(truth_test))
        print("Classifier purity:", np.sum(predict_test * truth_test) * 1./np.sum(predict_test))
        print("number (test/all)", np.sum(predict_test), np.sum(predict_test_all))


        # Now reverse it, and see what SDSS galaxies we can be sure are selected by the classifier.

        #sdss_matched, des_matched = DES_to_SDSS.match(sdss, des)
        x = arrange_data_for_fitting(sdss[train],tags=sdss_tags)
        y = np.zeros(sdss.size)
        y[predict_test_all[train] == 1] = 1
        
        x_test = arrange_data_for_fitting(sdss_matched[test],tags=sdss_tags)
        y_test[predict_test == 1] = 1

        x_all = arrange_data_for_fitting(sdss,tags=sdss_tags)
        y_all[predict_test_all == 1] = 1
        
        pl2 = kNN(n_neighbors=n_neighbors,weights='distance',p=2,n_jobs=-1)
        pl2.fit(x,y)
        predict_rev = pl2.predict(x_test)
        good = (predict_rev ==0) & (predict_test == 1)
        
        predict_rev_all = pl2.predict(x_all)
        good = (predict_rev_all ==0) & (predict_test_all == 1)
        
        print("Reverse classifier completeness:", np.sum(predict_rev * predict_test ) *1. / np.sum(predict_test))
        print("Reverse classifier purity:", np.sum(predict_rev * predict_test) * 1./np.sum(predict_rev))
        
        return pl, (predict_test_all == 1), (predict_test_all == 1)