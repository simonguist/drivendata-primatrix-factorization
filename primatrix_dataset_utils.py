'''
-------------------------------------------------------------------------------------------------------------------------------
Original source: Drivendata, https://gist.github.com/drivendata/70638e8a9e6a10fa020623f143259df3

Modifications:
    - Added new dataset type: small (224x224)
    - Changed num_frames
    - Implemented oversampling options: Animal oversampling and frequencies oversampling
    - Implemented seperate functions for getting indices of next batch and loading batch from given indexes:
        - batches_with_oversampling_get_indices & batches_by_indices
        - val_batches_get_indices & val_batches_by_indices
        - test_batches_get_indices & test_batches_by_indices
    - Added function convert_videos (convert raw videos to small size)
    - Added function update_predictions_at
-------------------------------------------------------------------------------------------------------------------------------
'''

from itertools import islice, cycle
import numpy as np
import os
import pandas as pd
import skvideo.io as skv
import random

from warnings import filterwarnings
filterwarnings('ignore')

from multilabel import multilabel_train_test_split

from scipy.misc import imresize

class Dataset(object):
    
    def __init__(self, datapath, dataset_type='nano', reduce_frames=True, non_blank_oversampling = False, val_size=0.3, batch_size=16, test=False):
        
        self.non_blank_oversampling = non_blank_oversampling
        self.datapath = datapath
        self.dataset_type = dataset_type
        self.reduce_frames = reduce_frames
        self.val_size = val_size
        self.batch_size = batch_size
        
        # boolean for test mode
        self.test = test
        
        # params based on dataset type
        if self.dataset_type == 'nano':
            self.height = 16
            self.width = 16
        elif self.dataset_type == 'micro':
            self.height = 64
            self.width = 64
        elif self.dataset_type == 'small':
            self.height = 224
            self.width = 224
        elif self.dataset_type == 'raw':
            print("\nRaw videos have variable size... \nsetting height and width to None... \nfirst video in test will determine size (test must be True ")
            self.height = None
            self.weidth = None
        else:
            raise NotImplementedError("Please set dataset_type as raw, small, micro, or nano.")
            
        # params based on frame reduction
        if self.reduce_frames:
            self.num_frames = 180
        else:
            self.num_frames = 360
        
        # for tracking errors
        self.bad_videos = []
        
        # training and validation
        self.X_train, self.X_val, self.y_train, self.y_val = self.split_training_into_validation()
    
        # params of data based on training data
        self.num_classes = self.y_train.shape[1]
        self.class_names = self.y_train.columns.values
        assert self.num_classes == self.y_val.shape[1]
        self.num_samples = self.y_train.shape[0]
        self.num_batches = self.num_samples // self.batch_size
        
        # test paths and prediction matrix
        self.X_test_ids, self.predictions = self.prepare_test_data_and_prediction()
        
        # variables to make batch generating easier
        self.batch_idx = cycle(range(self.num_batches))
        self.batch_num = next(self.batch_idx)
        
        self.num_val_batches = self.y_val.shape[0] // self.batch_size
        self.val_batch_idx = cycle(range(self.num_val_batches))
        self.val_batch_num = next(self.val_batch_idx)
        
        self.num_test_samples = self.X_test_ids.shape[0]
        self.num_test_batches = self.num_test_samples // self.batch_size
        self.test_batch_idx = cycle(range(self.num_test_batches))
        self.test_batch_num = next(self.test_batch_idx)
    
        # for testing iterator in test_mode
        #self.train_data_seen = pd.DataFrame(data={'seen': 0}, index=self.y_train.index)
        
        # test the generator
        if test:
            self._test_batch_generator()
    
    def prepare_test_data_and_prediction(self):
        """
        Returns paths to test data indexed by subject_id 
        and preallocates prediction dataframe.
        """
        
        predpath = os.path.join(self.datapath, 'Pri-matrix_Factorization_-_Submission_Format.csv')
        predictions = pd.read_csv(predpath, index_col='filename')
        test_idx = predictions.index
        subjpath = os.path.join(self.datapath, self.dataset_type)
        #subject_ids = pd.read_csv(subjpath, index_col=0)
        subject_ids = pd.DataFrame(data=subjpath, columns=['filepath'], index=test_idx)
        for row in subject_ids.itertuples():
            subject_ids.loc[row.Index] = os.path.join(row.filepath, row.Index) 
        
        return test_idx, predictions
  
    
    def split_training_into_validation(self, animal_oversampling_factor = 3):
        """
        Uses the multilabel_train_test_split function 
        to maintain class distributions between train
        and validation sets.
        """

        datapath = self.datapath
        dataset_type = self.dataset_type
        val_size = self.val_size
        
        # load training labels
        labelpath = os.path.join(datapath, 'Pri-matrix_Factorization_-_Training_Set_Labels.csv')
        labels = pd.read_csv(labelpath, index_col='filename')
        
        # load subject labels (assumed to have same index as training labels)
        subjpath = os.path.join(datapath, dataset_type)
        #subject_ids = pd.read_csv(subjpath, index_col=0)
        subject_ids = pd.DataFrame(data=subjpath, columns=['filepath'], index=labels.index)
        for row in subject_ids.itertuples():
            subject_ids.loc[row.Index] = os.path.join(row.filepath, row.Index)       
        
        if val_size == 0:
            X_train = subject_ids
            y_train = labels
            X_val = subject_ids
            y_val = labels
        else:
            X_train, X_val, y_train, y_val = multilabel_train_test_split(subject_ids, labels, size=val_size, min_count=1, seed=0)
     
        
        if self.non_blank_oversampling:
            y_train_blank = y_train[y_train.blank==1]
            blank_indices = y_train_blank.index

            X_train_oversampled = pd.DataFrame(columns=X_train.columns)
            y_train_oversampled = pd.DataFrame(columns=y_train.columns)

            for i in range(animal_oversampling_factor):
                X_train_i = X_train
                y_train_i = y_train
                for j in range(animal_oversampling_factor):
                    if i != j:
                        y_train_i = y_train_i.drop(blank_indices[j::animal_oversampling_factor])
                        X_train_i = X_train_i.drop(blank_indices[j::animal_oversampling_factor])
                X_train_oversampled = pd.concat([X_train_oversampled, X_train_i])
                y_train_oversampled = pd.concat([y_train_oversampled, y_train_i])
            
            X_train = X_train_oversampled
            y_train = y_train_oversampled
            
            
        # check distribution is maintained
        dist_diff = (y_train.sum()/y_train.shape[0] - y_val.sum() / y_val.shape[0]).sum()
        #print(dist_diff)
        #assert np.isclose(dist_diff, 0, rtol=1e-04, atol=1e-02)
     
        return X_train, X_val, y_train, y_val
        
        
       
    def batches(self, verbose=False, blank_non_blank_classification = False):
        """This method yields the next batch of videos for training."""
        
        reduce_frames = self.reduce_frames
        batch_size = self.batch_size
        num_train = self.y_train.shape[0]
        
        
        
        while 1:
            # get videos
            start = self.batch_size*self.batch_num
            stop = self.batch_size*(self.batch_num + 1)
            
            # print batch ranges if testing
            if self.test:
                print("batch {batch_num}:\t{start} --> {stop}".format(batch_num=self.batch_num, start=start, stop=stop-1))
            
            x_paths = self.X_train.iloc[start:stop]
            x, failed = self._get_video_batch(x_paths, 
                                              reduce_frames=reduce_frames, 
                                              verbose=verbose)
            x_paths = x_paths.drop(failed)
            self.bad_videos += failed

            # get labels
            y = self.y_train.iloc[start:stop]
            y = y.drop(failed)

            # check match for labels and videos
            assert (x_paths.index==y.index).all()
            assert x.shape[0] == y.shape[0]
            
            if blank_non_blank_classification:
                y = y['blank']

            # report failures if verbose
            if len(failed) != 0 and verbose==True:
                print("\t\t\t*** ERROR FETCHING BATCH {batch_num}/{num_batches} ***".format(batch_num=self.batch_num, num_batches = self.num_batches))
                print("Dropped {n} videos:".format(n=len(failed)))
                for failure in failed:
                    print(failure)

            # increment batch number
            self.batch_num = next(self.batch_idx)
            
            # update dataframe of seen training indices for testing
            #self.train_data_seen.loc[y.index.values] += 1
            yield (x, y)
            
            
            
    def batches_with_oversampling_get_indices(self, class_not_skip_frequencies = np.ones(24)):
        
        def keep(c):
            while(1):
                class_current_value[c] += class_not_skip_frequencies[c]
                if class_current_value[c]>=1:
                    class_current_value[c] -=1
                    yield True
                else:
                    yield False
        
        
        batch_size = self.batch_size
        
        class_current_value = np.zeros(24)
        
        skip_epochs = 0
        
        self.position_num = 0
        
        while 1:
            n_found_examples = 0
            found_indices = []
            while n_found_examples<batch_size:
                if self.position_num >= self.num_samples:
                    self.position_num = 0
                    skip_epochs +=1
                    class_current_value = (class_not_skip_frequencies * skip_epochs) % 1
                #c = random.choice(self.y_train.iloc[self.position_num].nonzero()[0])
                c = self.y_train.iloc[self.position_num].nonzero()[0][0]
                if next(keep(c)):
                    found_indices.append(self.position_num)
                    #print(self.position_num, ":", c, "(", class_current_value[c], ")")
                    n_found_examples +=1
                self.position_num += 1
            
            yield found_indices
            
    
            
    def batches_by_indices(self, indices, verbose = False):
        x_paths = self.X_train.ix[indices]
            
        x, failed = self._get_video_batch(x_paths, 
                                              reduce_frames=self.reduce_frames, 
                                              verbose=verbose)
        x_paths = x_paths.drop(failed)
        self.bad_videos += failed

        # get labels
        y = self.y_train.ix[indices]
        y = y.drop(failed)

        # check match for labels and videos
        assert (x_paths.index==y.index).all()
        assert x.shape[0] == y.shape[0]

        # report failures if verbose
        if len(failed) != 0 and verbose==True:
            print("\t\t\t*** ERROR FETCHING BATCH {batch_num}/{num_batches} ***".format(batch_num=self.batch_num, num_batches = self.num_batches))
            print("Dropped {n} videos:".format(n=len(failed)))
            for failure in failed:
                print(failure)
            
        # update dataframe of seen training indices for testing
        #self.train_data_seen.loc[y.index.values] = 1
        return (x, y)
    
    
            
    def val_batches(self, verbose=False):
        """This method yields the next batch of videos for validation."""
        
        reduce_frames = self.reduce_frames
        batch_size = self.batch_size
        num_val = self.y_val.shape[0]
        
        
        
        while 1:
            # get videos
            start = self.batch_size*self.val_batch_num
            stop = self.batch_size*(self.val_batch_num + 1)
            
            x_paths = self.X_val.iloc[start:stop]
            x, failed = self._get_video_batch(x_paths, 
                                              reduce_frames=reduce_frames, 
                                              verbose=verbose)
            x_paths = x_paths.drop(failed)
            self.bad_videos += failed

            # get labels
            y = self.y_val.iloc[start:stop]
            y = y.drop(failed)

            # check match for labels and videos
            assert (x_paths.index==y.index).all()
            assert x.shape[0] == y.shape[0]

            # report failures if verbose
            if len(failed) != 0 and verbose==True:
                print("\t\t\t*** ERROR FETCHING BATCH {batch_num}/{num_batches} ***".format(batch_num=self.batch_num, num_batches = self.num_batches))
                print("Dropped {n} videos:".format(n=len(failed)))
                for failure in failed:
                    print(failure)

            # increment batch number
            self.val_batch_num = next(self.val_batch_idx)
            
            yield (x, y)
            
            
            
    def val_batches_get_indices(self, verbose=False):
        """This method yields the next batch of videos for validation."""
        
        batch_size = self.batch_size
        num_val = self.y_val.shape[0]
        
        while 1:
            # get videos
            start = self.batch_size*self.val_batch_num
            stop = self.batch_size*(self.val_batch_num + 1)
            
            # increment batch number
            self.val_batch_num = next(self.val_batch_idx)
            
            yield np.asarray(range(start,stop))
            
            
    def val_batches_by_indices(self, indices, verbose=False):
        """This method yields the next batch of videos for validation."""
        
        reduce_frames = self.reduce_frames
        batch_size = self.batch_size
        num_val = self.y_val.shape[0]
        
        x_paths = self.X_val.iloc[indices]
        x, failed = self._get_video_batch(x_paths, 
                                          reduce_frames=reduce_frames, 
                                          verbose=verbose)
        x_paths = x_paths.drop(failed)
        self.bad_videos += failed

        # get labels
        y = self.y_val.iloc[indices]
        y = y.drop(failed)

        # check match for labels and videos
        assert (x_paths.index==y.index).all()
        assert x.shape[0] == y.shape[0]

        # report failures if verbose
        if len(failed) != 0 and verbose==True:
            print("\t\t\t*** ERROR FETCHING BATCH {batch_num}/{num_batches} ***".format(batch_num=self.batch_num, num_batches = self.num_batches))
            print("Dropped {n} videos:".format(n=len(failed)))
            for failure in failed:
                print(failure)

        # increment batch number
        self.val_batch_num = next(self.val_batch_idx)

        return (x, y)    
    
    
            
    def test_batches(self, verbose=False):
        """This method yields the next batch of videos for testing."""
        
        reduce_frames = self.reduce_frames
        batch_size = self.batch_size
        num_test = self.num_test_samples
        
        test_dir = os.path.join(self.datapath, self.dataset_type)
        
        
        while 1:
            # get videos
            start = self.batch_size*self.test_batch_num
            stop = self.batch_size*(self.test_batch_num + 1)
            
            x_ids = self.X_test_ids[start:stop]
            x_paths = pd.DataFrame(data=[os.path.join(test_dir, filename) for filename in x_ids], 
                                   columns=['filepath'],
                                   index=x_ids)
            #print(x_paths)
            x, failed = self._get_video_batch(x_paths, 
                                              reduce_frames=reduce_frames, 
                                              verbose=verbose)
            
            self.test_batch_ids = x_ids.values

            # increment batch number
            self.test_batch_num = next(self.test_batch_idx)
            
            yield x

            
    def test_batches_get_indices(self, verbose=False):
        """This method yields the next batch of videos for testing."""
        
        batch_size = self.batch_size
        
        test_dir = os.path.join(self.datapath, self.dataset_type)
        self.pos_test_by_ind = 0
        
        while 1:
            ind_list = []
            for i in range(batch_size):
                ind_list.append(self.pos_test_by_ind)
                self.pos_test_by_ind += 1
                if self.pos_test_by_ind == self.num_test_samples:
                    self.pos_test_by_ind = 0
                    
            yield ind_list
            
            
    def test_batches_by_indices(self, ind_list, verbose=False):
        
        x_ids = self.X_test_ids[ind_list]
        
        reduce_frames = self.reduce_frames
        batch_size = self.batch_size
        num_test = self.num_test_samples
        test_dir = os.path.join(self.datapath, self.dataset_type)
        
        x_paths = pd.DataFrame(data=[os.path.join(test_dir, filename) for filename in x_ids], 
                                   columns=['filepath'],
                                   index=x_ids)
        x, failed = self._get_video_batch(x_paths, 
                                              reduce_frames=reduce_frames, 
                                              verbose=verbose)
        return x
        
    def _get_video_batch(self, x_paths, as_grey=False, reduce_frames=True, verbose=False):
        """
        Returns ndarray of shape (batch_size, num_frames, width, height, channels).
        If as_grey, then channels dimension is squeezed out.
        """

        videos = []
        failed = []
        
        for row in x_paths.itertuples():
            filepath = row.filepath
            obf_id = row.Index
            
            try:
                # load
                video = skv.vread(filepath, as_grey=as_grey)
            except:
                try:
                    # try again
                    print("trying again to load id:\t", obf_id)        
                    video = skv.vread(filepath, as_grey=as_grey)        
                except:
                    if verbose:
                            print("FAILED TO LOAD:", filepath)
                    print("faild to load id:\t", obf_id)
                    failed.append(obf_id)

                    first_row = False
                    x_paths_removed_fault = x_paths.drop(row.Index)
                    x_paths_removed_fault = pd.concat([x_paths_removed_fault, \
                                              x_paths_removed_fault.ix[np.random.random_integers(0, self.batch_size -2, 1)]])
                    return self._get_video_batch(x_paths_removed_fault, as_grey, reduce_frames, verbose)
                
            
            # fill video if neccessary
            if video.shape[0] < self.num_frames:
                video = self._fill_video(video) 
            
            # reduce
            if reduce_frames:
                step_size = int(video.shape[0] / 90)
                frames = np.arange(random.randint(0,step_size-1), video.shape[0], step_size)
                try:
                    video = video[frames, :, :] #.squeeze()  
                    videos.append(video)
                
                except IndexError:
                    if verbose:
                        print("FAILED TO REDUCE:", filepath)
                    print("failed to reduce id:\t", obf_id)
                    failed.append(obf_id)
                       
        return np.array(videos), failed
 
    
    def convert_videos(self, square_size, save_folder, reduce_frames=True):
        
        x_paths = self.get_paths()
        failed = []
        
        for row in x_paths.itertuples():
            filepath = row.filepath
            obf_id = row.Index
            
            # load
            video = skv.vread(filepath, as_grey=False)
            
            self.height = video.shape[1]
            self.width = video.shape[2]
            
            # fill video if neccessary
            if video.shape[0] < self.num_frames:
                video = self._fill_video(video) 
            
            # reduce
            if reduce_frames:
                frames = np.arange(0, video.shape[0], 2)
                try:
                    video = video[frames, :, :] #.squeeze() 
                    
                except IndexError:
                    if verbose:
                        print("FAILED TO REDUCE:", filepath)
                    print("id:\t", obf_id)
                    failed.append(obf_id)
                    
            #reshape
            video = np.asarray([imresize(x, (224, 224, 3), interp='bilinear') for x in video])
            
            #save
            save_path = os.path.join(save_folder, obf_id)
            skv.vwrite(save_path, video)
           
        return failed
    
    def get_paths(self):
        predpath = os.path.join(self.datapath, 'Pri-matrix_Factorization_-_Submission_Format.csv')
        predictions = pd.read_csv(predpath, index_col='filename')
        test_idx = predictions.index
        subjpath = os.path.join(self.datapath, self.dataset_type)
        #subject_ids = pd.read_csv(subjpath, index_col=0)
        subject_ids = pd.DataFrame(data=subjpath, columns=['filepath'], index=test_idx)
        for row in subject_ids.itertuples():
            subject_ids.loc[row.Index] = os.path.join(row.filepath, row.Index)
        return subject_ids
    

    
    def _fill_video(self, video):
        """Returns a video with self.num_frames given at least one frame."""

        # establish boundaries
        target_num_frames = self.num_frames
        num_to_fill = target_num_frames - video.shape[0]

        # preallocate array for filler
        filler_frames = np.zeros(shape=(num_to_fill, self.width, self.height, 3)) # assumes color

        # fill frames
        source_frame = cycle(np.arange(0, video.shape[0]))
        for i in range(num_to_fill):
            filler_frames[i, :, :] = video[next(source_frame), :, :]

        return np.concatenate((video, filler_frames), axis=0)
    

    def _test_batch_generator(self):
        
        print('Testing train batch generation...')
        
        for i in range(self.num_batches):
            if self.batch_num % 10 == 0:
                print("\n\t\t\tBATCH \t{self.batch_num}/{self.num_batches}\n".format(batch_num=self.batch_num, num_batches = self.num_batches))      
            
            batch = self.batches(verbose=True)
            x,y = next(batch)
        
            # same batches for videos and labels
            assert x.shape[0] == y.shape[0]
            
            # square videos
            assert x.shape[2] == x.shape[3]
            
            # black and white
            assert x.shape[4] == 1
            
        
        # assert we've seen all data up to remainder of a batch
        assert (self.y_train.shape[0] - self.train_data_seen.sum().values[0]) < self.batch_size
        
        # check that batch_num is reset
        assert self.batch_num == 0
        
        # turn off test mode
        if self.test == True:
            self.test = False
        
        print('Test passed.')
        
    def update_predictions(self, results):
        self.predictions.loc[self.test_batch_ids] = results
        
        
    def update_predictions_at(self, ids, results):
        self.predictions.loc[ids] = results
