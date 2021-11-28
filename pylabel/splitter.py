import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit as sklearnGroupShuffleSplit

class Split():
    def  __init__(self, dataset=None):
        self.dataset = dataset

    def UnSplit(self):
        """Unsplit the dataset by setting all values of the split column to null."""
        self.dataset.df["split"] = np.nan

    def GroupShuffleSplit(df_main, train_pct=.6, test_pct=.25, val_pct=.25, group_col = 'img_filename', random_state=None):
        """
        This function uses the GroupShuffleSplit command from sklearn. It can split into 3 groups (train,
        test, and val) by applying the command twice. 
        """

        gss = sklearnGroupShuffleSplit(n_splits=1, train_size=train_pct)
        train_indexes, test_indexes = next(gss.split(X=df_main, y=df_main[group_col], groups=df_main.index.values))

        df_main.loc[train_indexes,'split'] = "train"
        df_main.loc[test_indexes,'split'] = "test"

        if val_pct:
            df_train = df_main.loc[df_main['split'] == 'train']
            df_test = df_main.loc[df_main['split'] == 'test']
            df_test = df_test.reset_index()
            second_split_pct = float(test_pct/(test_pct+val_pct))
            gss2 = sklearnGroupShuffleSplit(n_splits=1, train_size=second_split_pct)
            test_indexes_2, val_indexes_2 = next(gss2.split(X=df_test, y=df_test[group_col], groups=df_test.index.values))
            df_test.loc[test_indexes_2,'split'] = "test"
            df_test.loc[val_indexes_2,'split'] = "val"
            return df_train.append(df_test)

        else:
            return df_main

    #Written with the help of https://stackoverflow.com/questions/56872664/complex-dataset-split-stratifiedgroupshufflesplit 
    def StratifiedGroupShuffleSplit(df_main, train_pct=.7, test_pct=.3, val_pct=.0, weight=0.01, 
        group_col = 'img_filename', cat_col = 'cat_name', batch_size=1):
        """
        This function will 'split" the dataframe by setting the split collumn equal to 
        train, test, or val. When a split dataset is exported the annotations will be split into
        seperate groups so that can be used used in model training, testing, and validation.
        """

        df_main = df_main.reindex(np.random.permutation(df_main.index)) # shuffle dataset

        # create empty train, val and test datasets
        df_train = pd.DataFrame()
        df_val = pd.DataFrame()
        df_test = pd.DataFrame()

        subject_grouped_df_main = df_main.groupby([group_col], sort=False, as_index=False)
        category_grouped_df_main = df_main.groupby(cat_col).count()[[group_col]]/len(df_main)*100

        #Check inputs 
        assert (0 <= weight <= 1), "Weight must be between 0 and 1"
        total_splits = round((train_pct) + float(test_pct) + float(val_pct),1)
        assert (total_splits == 1), "Sum of train_pct, test_pct, and val_pct must equal 1."
        assert (batch_size >= 1 and batch_size <= subject_grouped_df_main.ngroups / 10 ), \
            "Batch must be greater than 1 and less than 1/10 count of groups"

        def calc_mse_loss(df):
            grouped_df = df.groupby(cat_col).count()[[group_col]]/len(df)*100
            df_temp = category_grouped_df_main.join(grouped_df, on = cat_col, how = 'left', lsuffix = '_main')
            df_temp.fillna(0, inplace=True)
            df_temp['diff'] = (df_temp['img_filename_main'] - df_temp[group_col])**2
            mse_loss = np.mean(df_temp['diff'])
            return mse_loss

        i = 0 #counter for all items in dataset
        b = 0 #counter for the batches
        batch_df = df_main[0:0]

        for _, group in subject_grouped_df_main:
            if (i < 3):
                if (i == 0):
                    df_train = df_train.append(pd.DataFrame(group), ignore_index=True)
                    i += 1
                    continue
                elif (i == 1):
                    df_val = df_val.append(pd.DataFrame(group), ignore_index=True)
                    i += 1
                    continue
                else:
                    df_test = df_test.append(pd.DataFrame(group), ignore_index=True)
                    i += 1
                    continue

            #Add groups to the 
            batch_df = batch_df.append(group)
            b += 1
            if b < batch_size and i < subject_grouped_df_main.ngroups-3:
                i += 1
                continue

            mse_loss_diff_train = calc_mse_loss(df_train) - calc_mse_loss(df_train.append(batch_df, ignore_index=True))
            mse_loss_diff_val = calc_mse_loss(df_val) - calc_mse_loss(df_val.append(batch_df, ignore_index=True))
            mse_loss_diff_test = calc_mse_loss(df_test) - calc_mse_loss(df_test.append(batch_df, ignore_index=True))

            total_records = len(df_train) + len(df_val) + len(df_test)

            len_diff_train = (train_pct - (len(df_train)/total_records))
            len_diff_val = (val_pct - (len(df_val)/total_records))
            len_diff_test = (test_pct - (len(df_test)/total_records)) 

            len_loss_diff_train = len_diff_train * abs(len_diff_train)
            len_loss_diff_val = len_diff_val * abs(len_diff_val)
            len_loss_diff_test = len_diff_test * abs(len_diff_test)

            loss_train = (weight * mse_loss_diff_train) + ((1-weight) * len_loss_diff_train)
            loss_val = (weight * mse_loss_diff_val) + ((1-weight) * len_loss_diff_val)
            loss_test = (weight * mse_loss_diff_test) + ((1-weight) * len_loss_diff_test)

            if (max(loss_train,loss_val,loss_test) == loss_train):
                df_train = df_train.append(batch_df, ignore_index=True)
            elif (max(loss_train,loss_val,loss_test) == loss_val):
                df_val = df_val.append(batch_df, ignore_index=True)
            else:
                df_test = df_test.append(batch_df, ignore_index=True)

            #print ("Group " + str(i) + ". loss_train: " + str(loss_train) + " | " + "loss_val: " + str(loss_val) + " | " + "loss_test: " + str(loss_test) + " | ")
            i += 1
            #Reset the batch
            b = 0
            batch_df = df_main[0:0]


        ######
        # Final prep tasks before returning the split dataframe

        #Sometimes the algo will put some rows in the val set even if the split percent was set to zero
        #In those cases move the rows from val to test 
        if round(val_pct,1) == round(0,1):
            df_test.append(df_val)
            df_val = df_val[0:0] #remove the values from 

        #Apply train, split, val labels to the split collumn 
        df_train['split'] = 'train'
        df_test['split'] = 'test'
        df_val['split'] = 'val'

        df = df_train.append(df_test).append(df_val)
        
        assert df.shape == df_main.shape, "Output shape does not match input shape. Data loss has occured."

        return df