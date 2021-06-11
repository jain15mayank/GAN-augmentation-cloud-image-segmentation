import numpy as np
from sklearn.cross_decomposition import PLSRegression as plsr
from sklearn.model_selection import KFold
from PIL import Image
import shutil, os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed
from copy import copy


def train_pls(X, Y, n_comp):
    pls = plsr(n_components = n_comp)
    pls.fit(X,Y)
    return pls, n_comp

def find_optimal_nComp(X_train, Y_train, X_val=None, Y_val=None, nComp_min = 1, nComp_max = 40, n_splits = None, plot_save_dir=None):
    if X_val is not None and Y_val is not None: # Simple case - no k-fold cross-validation required
        scoreT = np.zeros(nComp_max)
        scoreV = np.zeros(nComp_max)
        results = Parallel(n_jobs=5)(delayed(train_pls)(X_train, Y_train, nComp) for nComp in range(nComp_min, nComp_max+1))
        for i in range(len(results)):
            scoreT[results[i][1]] = results[i][0].score(X_train, Y_train)
            scoreV[results[i][1]] = results[i][0].score(X_val, Y_val)
        #for nComp in range(nComp_min, nComp_max+1):
        #    pls = train_pls(X_train, Y_train, nComp)
        #    scoreT.append(pls.score(X_train, Y_train))
        #    scoreV.append(pls.score(X_val, Y_val))
        optimal_nComp = np.argmax(scoreV)
        if plot_save_dir is not None:
            # Plot the optimization plot
            plt.rc('font', size=19)         # controls default text sizes
            plt.rc('axes', titlesize=22)    # fontsize of the axes title
            plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
            plt.rc('xtick', labelsize=19)   # fontsize of the tick labels
            plt.rc('ytick', labelsize=19)   # fontsize of the tick labels
            plt.rc('legend', fontsize=20)   # legend fontsize
            plt.rc('figure', titlesize=22)  # fontsize of the figure title
            plt.figure(figsize=(10,6))
            plt.plot(np.arange(1, 41).tolist(), scoreT, label='Training Set')
            plt.plot(np.arange(1, 41).tolist(), scoreV, label='Validation Set')
            markX = np.argmax(scoreV) + 1
            markY = np.max(scoreV)
            if markX<16:
                annotX = 23
            elif markX<26:
                annotX = 27
            else:
                annotX = 10
            if markY<0.35:
                annotY = 0.4
            elif markY<0.6:
                annotY = markY+0.05
            else:
                annotY = 0.15
            if markX is not None and markY is not None:
                plt.hlines(markY, 0, markX, linestyles='dashed')
                plt.vlines(markX, 0, markY, linestyles='dashed')
                plt.scatter([markX],[markY], c='r')
                if annotX is not None and annotY is not None:
                    if annotY>markY:
                        plt.annotate('Max Score on\nValidation Set\n'+str((markX, np.around(markY,3))),
                                     xy=(markX, markY+0.002), xytext=(annotX, annotY),
                                     arrowprops=dict(arrowstyle="->",
                                     connectionstyle="angle3,angleA=0,angleB=-90"));
                    else:
                        plt.annotate('Max Score on\nValidation Set\n'+str((markX, np.around(markY,3))),
                                     xy=(markX, markY-0.002), xytext=(annotX, annotY),
                                     arrowprops=dict(arrowstyle="->",
                                     connectionstyle="angle3,angleA=0,angleB=-90"));
            plt.xlabel("Number of PLS Components")
            plt.ylabel("Coeff. of Determination (R^2)")
            plt.legend()
            low = min(scoreT + scoreV)
            high = max(scoreT + scoreV)
            plt.xlim([0, 41])
            plt.ylim([max(0,low-0.8*(high-low)), min(high+0.4*high, high+0.4*(high-low))])
            plt.savefig(plot_save_dir+'optimizationPLSnComp.png', bbox_inches = 'tight', pad_inches = 0.05)
            plt.savefig(plot_save_dir+'optimizationPLSnComp.pdf', bbox_inches = 'tight', pad_inches = 0.05)
            plt.close()
    
    elif X_val is None and Y_val is None: # k-fold cross-validation is required
        if n_splits is None:
            raise('k-fold cross-validation is sought but the argument to define number of splits/folds is not passed!')
        kf = KFold(n_splits=n_splits)
        n_splits = kf.get_n_splits(X_train, Y_train)
        split_count = 0
        optimal_nComp = []
        scoreT = []
        scoreV = []
        for train_index, val_index in kf.split(X_train, Y_train):
            split_count+=1
            #print("TRAIN:", train_index, "VAL:", val_index)
            X_train_kFold = [X_train[i] for i in train_index]
            X_val_kFold = [X_train[i] for i in val_index]
            Y_train_kFold = [Y_train[i] for i in train_index]
            Y_val_kFold = [Y_train[i] for i in val_index]
            
            scoreT_kFold = np.zeros(nComp_max)
            scoreV_kFold = np.zeros(nComp_max)
            results = Parallel(n_jobs=5)(delayed(train_pls)(X_train_kFold, Y_train_kFold, nComp) for nComp in range(nComp_min, nComp_max+1))
            for i in range(len(results)):
                scoreT_kFold[results[i][1]] = results[i][0].score(X_train_kFold, Y_train_kFold)
                scoreV_kFold[results[i][1]] = results[i][0].score(X_val_kFold, Y_val_kFold)
            print('Training of PLS finished on current split!')
            '''
            scoreT_kFold = []
            scoreV_kFold = []
            for nComp in range(nComp_min, nComp_max+1):
                pls = train_pls(X_train_kFold, Y_train_kFold, nComp)
                scoreT_kFold.append(pls.score(X_train_kFold, Y_train_kFold))
                scoreV_kFold.append(pls.score(X_val_kFold, Y_val_kFold))
            '''
            optimal_nComp.append(np.argmax(scoreV_kFold))
            scoreT.append(scoreT_kFold)
            scoreV.append(scoreV_kFold)
            if plot_save_dir is not None:
                # Plot the optimization plot
                plt.rc('font', size=19)         # controls default text sizes
                plt.rc('axes', titlesize=22)    # fontsize of the axes title
                plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
                plt.rc('xtick', labelsize=19)   # fontsize of the tick labels
                plt.rc('ytick', labelsize=19)   # fontsize of the tick labels
                plt.rc('legend', fontsize=20)   # legend fontsize
                plt.rc('figure', titlesize=22)  # fontsize of the figure title
                plt.figure(figsize=(10,6))
                plt.plot(np.arange(1, 41).tolist(), scoreT_kFold, label='Training Set')
                plt.plot(np.arange(1, 41).tolist(), scoreV_kFold, label='Validation Set')
                markX = np.argmax(scoreV_kFold) + 1
                markY = np.max(scoreV_kFold)
                if markX<16:
                    annotX = 23
                elif markX<26:
                    annotX = 27
                else:
                    annotX = 10
                if markY<0.35:
                    annotY = 0.4
                elif markY<0.6:
                    annotY = markY+0.05
                else:
                    annotY = 0.15
                if markX is not None and markY is not None:
                    plt.hlines(markY, 0, markX, linestyles='dashed')
                    plt.vlines(markX, 0, markY, linestyles='dashed')
                    plt.scatter([markX],[markY], c='r')
                    if annotX is not None and annotY is not None:
                        if annotY>markY:
                            plt.annotate('Max Score on\nValidation Set\n'+str((markX, np.around(markY,3))),
                                         xy=(markX, markY+0.002), xytext=(annotX, annotY),
                                         arrowprops=dict(arrowstyle="->",
                                         connectionstyle="angle3,angleA=0,angleB=-90"));
                        else:
                            plt.annotate('Max Score on\nValidation Set\n'+str((markX, np.around(markY,3))),
                                         xy=(markX, markY-0.002), xytext=(annotX, annotY),
                                         arrowprops=dict(arrowstyle="->",
                                         connectionstyle="angle3,angleA=0,angleB=-90"));
                plt.xlabel("Number of PLS Components")
                plt.ylabel("Coeff. of Determination (R^2)")
                plt.legend()
                low = min(scoreT_kFold + scoreV_kFold)
                high = max(scoreT_kFold + scoreV_kFold)
                plt.xlim([0, 41])
                plt.ylim([max(0,low-0.8*(high-low)), min(high+0.4*high, high+0.4*(high-low))])
                plt.savefig(plot_save_dir+'split'+str(split_count)+'_optimizationPLSnComp.png', bbox_inches = 'tight', pad_inches = 0.05)
                plt.savefig(plot_save_dir+'split'+str(split_count)+'_optimizationPLSnComp.pdf', bbox_inches = 'tight', pad_inches = 0.05)
                plt.close()
            
        optimal_nComp = max(optimal_nComp)
    else:
        raise('Inappropriate number of argments passed to the function!')
    return scoreT, scoreV, optimal_nComp

def read_data(all_ori_source, all_target_source):
    file_list = [f for f in os.listdir(all_target_source) if os.path.isfile(os.path.join(all_target_source, f))]
    X = []
    Y = []
    imId = []
    for file in file_list:
        # Read and Store Target Image
        image_obj = Image.open(all_target_source + file).resize((100,100))
        image = np.asarray(image_obj)
        temp = image.copy()
        temp[temp<130] = 0
        temp[temp>=130] = 255
        Y.append(temp.flatten())
        # Read and Store Corresponding Input Image
        if file.find('_')!=-1:
            imageID = int(file.split('_')[0])
        else:
            imageID = int(file.split('.')[0])
        imId.append(imageID)
        image_obj = Image.open(all_ori_source + str(imageID) + '.png').resize((100,100))
        image = np.asarray(image_obj)
        X.append(image.flatten())
    return X, Y, imId

def genGANaugmentedErrors(X_train, Y_train, Im_Xgan, Im_Ygan, ganImID, X_val, Y_val):
    aug_pls, _ = train_pls(X_train+[Im_Xgan], Y_train+[Im_Ygan], n_comp = 8)
    trainError = aug_pls.score(X_train+[Im_Xgan], Y_train+[Im_Ygan])
    valError = aug_pls.score(X_val, Y_val)
    print("Done with GAN image id %d" % ganImID)
    return trainError, valError, ganImID

def save_images(input_images, image_ids, save_path):
    Path(save_path).mkdir(parents=True, exist_ok=True)
    for (idx,image) in enumerate(input_images):
        if image.dtype == 'uint8':
            im = Image.fromarray(flattened_to_image(image))
            im = im.convert("L")
            im.save(save_path + str(image_ids[idx]) + '.png')
        else:
            raise('Inputted image to be saved is not of type uint8')

def flattened_to_image(flat_image):
    if len(flat_image.shape)>1:
        if flat_image.shape[1]>1:
            raise('Inputted array is not in one-dimension. It was expected to be flattened!')
    if not int(np.sqrt(len(flat_image)))*int(np.sqrt(len(flat_image))) == len(flat_image):
        raise('The flattened input array length is not a perfect square. Cannot be turned back into a square image!')
    else:
        return np.reshape(flat_image, (int(np.sqrt(len(flat_image))), int(np.sqrt(len(flat_image)))))

def plsPredictions_to_images(pls_prediction, threshold=127):
    # Converts the PLS predicted values to uint8 image in accordance to the
    # specified threshold. Black = <=threshold; White= >threshold
    flatImgs = []
    for img in pls_prediction:
        flat_image = np.zeros(len(img)).astype('uint8')
        for i in range(len(img)):
            if img[i]>threshold:
                flat_image[i] = 255
        flatImgs.append(flat_image)
    return flatImgs

def compute_confusion(arrayTrue, arrayPredicted, negative=0, positive=255):
    if not len(arrayTrue) == len(arrayPredicted):
        raise("Entered arrays are not of equal length!")
    if len(np.unique(arrayTrue))>2 or len(np.unique(arrayPredicted))>2:
        print(np.unique(arrayTrue))
        print(np.unique(arrayPredicted))
        raise("Not a binary classification task!")
    if not np.array_equal(np.unique(arrayTrue), np.unique(arrayPredicted)):
        if len(np.unique(arrayTrue)) == len(np.unique(arrayPredicted)):
            raise("Different labels in true and predicted arrays!")
        elif len(np.unique(arrayTrue)) < len(np.unique(arrayPredicted)) and (not np.unique(arrayTrue)[0] == np.unique(arrayPredicted)[0]) and (not np.unique(arrayTrue)[0] == np.unique(arrayPredicted)[1]):
            raise("Different labels in true and predicted arrays!")
        elif len(np.unique(arrayTrue)) > len(np.unique(arrayPredicted)) and (not np.unique(arrayTrue)[0] == np.unique(arrayPredicted)[0]) and (not np.unique(arrayTrue)[1] == np.unique(arrayPredicted)[0]):
            raise("Different labels in true and predicted arrays!")
    tp = 0  # True Positives
    tn = 0  # True Negatives
    fp = 0  # False Positives
    fn = 0  # False Negatives
    for i in range(len(arrayTrue)):
        if arrayTrue[i]==arrayPredicted[i]:
            if arrayTrue[i]==negative:
                tn+=1
            else:
                tp+=1
        else:
            if arrayTrue[i]==negative:
                fp+=1
            else:
                fn+=1
    return tp, tn, fp, fn

def compute_precision(arrayTrue, arrayPredicted, negative=0, positive=255):
    tp, tn, fp, fn = compute_confusion(arrayTrue, arrayPredicted, negative, positive)
    if tp+fp == 0:
        return 0
    else:
        precision = tp/(tp+fp)
        return precision

def compute_recall(arrayTrue, arrayPredicted, negative=0, positive=255):
    tp, tn, fp, fn = compute_confusion(arrayTrue, arrayPredicted, negative, positive)
    if tp+fn == 0:
        return 0
    else:
        recall = tp/(tp+fn)
        return recall

def compute_fScore(arrayTrue, arrayPredicted, negative=0, positive=255):
    precision = compute_precision(arrayTrue, arrayPredicted, negative, positive)
    recall = compute_recall(arrayTrue, arrayPredicted, negative, positive)
    if (precision+recall)==0:
        return 0
    else:
        fScore = (2*(precision*recall))/(precision+recall)
        return fScore

def precision_recall_fScore_byThreshold(threshold, raw_prediction, ground_truth, negative=0, positive=255):
    predicted_image = plsPredictions_to_images([raw_prediction], threshold=threshold)[0]
    precision = compute_precision(ground_truth, predicted_image, negative=negative, positive=positive)
    recall = compute_recall(ground_truth, predicted_image, negative=negative, positive=positive)
    fScore = compute_fScore(ground_truth, predicted_image, negative=negative, positive=positive)
    return precision, recall, fScore



def generate_ROC(ori_raw_prediction, aug_raw_prediction, ground_truth, negative=0, positive=255):
    # 3 Rows - Threshold, True Positive Rate (TPR), False Positive Rate (FPR)
    ori_ROC = np.zeros((3, int(np.ceil(max(max(ori_raw_prediction), max(aug_raw_prediction)))) - int(np.floor(min(min(ori_raw_prediction), min(aug_raw_prediction))))))
    aug_ROC = np.zeros((3, int(np.ceil(max(max(ori_raw_prediction), max(aug_raw_prediction)))) - int(np.floor(min(min(ori_raw_prediction), min(aug_raw_prediction))))))
    
    for thr in range(int(np.floor(min(min(ori_raw_prediction), min(aug_raw_prediction)))), int(np.ceil(max(max(ori_raw_prediction), max(aug_raw_prediction))))):
        ori_predicted = plsPredictions_to_images([ori_raw_prediction], threshold=thr)[0]
        aug_predicted = plsPredictions_to_images([aug_raw_prediction], threshold=thr)[0]
        ori_tp, ori_tn, ori_fp, ori_fn = compute_confusion(ground_truth, ori_predicted, negative, positive)
        aug_tp, aug_tn, aug_fp, aug_fn = compute_confusion(ground_truth, aug_predicted, negative, positive)
        
        if ori_tp+ori_fn == 0:
            ori_TPR = 0
        else:
            ori_TPR = ori_tp/(ori_tp+ori_fn)
        if ori_fp+ori_tn == 0:
            ori_FPR = 0
        else:
            ori_FPR = ori_fp/(ori_fp+ori_tn)
        
        if aug_tp+aug_fn == 0:
            aug_TPR = 0
        else:
            aug_TPR = aug_tp/(aug_tp+aug_fn)
        if aug_fp+aug_tn == 0:
            aug_FPR = 0
        else:
            aug_FPR = aug_fp/(aug_fp+aug_tn)
        
        ori_ROC[0, thr] = thr
        ori_ROC[1, thr] = ori_TPR
        ori_ROC[2, thr] = ori_FPR
        
        aug_ROC[0, thr] = thr
        aug_ROC[1, thr] = aug_TPR
        aug_ROC[2, thr] = aug_FPR
    
    return ori_ROC, aug_ROC

def compute_auc(sorted_x, sorted_y):
    return np.trapz(np.concatenate((np.array([0]), sorted_y, np.array([1]))), np.concatenate((np.array([0]), sorted_x, np.array([1]))))

def create_roc_byIndex(imgID, savePath, ori_raw_prediction, aug_raw_prediction, ground_truth, negative=0, positive=255):
    ori_ROC, aug_ROC = generate_ROC(ori_raw_prediction, aug_raw_prediction, ground_truth, negative, positive)
    
    # Calculate the best threshold value for given image ID - original PLS prediction
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    temp1 = np.divide(ori_ROC[1].astype('float64'), 0.3+ori_ROC[2].astype('float64'))
    np.seterr(**old_settings)  # reset to default
    temp2 = np.nan_to_num(temp1, nan=0, posinf=0, neginf=0)
    best_thr_ori = ori_ROC[0, np.argmax(temp2)]
    best_TPR_ori = ori_ROC[1, np.argmax(temp2)]
    best_FPR_ori = ori_ROC[2, np.argmax(temp2)]
    # Calculate Precision, Recall and F-Score with this threshold
    precision_ori, recall_ori, fScore_ori = precision_recall_fScore_byThreshold(best_thr_ori, ori_raw_prediction, ground_truth, negative, positive)
    
    # Calculate the best threshold value for given image ID - augmented PLS prediction
    old_settings = np.seterr(divide='ignore', invalid='ignore')
    temp1 = np.divide(aug_ROC[1].astype('float64'), 0.3+aug_ROC[2].astype('float64'))
    np.seterr(**old_settings)  # reset to default
    temp2 = np.nan_to_num(temp1, nan=0, posinf=0, neginf=0)
    best_thr_aug = aug_ROC[0, np.argmax(temp2)]
    best_TPR_aug = aug_ROC[1, np.argmax(temp2)]
    best_FPR_aug = aug_ROC[2, np.argmax(temp2)]
    # Calculate Precision, Recall and F-Score with this threshold
    precision_aug, recall_aug, fScore_aug = precision_recall_fScore_byThreshold(best_thr_aug, aug_raw_prediction, ground_truth, negative, positive)
    
    # Plot the ROC and Save
    Path(savePath).mkdir(parents=True, exist_ok=True)
    sorted_ori_ROC = ori_ROC[:, ori_ROC[1].argsort()]
    sorted_aug_ROC = aug_ROC[:, aug_ROC[1].argsort()]
    plt.rc('font', size=19)         # controls default text sizes
    plt.rc('axes', titlesize=21)    # fontsize of the axes title
    plt.rc('axes', labelsize=21)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)   # fontsize of the tick labels
    plt.rc('legend', fontsize=16)   # legend fontsize
    plt.rc('figure', titlesize=19)  # fontsize of the figure title
    plt.figure(figsize=(10,6))
    plt.plot(sorted_ori_ROC[2,:], sorted_ori_ROC[1,:], 'c', label='without augmentation; AUC='+str(round(compute_auc(sorted_ori_ROC[2,:], sorted_ori_ROC[1,:]),3)))
    plt.plot(sorted_aug_ROC[2,:], sorted_aug_ROC[1,:], 'm', label='after augmentation; AUC='+str(round(compute_auc(sorted_aug_ROC[2,:], sorted_aug_ROC[1,:]),3)))
    plt.scatter([best_FPR_ori], [best_TPR_ori], c='b')
    plt.scatter([best_FPR_aug], [best_TPR_aug], c='r')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.title('Without Augmentation: P = '+str(round(precision_ori,3))+'; R = '+str(round(recall_ori,3))+'; F = '+str(round(fScore_ori,3))+'\nAfter Augmentation: P = '+str(round(precision_aug,3))+'; R = '+str(round(recall_aug,3))+'; F = '+str(round(fScore_aug,3)))
    plt.savefig(savePath+str(imgID)+'_roc.png', bbox_inches = 'tight', pad_inches = 0.05)
    plt.savefig(savePath+str(imgID)+'_roc.pdf', bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()
    
    print('Done with ROC and analysis for Image #'+str(imgID))
    
    return [imgID, precision_ori, recall_ori, fScore_ori, precision_aug, recall_aug, fScore_aug]

if __name__ == '__main__':
    r_b_channel_path = './data/R-B_Original/'
    ori_GT_binMap_path = './data/GTmaps/'
    
    final_gan_gen_images = './data/final_generated_images/'
    binMaps_final_gen_images = './data/genBINmaps_smoothened_images/'#'./data/BINmaps_generated_images/'
    
    nComp_vs_score_fig = './optimizingPlotPLS.png'
    nComp_vs_score_fig_aug = './optimizingPlotPLS_ganAugmented.png'
    
    nComp_vs_score_fig_pdf = './optimizingPlotPLS.pdf'
    nComp_vs_score_fig_aug_pdf = './optimizingPlotPLS_ganAugmented.pdf'
    
    favourable_GAN_Images = "./data/favourable_GAN_Images/"
    favourable_GAN_BMaps = "./data/favourable_GAN_BMaps/"
    
    train_set_images = "./data/testing/train_set_images/"
    train_set_BMaps = "./data/testing/train_set_BMaps/"
    val_set_images = "./data/testing/val_set_images/"
    val_set_BMaps = "./data/testing/val_set_BMaps/"
    test_set_images = "./data/testing/test_set_images/"
    test_set_BMaps = "./data/testing/test_set_BMaps/"
    predicted_BMaps_oriPLS = "./data/testing/predicted_BMaps_oriPLS/"
    predicted_BMaps_augPLS = "./data/testing/predicted_BMaps_augPLS/"
    
    kFold_train_set_images = "./data/kFold/train_set_images"
    kFold_train_set_BMaps = "./data/kFold/train_set_BMaps"
    kFold_test_set_images = "./data/kFold/test_set_images"
    kFold_test_set_BMaps = "./data/kFold/test_set_BMaps"
    
    roc_images = "./data/testing/roc/"
    
    X_ori_all, Y_ori_all, ori_ids = read_data(r_b_channel_path, ori_GT_binMap_path)
    
    X_train, X_trainC, Y_train, Y_trainC, ID_train, ID_trainC = train_test_split(X_ori_all, Y_ori_all, ori_ids, test_size=0.4, random_state=42)
    X_val, X_test, Y_val, Y_test, ID_val, ID_test = train_test_split(X_trainC, Y_trainC, ID_trainC, test_size=0.6, random_state=42)
    print("Number of images in train set: ", len(X_train))
    print("Number of images in validation set: ", len(X_val))
    print("Number of images in test set: ", len(X_test))
    
    # Save splitted image data
    print("Saving Images...")
    save_images(X_train, ID_train, train_set_images)
    save_images(Y_train, ID_train, train_set_BMaps)
    save_images(X_val, ID_val, val_set_images)
    save_images(Y_val, ID_val, val_set_BMaps)
    save_images(X_test, ID_test, test_set_images)
    save_images(Y_test, ID_test, test_set_BMaps)
    print("Saving Complete!")
    
    ##X_train, X_test, Y_train, Y_test, ID_train, ID_test = train_test_split(X_ori_all, Y_ori_all, ori_ids, test_size=0.3, random_state=42)
    
    # Save splitted image data
    print("Saving Images...")
    save_images(X_train, ID_train, kFold_train_set_images)
    save_images(Y_train, ID_train, kFold_train_set_images)
    save_images(X_test, ID_test, kFold_test_set_images)
    save_images(Y_test, ID_test, kFold_test_set_BMaps)
    print("Saving Complete!")
    
    scoreT, scoreV = find_optimal_nComp(X_train, Y_train, nComp_min = 1, nComp_max = 40, n_splits=8, plot_save_dir='./optimizationPlotsPLS/')
    print(scoreT)
    print(scoreV)
    
    '''
    # Score arrays as obtained for different nComp values in range [1, 40] - non-augmented train set training
    scoreT = [0.1837379899237719, 0.3129873840229538, 0.40796601464286714, 0.4543889937249687, 0.49228503838755894, 0.5289692150155484, 0.5565490157102505, 0.5828619203402035, 0.602552204981483, 0.6240906632884425, 0.6418339300995215, 0.6597181761569783, 0.680625505715245, 0.6938725013141746, 0.7082037102672241, 0.725756884193244, 0.7366540080511073, 0.7456396716061361, 0.7577446331811476, 0.7690278633188565, 0.7785133597415976, 0.7878597402546276, 0.7945901973176037, 0.8025534681150731, 0.8103404660303161, 0.8184620471400692, 0.826529806927161, 0.8327924680387471, 0.8386022690480289, 0.8459131289053781, 0.8518051596795613, 0.8591393588239825, 0.8650650690585384, 0.8715726110812154, 0.877696847514776, 0.8844036827789757, 0.8899811438405921, 0.8954348844740578, 0.9005369039118787, 0.9053274933941166]
    scoreV = [0.07620116772706641, 0.12175835608735543, 0.15213309100993483, 0.16982514918528002, 0.20147110733790743, 0.22681250220152147, 0.23785438303046422, 0.24810500815890288, 0.24533818432941087, 0.2378152898454061, 0.21712114495718524, 0.2006054054371563, 0.22850295392457706, 0.23001648197432228, 0.22919967824634904, 0.2199990267007122, 0.20580201769297368, 0.20413923444013438, 0.20202837897683285, 0.18404514723428655, 0.17680704446995704, 0.16525303815298711, 0.1453472174274316, 0.13634547799571423, 0.1294085039673575, 0.12711796126980185, 0.10599171732133018, 0.10595535081667831, 0.09578255084273955, 0.08781875627562791, 0.09130985670360552, 0.08423081986067528, 0.08142302745764787, 0.07354474997551261, 0.05896084339037937, 0.057062836781970556, 0.02618312601252592, -0.0031763034985333887, -0.008759147179501815, -0.0047580397108942865]
    
    plt.rc('font', size=19)         # controls default text sizes
    plt.rc('axes', titlesize=22)    # fontsize of the axes title
    plt.rc('axes', labelsize=22)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=19)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=19)   # fontsize of the tick labels
    plt.rc('legend', fontsize=20)   # legend fontsize
    plt.rc('figure', titlesize=22)  # fontsize of the figure title
    plt.figure(figsize=(10,6))
    plt.plot(np.arange(1, 41).tolist(), scoreT, label='Training Set')
    plt.plot(np.arange(1, 41).tolist(), scoreV, label='Validation Set')
    markX = 8
    markY = 0.24810500815890288
    annotX = 23
    annotY = 0.4
    if markX is not None and markY is not None:
        plt.hlines(markY, 0, markX, linestyles='dashed')
        plt.vlines(markX, 0, markY, linestyles='dashed')
        plt.scatter([markX],[markY], c='r')
        if annotX is not None and annotY is not None:
            if annotY>markY:
                plt.annotate('Max Score on\nValidation Set\n'+str((markX, np.around(markY,3))),
                             xy=(markX, markY+0.002), xytext=(annotX, annotY),
                             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle3,angleA=0,angleB=-90"));
            else:
                plt.annotate('Max Score on\nValidation Set\n'+str((markX, np.around(markY,3))),
                             xy=(markX, markY-0.002), xytext=(annotX, annotY),
                             arrowprops=dict(arrowstyle="->",
                             connectionstyle="angle3,angleA=0,angleB=-90"));
    plt.xlabel("Number of PLS Components")
    plt.ylabel("Coeff. of Determination (R^2)")
    plt.legend()
    low = min(scoreT + scoreV)
    high = max(scoreT + scoreV)
    plt.xlim([0, 41])
    plt.ylim([max(0,low-0.8*(high-low)), min(high+0.4*high, high+0.4*(high-low))])
    plt.savefig(nComp_vs_score_fig, bbox_inches = 'tight', pad_inches = 0.05)
    plt.savefig(nComp_vs_score_fig_pdf, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()
    '''
    
    '''
    ori_pls, _ = train_pls(X_train, Y_train, n_comp = 8)
    oriTrainError = ori_pls.score(X_train, Y_train)
    oriValError = ori_pls.score(X_val, Y_val)
    oriTestError = ori_pls.score(X_test, Y_test)
    print("Train Score = ", oriTrainError)
    print("Test Score = ", oriTestError)
    
    X_gan, Y_gan, ganImID = read_data(final_gan_gen_images, binMaps_final_gen_images)
    trainError = []
    valError = []
    ganIDsError = []
    results = Parallel(n_jobs=5)(delayed(genGANaugmentedErrors)(X_train, Y_train, X_gan[i], Y_gan[i], ganImID[i], X_val, Y_val) for i in range(len(X_gan)))
    #print(len(results))
    #print(results)
    new_X_gan = []
    new_Y_gan = []
    new_ganImID = []
    Path(favourable_GAN_Images).mkdir(parents=True, exist_ok=True)
    Path(favourable_GAN_BMaps).mkdir(parents=True, exist_ok=True)
    for i in range(len(results)):
        trainError.append(results[i][0])
        valError.append(results[i][1])
        ganIDsError.append(results[i][2])
        if results[i][1] >= oriValError:
            new_X_gan.append(X_gan[ganImID.index(results[i][2])])
            new_Y_gan.append(Y_gan[ganImID.index(results[i][2])])
            new_ganImID.append(results[i][2])
            #shutil.copy(final_gan_gen_images+str(results[i][2])+'.png', favourable_GAN_Images)
            #shutil.copy(binMaps_final_gen_images+str(results[i][2])+'.png', favourable_GAN_BMaps)
    #print(trainError)
    #print(valError)
    #print(ganIDsError)
    
    
    plt.rc('font', size=19)         # controls default text sizes
    plt.rc('axes', titlesize=21)    # fontsize of the axes title
    plt.rc('axes', labelsize=21)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)   # fontsize of the tick labels
    plt.rc('legend', fontsize=16)   # legend fontsize
    plt.rc('figure', titlesize=19)  # fontsize of the figure title
    plt.figure(figsize=(10,6))
    plt.scatter(ganIDsError, trainError, label='Train Error')
    plt.scatter(ganIDsError, valError, label='Validation Error')
    plt.hlines(oriTrainError, 0, max(ganIDsError), linestyles='dashed', label='Original Train Error')
    plt.hlines(oriValError, 0, max(ganIDsError), linestyles='dashed', label='Original Validation Error')
    plt.xlabel("Augmented GAN Image ID")
    plt.ylabel("Coeff. of Determination (R^2)")
    plt.legend()
    low = min(trainError + valError)
    high = max(trainError + valError)
    plt.xlim([0, max(ganIDsError)+1])
    plt.ylim([max(0,low-0.2*(high-low)), min(high+0.2*high, high+0.2*(high-low))])
    plt.savefig(nComp_vs_score_fig_aug, bbox_inches = 'tight', pad_inches = 0.05)
    plt.savefig(nComp_vs_score_fig_aug_pdf, bbox_inches = 'tight', pad_inches = 0.05)
    plt.close()
    '''
    '''
    # Train again on original data and print results
    ori_pls, _ = train_pls(X_train, Y_train, n_comp = 8)
    oriTrainError = ori_pls.score(X_train, Y_train)
    oriValError = ori_pls.score(X_val, Y_val)
    oriTestError = ori_pls.score(X_test, Y_test)
    print("Train Score = ", oriTrainError)
    print("Test Score = ", oriTestError)
    
    # Read previously stored favourable GAN images
    new_X_gan, new_Y_gan, new_ganImID = read_data(favourable_GAN_Images, favourable_GAN_BMaps)
    print("Number of favourable GAN images = ", len(new_ganImID))
    aug_pls, _ = train_pls(X_train+new_X_gan, Y_train+new_Y_gan, n_comp = 8)
    print("New Train Score = ", aug_pls.score(X_train+new_X_gan, Y_train+new_Y_gan))
    print("New Test Score = ", aug_pls.score(X_test, Y_test))
    
    ori_raw_prediction = ori_pls.predict(X_test)
    aug_raw_prediction = aug_pls.predict(X_test)
    
    #for i in range(len(ID_test)):
    #    create_roc_byIndex(ID_test[i], roc_images, ori_raw_prediction[i], aug_raw_prediction[i], Y_test[i], negative=0, positive=255)
    #    print('Done with Image #'+str(i))
    resultsROC = Parallel(n_jobs=5)(delayed(create_roc_byIndex)(ID_test[i], roc_images, ori_raw_prediction[i], aug_raw_prediction[i], Y_test[i], negative=0, positive=255) for i in range(len(ID_test)))
    
    avg_ori_precision = 0
    avg_ori_recall = 0
    avg_ori_fScore = 0
    avg_aug_precision = 0
    avg_aug_recall = 0
    avg_aug_fScore = 0
    for i in range(len(resultsROC)):
        avg_ori_precision   += resultsROC[i][1]
        avg_ori_recall      += resultsROC[i][2]
        avg_ori_fScore      += resultsROC[i][3]
        avg_aug_precision   += resultsROC[i][4]
        avg_aug_recall      += resultsROC[i][5]
        avg_aug_fScore      += resultsROC[i][6]
    print(i)
    avg_ori_precision   /= len(resultsROC)
    avg_ori_recall      /= len(resultsROC)
    avg_ori_fScore      /= len(resultsROC)
    avg_aug_precision   /= len(resultsROC)
    avg_aug_recall      /= len(resultsROC)
    avg_aug_fScore      /= len(resultsROC)
    
    print("\n")
    print("Results before augmentation:\n")
    print("Precision = ", avg_ori_precision)
    print("Recall = ", avg_ori_recall)
    print("fScore = ", avg_ori_fScore)
    print("\n")
    print("Results after augmentation:\n")
    print("Precision = ", avg_aug_precision)
    print("Recall = ", avg_aug_recall)
    print("fScore = ", avg_aug_fScore)
    '''
    '''
    # Generate binary maps using trained PLS models
    ori_predicted = plsPredictions_to_images(ori_pls.predict(X_test))
    aug_predicted = plsPredictions_to_images(aug_pls.predict(X_test))
    for i in range(len(X_test)):
        print('For imageID = ', ID_test[i])
        print('Precision = ', compute_precision(Y_test[i], ori_predicted[i]), '; Recall = ', compute_recall(Y_test[i], ori_predicted[i]), '; F-Score = ', compute_fScore(Y_test[i], ori_predicted[i]))
        print('Precision = ', compute_precision(Y_test[i], aug_predicted[i]), '; Recall = ', compute_recall(Y_test[i], aug_predicted[i]), '; F-Score = ', compute_fScore(Y_test[i], aug_predicted[i]))
        print('')
    '''