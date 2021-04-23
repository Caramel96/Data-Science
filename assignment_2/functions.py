def drop_columns(data):
  data_drop = data.copy()
  cols_to_drop = ['Date', 'Part of a policing operation', 'Policing operation', 'Latitude', 'Longitude', 'Self-defined ethnicity', 'Removal of more than just outer clothing', 'Outcome linked to object of search']
 
  data_drop.drop(columns=cols_to_drop, inplace=True)

  data_drop.dropna(inplace=True)

  data_drop = data_drop.replace({'Outcome' :  {'Arrest': 'Action Taken', 'Community resolution': 'Action Taken', 'Nothing found - no further action': 'A no further action disposal', 'Khat or Cannabis warning': 'Action Taken', 
           'Suspect arrested': 'Action Taken', 'Local resolution': 'Action Taken', 'Summons / charged by post': 'Action Taken', 
           'Caution (simple or conditional)': 'Action Taken', 'Offender given drugs possession warning': 'Action Taken', 'Suspect summonsed to court': 'Action Taken', 'Penalty Notice for Disorder': 'Action Taken', 'Offender cautioned':'Action Taken', 'Offender given penalty notice': 'Action Taken'}})

  return data_drop

def drop_met(data):
 data_drop_met = data.copy()
 cols_to_drop_met = ['Date', 'Policing operation',  'Latitude', 'Longitude', 'Self-defined ethnicity', 'Removal of more than just outer clothing', 'Outcome linked to object of search', 'Force']
 data_drop_met.drop(columns=cols_to_drop_met, inplace=True)
 data_drop_met.dropna(inplace=True)
 data_drop_met = data_drop_met.replace({'Outcome' :  {'Arrest': 'Action Taken', 'Community resolution': 'Action Taken', 'Nothing found - no further action': 'A no further action disposal', 'Khat or Cannabis warning': 'Action Taken', 
           'Suspect arrested': 'Action Taken', 'Local resolution': 'Action Taken', 'Summons / charged by post': 'Action Taken', 
           'Caution (simple or conditional)': 'Action Taken', 'Offender given drugs possession warning': 'Action Taken', 'Suspect summonsed to court': 'Action Taken', 'Penalty Notice for Disorder': 'Action Taken', 'Offender cautioned':'Action Taken', 'Offender given penalty notice': 'Action Taken'}})
 return data_drop_met

def drop_man(data):
 data_drop_man = data.copy()
 cols_to_drop_man = ['Date', 'Policing operation',  'Latitude', 'Longitude', 'Self-defined ethnicity', 'Outcome linked to object of search', 'Force']
 data_drop_man.drop(columns=cols_to_drop_man, inplace=True)
 data_drop_man.dropna(inplace=True)
 data_drop_man = data_drop_man.replace({'Outcome' :  {'Arrest': 'Action Taken', 'Community resolution': 'Action Taken', 'Nothing found - no further action': 'A no further action disposal', 'Khat or Cannabis warning': 'Action Taken', 
           'Suspect arrested': 'Action Taken', 'Local resolution': 'Action Taken', 'Summons / charged by post': 'Action Taken', 
           'Caution (simple or conditional)': 'Action Taken', 'Offender given drugs possession warning': 'Action Taken', 'Suspect summonsed to court': 'Action Taken', 'Penalty Notice for Disorder': 'Action Taken', 'Offender cautioned':'Action Taken', 'Offender given penalty notice': 'Action Taken'}})
 return data_drop_man

def drop_leic(data):
 data_drop_lc = data.copy()
 cols_to_drop_lc = ['Date', 'Part of a policing operation', 'Policing operation', 'Latitude', 'Longitude', 'Self-defined ethnicity', 'Removal of more than just outer clothing', 'Outcome linked to object of search', 'Force']
 data_drop_lc.drop(columns=cols_to_drop_lc, inplace=True)
 data_drop_lc.dropna(inplace=True)
 data_drop_lc = data_drop_lc.replace({'Outcome' :  {'Arrest': 'Action Taken', 'Community resolution': 'Action Taken', 'Nothing found - no further action': 'A no further action disposal', 'Khat or Cannabis warning': 'Action Taken', 
           'Suspect arrested': 'Action Taken', 'Local resolution': 'Action Taken', 'Summons / charged by post': 'Action Taken', 
           'Caution (simple or conditional)': 'Action Taken', 'Offender given drugs possession warning': 'Action Taken', 'Suspect summonsed to court': 'Action Taken', 'Penalty Notice for Disorder': 'Action Taken'}})
 return data_drop_lc

def encode_data(data):
  categorical_features = ['Type', 'Gender', 'Age range', 'Officer-defined ethnicity', 'Legislation', 'Object of search', 'Outcome', 'Force']
  data_encoded = data.copy()

  categorical_names = {}
  encoders = {}


  for feature in categorical_features:
    le = LabelEncoder()
    le.fit(data_encoded[feature])
    
    data_encoded[feature] = le.transform(data_encoded[feature])
    
    categorical_names[feature] = le.classes_
    encoders[feature] = le
  return data_encoded

def decode_dataset(data, encoders, categorical_features):
  categorical_features = ['Type', 'Gender', 'Age range', 'Officer-defined ethnicity', 'Legislation', 'Object of search', 'Outcome', 'Force']
    df = data.copy()
    for feat in categorical_features:
        df[feat] = encoders[feat].inverse_transform(df[feat])
    return df
def StandardDataset_all(data):

  privileged_sex = np.where(categorical_names['Gender'] == 'Male')[0]
  privileged_race = np.where(categorical_names['Officer-defined ethnicity'] == 'White')[0]
  data_orig_outcomes_all = StandardDataset(data, 
                               label_name='Outcome', 
                               favorable_classes=[1], 
                               protected_attribute_names=['Gender','Officer-defined ethnicity'], 
                               privileged_classes=[privileged_sex, privileged_race])
  return data_orig_outcomes_all

def StandardDataset_man(data):

  privileged_sex_man = np.where(categorical_names_man['Gender'] == 'Male')[0]
  privileged_race_man = np.where(categorical_names_man['Officer-defined ethnicity'] == 'White')[0]
  data_orig_outcomes_man = StandardDataset(data, 
                               label_name='Outcome', 
                               favorable_classes=[1], 
                               protected_attribute_names=['Gender','Officer-defined ethnicity'], 
                               privileged_classes=[privileged_sex_man, privileged_race_man])
  return data_orig_outcomes_man

def StandardDataset_lc(data):

  privileged_sex = np.where(categorical_names_lc['Gender'] == 'Male')[0]
  privileged_race = np.where(categorical_names_lc['Officer-defined ethnicity'] == 'White')[0]
  data_orig_outcomes_lc = StandardDataset(data, 
                               label_name='Outcome', 
                               favorable_classes=[1], 
                               protected_attribute_names=['Gender','Officer-defined ethnicity'], 
                               privileged_classes=[privileged_sex, privileged_race])
  return data_orig_outcomes_lc

def StandardDataset_met(data):

  privileged_sex = np.where(categorical_names_met['Gender'] == 'Male')[0]
  privileged_race = np.where(categorical_names_met['Officer-defined ethnicity'] == 'White')[0]
  data_orig_outcomes_met = StandardDataset(data, 
                               label_name='Outcome', 
                               favorable_classes=[1], 
                               protected_attribute_names=['Gender','Officer-defined ethnicity'], 
                               privileged_classes=[privileged_sex, privileged_race])
  return data_orig_outcomes_met



def meta_data(dataset):
    # print out some labels, names, etc.
    display(Markdown("#### Dataset shape"))
    print(dataset.features.shape)
    display(Markdown("#### Favorable and unfavorable labels"))
    print(dataset.favorable_label, dataset.unfavorable_label)
    display(Markdown("#### Protected attribute names"))
    print(dataset.protected_attribute_names)
    display(Markdown("#### Privileged and unprivileged protected attribute values"))
    print(dataset.privileged_protected_attributes, dataset.unprivileged_protected_attributes)
    display(Markdown("#### Dataset feature names"))
    print(dataset.feature_names)

def get_attributes(data, selected_attr=None):
    unprivileged_groups = []
    privileged_groups = []
    if selected_attr == None:
        selected_attr = data.protected_attribute_names
    
    for attr in selected_attr:
            idx = data.protected_attribute_names.index(attr)
            privileged_groups.append({attr:data.privileged_protected_attributes[idx]}) 
            unprivileged_groups.append({attr:data.unprivileged_protected_attributes[idx]}) 

    return privileged_groups, unprivileged_groups

def get_model_performance(X_test, y_true, y_pred, probs):
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='micro')
    preds = probs[:, 1]
    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)

    return accuracy, matrix, f1, fpr, tpr, roc_auc

def plot_model_performance(model, X_test, y_true):
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    accuracy, matrix, f1, fpr, tpr, roc_auc = get_model_performance(X_test, y_true, y_pred, probs)

    display(Markdown('#### Accuracy of the model :'))
    print(accuracy)
    display(Markdown('#### F1 score of the model :'))
    print(f1)

    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(1, 2, 1)
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt='g')
    plt.title('Confusion Matrix')

    ax = fig.add_subplot(1, 2, 2)
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="lower right")

def test(dataset, model, thresh_arr):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = np.where(model.classes_ == dataset.favorable_label)[0][0]
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0
    
    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
                dataset, dataset_pred,
                unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)

        metric_arrs['bal_acc'].append((metric.true_positive_rate()
                                     + metric.true_negative_rate()) / 2)
        metric_arrs['avg_odds_diff'].append(metric.average_odds_difference())
        metric_arrs['disp_imp'].append(metric.disparate_impact())
        metric_arrs['stat_par_diff'].append(metric.statistical_parity_difference())
        metric_arrs['eq_opp_diff'].append(metric.equal_opportunity_difference())
        metric_arrs['theil_ind'].append(metric.theil_index())
    
    return metric_arrs
  
def plot(x, x_name, y_left, y_left_name, y_right, y_right_name):
    fig, ax1 = plt.subplots(figsize=(10,7))
    ax1.plot(x, y_left)
    ax1.set_xlabel(x_name, fontsize=16, fontweight='bold')
    ax1.set_ylabel(y_left_name, color='b', fontsize=16, fontweight='bold')
    ax1.xaxis.set_tick_params(labelsize=14)
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_ylim(0.5, 0.8)

    ax2 = ax1.twinx()
    ax2.plot(x, y_right, color='r')
    ax2.set_ylabel(y_right_name, color='r', fontsize=16, fontweight='bold')
    if 'DI' in y_right_name:
        ax2.set_ylim(0., 0.7)
    else:
        ax2.set_ylim(-0.25, 0.1)

    best_ind = np.argmax(y_left)
    ax2.axvline(np.array(x)[best_ind], color='k', linestyle=':')
    ax2.yaxis.set_tick_params(labelsize=14)
    ax2.grid(True)

def add_to_df_algo_metrics(algo_metrics, model, fair_metrics, preds, probs, name):
    return algo_metrics.append(pd.DataFrame(data=[[model, fair_metrics, preds, probs]], columns=['model', 'fair_metrics', 'prediction', 'probs'], index=[name]))

def get_fair_metrics_and_plot(data, model, plot=True, model_aif=False):
    pred = model.predict(data).labels if model_aif else model.predict(data.features)
    # fair_metrics function available in the metrics.py file
    fair = fair_metrics(data, pred)

    if plot:
        # plot_fair_metrics function available in the visualisations.py file
        # The visualisation of this function is inspired by the dashboard on the demo of IBM aif360 
        plot_fair_metrics(fair)
        display(fair)
    
    return fair

def fair_metrics(dataset, pred, pred_is_dataset=False):
    if pred_is_dataset:
        dataset_pred = pred
    else:
        dataset_pred = dataset.copy()
        dataset_pred.labels = pred
    
    cols = ['statistical_parity_difference', 'equal_opportunity_difference', 'average_abs_odds_difference',  'disparate_impact', 'theil_index']
    obj_fairness = [[0,0,0,1,0]]
    
    fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)
    
    for attr in dataset_pred.protected_attribute_names:
        idx = dataset_pred.protected_attribute_names.index(attr)
        privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}] 
        unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}] 
        
        classified_metric = ClassificationMetric(dataset, 
                                                     dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        acc = classified_metric.accuracy()

        row = pd.DataFrame([[metric_pred.mean_difference(),
                                classified_metric.equal_opportunity_difference(),
                                classified_metric.average_abs_odds_difference(),
                                metric_pred.disparate_impact(),
                                classified_metric.theil_index()]],
                           columns  = cols,
                           index = [attr]
                          )
        fair_metrics = fair_metrics.append(row)    
    
    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)
        
    return fair_metrics

def plot_fair_metrics(fair_metrics):
    fig, ax = plt.subplots(figsize=(20,4), ncols=5, nrows=1)

    plt.subplots_adjust(
        left    =  0.125, 
        bottom  =  0.1, 
        right   =  0.9, 
        top     =  0.9, 
        wspace  =  .5, 
        hspace  =  1.1
    )

    y_title_margin = 1.2

    plt.suptitle("Fairness metrics", y = 1.09, fontsize=20)
    sns.set(style="dark")

    cols = fair_metrics.columns.values
    obj = fair_metrics.loc['objective']
    size_rect = [0.2,0.2,0.2,0.4,0.25]
    rect = [-0.1,-0.1,-0.1,0.8,0]
    bottom = [-1,-1,-1,0,0]
    top = [1,1,1,2,1]
    bound = [[-0.1,0.1],[-0.1,0.1],[-0.1,0.1],[0.8,1.2],[0,0.25]]

    display(Markdown("### Check bias metrics :"))
    display(Markdown("A model can be considered bias if just one of these five metrics show that this model is biased."))
    for attr in fair_metrics.index[1:len(fair_metrics)].values:
        display(Markdown("#### For the %s attribute :"%attr))
        check = [bound[i][0] < fair_metrics.loc[attr][i] < bound[i][1] for i in range(0,5)]
        display(Markdown("With default thresholds, bias against unprivileged group detected in **%d** out of 5 metrics"%(5 - sum(check))))

    for i in range(0,5):
        plt.subplot(1, 5, i+1)
        ax = sns.barplot(x=fair_metrics.index[1:len(fair_metrics)], y=fair_metrics.iloc[1:len(fair_metrics)][cols[i]])
        
        for j in range(0,len(fair_metrics)-1):
            a, val = ax.patches[j], fair_metrics.iloc[j+1][cols[i]]
            marg = -0.2 if val < 0 else 0.1
            ax.text(a.get_x()+a.get_width()/5, a.get_y()+a.get_height()+marg, round(val, 3), fontsize=15,color='black')

        plt.ylim(bottom[i], top[i])
        plt.setp(ax.patches, linewidth=0)
        ax.add_patch(patches.Rectangle((-5,rect[i]), 10, size_rect[i], alpha=0.3, facecolor="green", linewidth=1, linestyle='solid'))
        plt.axhline(obj[i], color='black', alpha=0.3)
        plt.xticks(rotation = 45)
        plt.title(cols[i])
        ax.set_ylabel('')    
        ax.set_xlabel('')
        
