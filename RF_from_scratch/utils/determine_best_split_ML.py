def determine_best_split(X, y, potential_splits,typ="regression",k=0):
    
    'selects which split lowered gini/mse the most'
    first_iteration = True
    # n_final=len(y)
    overall_impurity = calculate_impurity(y,typ=typ,k=k)  # the function will loop over and replace this with lower impurity values
    overall_impurity_for_gain = overall_impurity.copy()
    min_impurity=[]
    best_split_values={}
    
    n_final = []
    for column_index in potential_splits:
        i = 0
        num_pot_splits = len(potential_splits[column_index])
        current_overall_impurity = np.zeros(num_pot_splits)
        for value in potential_splits[column_index]:
            _, _, y_below, y_above = split_data(X, y, split_column=column_index, split_value=value)
            # BTW, I think the returned n is always the same, no ?? 
            current_overall_impurity[i], n = calculate_overall_impurity(y_below, y_above,k=k, typ=typ)
            i+=1
            
        min_imp = np.min(current_overall_impurity)
        min_indices = np.where(current_overall_impurity == min_imp)
        min_impurity.append(min_imp)
        best_split_values[column_index] = potential_splits[column_index][min_indices]
        n_final.append(n)

    #are features ties in their impurity ?
    min_imp = np.min(min_impurity)
    min_indices = np.where(min_impurity == min_imp)

    # randomly select multiple potential splits
    ftrIndex = np.random.choice(min_indices,1)
    best_split_column = ftrIndex
    #if (len(best_split_values[ftrIndex]) > 1)
    best_split_value = np.random.choice(best_split_values[ftrIndex],1)
    n_final = n[ftrIndex]
    record = pd.DataFrame([best_split_column,best_split_value,n_final]).transpose()
    record.columns = ['best_split_column','best_split_value','n_final']
    # breakpoint()
    #result = record.sample()
    # try:
    #     result = record.sample()
    # except:
    #     # print(len(y_below), len(y))
    #     print(potential_splits)
    #     print(record)
    #     raise

    gain = overall_impurity_for_gain - min_imp
    rescale_gain = gain*result.n_final/len(y) #might only use rescale_gain
    return int(result.best_split_column), float(result.best_split_value), rescale_gain

def get_potential_splits(X, y, random_subspace = None, random_state=None, k=0, min_samples_leaf=1):
        
    'first, takes every unique value of every feature in the feature space, then finds the midpoint between each value'
    'modified to add random_subspace for random forest'
    # Get valid random state
    # random_state = check_random_state(random_state)
    potential_splits = {}
    _, n_columns = X.shape  # No need for rows, we choose the column to split on
    # Only need second value of .shape which is columns
    
    column_indices = list(range(n_columns))
    if random_subspace and random_subspace <= len(column_indices):  # Randomly chosen features
        # column_indices = random.sample(population=column_indices, k=random_subspace)
        # random_instance = check_random_state(random_state)
        column_indices = np.array(column_indices)[sample_without_replacement(n_population=len(column_indices),\
         n_samples=random_subspace, random_state=random_state)]

    for column_index in column_indices:
        potential_splits[column_index] = [] 
        values = X[:, column_index] 
        unique_values = np.unique(values)  # Get all unique values in each column
        #vectorized midpoints
        n = len(unique_values)
        potential_split_candidates  = (unique_values[0:(n-1)] + unique_values[1:n])/2

        #no need to check the full array with split_data, it should be just the boundaries that are potential trouble:
        j_left=0
        _, _, y_below, y_above = split_data(X, y, split_column=column_index, split_value=potential_split_candidates[j_left])
        # Reject if min_samples_leaf is not guaranteed
        # check that both children have samples sizes at least k+1! 
        while (len(y_below) < (k+1)) or (len(y_above) < (k+1)) or (len(y_below) < min_samples_leaf) or (len(y_above) < min_samples_leaf): 
            j_left+=1
            _, _, y_below, y_above = split_data(X, y, split_column=column_index, split_value=potential_split_candidates[j_left])    
        j_right=n
        _, _, y_below, y_above = split_data(X, y, split_column=column_index, split_value=potential_split_candidates[j_right])
        # Reject if min_samples_leaf is not guaranteed
        # check that both children have samples sizes at least k+1! 
        while (len(y_below) < (k+1)) or (len(y_above) < (k+1)) or (len(y_below) < min_samples_leaf) or (len(y_above) < min_samples_leaf): 
            j_right-=1
            _, _, y_below, y_above = split_data(X, y, split_column=column_index, split_value=potential_split_candidates[j_right]) 

        potential_splits[column_index] = potential_split_candidates[j_left:j_right]

        #for index in range(len(unique_values)):  # All unique feature values
            #if index != 0:  # Skip first value, we need the difference between next values
                # Stop early if remaining features are constant
                # current_value = unique_values[index]
                # previous_value = unique_values[index - 1]  # Find a value and the next smallest value
                #potential_split = (unique_values[index] + unique_values[index - 1]) / 2  # Find difference between the two as a potential split
                # print(random_state)
                # potential_split = rand_uniform(previous_value,current_value,random_state)
                # if potential_split == current_value:
                #     potential_split = previous_value

                # try to split the data
                #_, _, y_below, y_above = split_data(X, y, split_column=column_index, split_value=potential_split)
                # Reject if min_samples_leaf is not guaranteed
                # check that both children have samples sizes at least k+1! 
                #if (len(y_below) >= (k+1)) and (len(y_above) >= (k+1)) and (len(y_below) >= min_samples_leaf) and (len(y_above) >= min_samples_leaf): 
                #    potential_splits[column_index].append(potential_split)
                # Reject if min_samples_leaf is not guaranteed

        if potential_splits[column_index] == []:
            potential_splits = {key:val for key, val in potential_splits.items() if key != column_index}
                    # potential_splits.remove(column_index)
    # print(column_indices)
    return potential_splits