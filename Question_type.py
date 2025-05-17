"""
Utility module for handling question categorization and evaluating metrics for
Continual Visual Question Answering (CVQA) tasks.
"""

import random
import json
import matplotlib.pyplot as plt
import numpy

# Set random seed for reproducibility
seed = 66666
random.seed(seed)
print('random seed', seed)

def random_dic(dicts):
    """
    Randomly shuffle dictionary keys and return a new dictionary.
    
    Args:
        dicts: Input dictionary
        
    Returns:
        New dictionary with randomly shuffled keys
    """
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dict = {}
    for key in dict_key_ls:
        new_dict[key] = dicts.get(key)
    return new_dict

# Define all available tasks
All_task = [1,2,3,4,5,6,7,8]

# Define category splits for each task
Category_splits = {}
for i in range(1, 9):
    Category_splits[str(i)] = { 'G0' : [0,1],\
                           'G1' : [2,3],\
                           'G2' : [4,5],\
                           'G3' : [6,7],\
                            'G4': [8,9]}

# Load task mapping data
with open('/root/autodl-tmp/datasets/QuesId_task_map.json') as fp:
    QuesId_task_map = json.load(fp)

with open('/root/autodl-tmp/datasets/ImgId_cate_map.json') as fp:
    ImgId_cate_map = json.load(fp)

print("Success to load the QuesId_task_map and ImgId_cate_map")

# Question indices for special evaluation
_6Q_idx = []

# Create key lists for tracking
_key_list = []
for key in All_task:
    _key_list.append(key)

All_task_list = []
for key in All_task:
    All_task_list.append(key)

def show_results_matrix(results, start=0):
    """
    Display the evaluation results in a matrix format.
    
    Args:
        results: Dictionary containing evaluation results
        start: Index to start displaying results from
    """
    matrix = numpy.zeros([len(results), len(results)], dtype=float)
    key_list = []
    for key in results:
        print(key, end='\t')
        key_list.append(key)
    print('\n')
    for i in range(start,len(results)):
        avg = 0
        for j in range(start,len(results)):
            if j < i+1:
                matrix[i][j] = results[key_list[i]][key_list[j]]
                avg += matrix[i][j]
            print(round(matrix[i][j], 2), end='\t')

        print("Avg:", round(avg / (len(results)-start),2))


def Update_memory(M, task_idx, task):
    """
    Update memory buffer with examples from the current task.
    
    Args:
        M: Memory buffer size
        task_idx: Current task index
        task: Current task
        
    Returns:
        Dictionary of exemplar sets organized by category
    """
    Examplar_set = {'G0': [], 'G1': [], 'G2': [], 'G3': [], 'G4': []}
    return Examplar_set

def evaluate_metric(results, start=0):
    """
    Evaluate continual learning metrics including incremental accuracy and forgetting.
    
    Args:
        results: Dictionary containing evaluation results
        start: Index to start evaluation from
        
    Returns:
        Dictionary with various metrics:
        - Incremental average accuracy
        - Average accuracy
        - Incremental average forgetting
        - Average forgetting
        - Specialized metrics for specific question types
    """
    matrix = numpy.zeros([len(results), len(results)], dtype=float)-1
    key_list = []
    for key in results:
        key_list.append(key)
    for i in range(start, len(results)):
        avg = 0
        for j in range(start, len(results)):
            if j < i + 1:
                matrix[i][j] = results[key_list[i]][key_list[j]]

    # Calculate incremental accuracy performance
    Incre_avg_accuracy = []
    Incre_avg_accuracy_6Q = []
    for t in range(start, len(results)):
        now_acc = matrix[t]
        all_acc = 0
        num = 0
        for acc in now_acc:
            if acc != -1:
                all_acc += acc
                num += 1
        avg_acc = all_acc / num
        Incre_avg_accuracy.append(avg_acc)

        all_acc = 0
        num = 0
        for i in range(len(now_acc)):
            if i in _6Q_idx:
                if now_acc[i] != -1:
                    all_acc += now_acc[i]
                    num += 1
        if num!=0:
            avg_acc = all_acc / num
        else:
            avg_acc = -1
        Incre_avg_accuracy_6Q.append(avg_acc)

    # Calculate final average accuracy
    Avg_accuracy = Incre_avg_accuracy[-1]
    Avg_accuracy_6Q = Incre_avg_accuracy_6Q[-1]

    # Calculate forgetting metrics
    Incre_avg_forget = [0]
    Incre_avg_forget_6Q = [0]
    for t in range(1+start,len(results)):
        results_now = matrix[:t+1, :t+1]
        t_forget = []
        for idx in range(start, len(results_now)-1):
            task_list = results_now[:-1,idx]
            final = results_now[-1,idx]
            pre_max = max(task_list)
            if pre_max == -1:
                t_forget.append(0)
            else:
                t_forget.append(pre_max - final)
        Avg_forget = sum(t_forget)/len(t_forget)
        Incre_avg_forget.append(Avg_forget)

        t_forget_6Q = []
        for i_ in range(len(t_forget)):
            if i_+1 in _6Q_idx:
                t_forget_6Q.append(t_forget[i_])
        if len(t_forget_6Q) > 0:
            Avg_forget = sum(t_forget_6Q) / len(t_forget_6Q)
        else:
            Avg_forget = -1
        Incre_avg_forget_6Q.append(Avg_forget)

    Avg_forget = Incre_avg_forget[-1]
    Avg_forget_6Q = Incre_avg_forget_6Q[-1]

    # Compile all metrics into a dictionary
    output_dict = {'Incre_avg_acc': Incre_avg_accuracy,
                   'Avg_acc': Avg_accuracy,
                   'Incre_avg_forget': Incre_avg_forget,
                   'Avg_forget':Avg_forget,
                   'Incre_avg_acc_6Q': Incre_avg_accuracy_6Q,
                   'Avg_acc_6Q': Avg_accuracy_6Q,
                   'Incre_avg_forget_6Q': Incre_avg_forget_6Q,
                   'Avg_forget_6Q': Avg_forget_6Q,
                   }
    return output_dict


if __name__ == "__main__":
    result_matrix = {}
