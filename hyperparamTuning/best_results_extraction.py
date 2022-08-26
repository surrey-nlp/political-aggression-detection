import os
import json


obj = os.scandir()
parent_dir_name = os.getcwd()[os.getcwd().rfind('\\')+1:]

# print("Files and Directories in PWD")
for entry in obj:
    if entry.is_dir():
        # print(entry.name)
        inner_obj = os.scandir(entry.name)
        # res_ls = {}

        
        file = open(parent_dir_name+'_tuning_results.txt', 'a')
        file.write(entry.name+':\n')


        for inner_entry in inner_obj:
            if inner_entry.is_dir():
                # print(inner_entry.name)
                inner_folder_path = entry.name + "/" + inner_entry.name
                file_to_chk_path = inner_folder_path + "/result.json"
                # print(file_to_chk_path)
                #Getting best f1 from each trial
                res_json = [json.loads(line) for line in open(file_to_chk_path,'r')]
                best_score = 0.0
                best_conf = {}
                for dt in res_json:
                    if dt['eval_f1'] > best_score:
                        best_score = dt['eval_f1']
                        best_conf = dt
                    # print(dt['eval_f1'])
                
                # res_ls[inner_entry.name] = best_conf
                
                # Writing results to a file
                file.write(inner_entry.name + ' -> ' + str(best_conf) + '\n')
        file.write('----------------------------------------------------\n')  
file.close()
