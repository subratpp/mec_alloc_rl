#!/usr/bin/env python
# coding: utf-8

# # RL Agent for Training Num of Users for a Server

import os
import time
import sys
import csv
import re


# =================================================================== Utility Functions
#writes the location of image to the file
def write_file(num_of_users):
    f = open("./darknet/data/train.txt", "w")
    for user in range(num_of_users):
        f.write("data/dog.jpg\n")
    f.close()
#Calls YOLO execution with number of images and returns the time taken for execution
def run_yolo(num_of_users, ram):
    """
    number_of_users: Number of images being processed
    RAM: in MB
    """
    write_file(num_of_users)
    #take average by running few times
    num_of_execution = 1 #number of times the programs needs to be aveaged
    total_time = 0
    for i in range(num_of_execution):
        begin = time.time() 
        os.popen("cd darknet; sudo systemd-run --scope -p MemoryMax={}M ./darknet detector test ./cfg/coco.data ./cfg/yolov3.cfg ./yolov3.weights -dont_show < data/train.txt > result.txt 2>&1".format(ram)).read()
        end = time.time()
        total_time += end - begin
    avg_time = total_time/num_of_execution
    return avg_time #returns the total time taken

#workload generation function
def generate_cpu_workload(workload_cpu_stress, num_cpu_cores): #workload in percentage
    workload_pid = os.popen("stress-ng -c {} -l {} > /dev/null 2>&1 & echo $!".format(num_cpu_cores, workload_cpu_stress)).read().strip('\n')
    print('PID of workload is {} and number of cores used is {}'.format(workload_pid, num_cpu_cores))
    return workload_pid

#generate the workload for GPU
def generate_gpu_workload(workload_gpu_stress):
    matrix = workload_gpu_stress #size of the matrix 7000
    workload_gpu_pid = os.popen("./workload {} {} {} > /dev/null 2>&1 & echo $!".format(matrix, matrix, matrix)).read().strip('\n')
    return workload_gpu_pid

#kill the generated workload
def kill_workload_cpu(workload_cpu_pid):
    os.popen("kill -9 {}".format(workload_cpu_pid))


#kill the GPU generated workload
def kill_workload_gpu(workload_gpu_pid):
    os.popen("kill -9 {}".format(workload_gpu_pid))

#Limit the number of cpu cores
def limit_cpu_core(num_of_cores):
    core_list = '' # generate list of cores: string
    for cores in range(num_of_cores):
        core_list += f'{cores}'
        if cores < num_of_cores-1:
            core_list += ','

    #Taskset CPU to partiicular cores
    bash_pid = os.getppid() #get the ID of bash program
    try:
        os.popen("taskset -cp {} {}".format(core_list, bash_pid)).read() #limit the cores
        os.popen("taskset -cp {} {}".format(core_list, os.getpid())).read()
    except:
        sys.exit("Please check if -taskset- is working correctly!")
    return bash_pid

#check gpu workload
def check_gpu(workload_gpu_pid):
    try:
        workload_res = os.popen("nvidia-smi | awk '/{}/{{print $8}}'".format(workload_gpu_pid)).read()
        print("gpuworkload", workload_res)
        workload_gpu = (float(re.findall(r'[0-9 .]+', workload_res)[0])/2000)*100
    except:
        workload_gpu = '' #no values found
    return workload_gpu #in percentage

# ===============================================================Final Wrapper Function

#final yolo call function with limits on number of cores, ram and workload
def yolo_execution_time(num_of_users, num_of_cores, ram, workload_cpu, workload_gpu):
    bash_pid = limit_cpu_core(num_of_cores) #limit the cpu cores
    workload_cpu_pid = generate_cpu_workload(workload_cpu, num_of_cores) #start workload
    workload_gpu_pid = generate_gpu_workload(workload_gpu)
    
    time.sleep(2)
    wl_gpu = check_gpu(workload_gpu_pid) #check workload GPU use
    
    exe_time = run_yolo(num_of_users, ram) #start yolo execution
    
    
    kill_workload_cpu(workload_cpu_pid)  #end CPU workload
    print('cpu_killed')
    kill_workload_gpu(workload_gpu_pid) #end GPU workload
    
    return exe_time, wl_gpu #return the execution time

print("\n>>>>Press 1 to skip system checks:")
system_check = int(input())
if system_check!=1:
    # ===================================Check System and Functions
    print(">>>>>>Staring Function and System Checks<<<<<<")

    #Turnoff the swap
    try:
        print(">>>>Turning off swap")
        os.popen("sudo swapoff -a")
        print("Swap off Done!")
    except:
        sys.exit("Turn off swap memory: Failure")

    #check cpu core limit
    try:
        print(">>>>testing core limit function")
        bash_pid = limit_cpu_core(3)
        print("CPU Core limit working fine!")
    except:
        sys.exit("CPU core limiting function not working correctly: Failure")


    # #Test run_yolo function
    try:
        print(">>>>Testing YOLO")
        run_time = run_yolo(3, 2000) #RAM in MB
        print("YOLO working! Time taken: {}s with 2000MB RAM".format(run_time))
    except:
        sys.exit("Check run_yolo function: Failure")
        
    #Test workload
    try:
        print(">>>>Generating CPU Workload")
        workload_cpu_pid = generate_cpu_workload(40, 3)
        time.sleep(8)
        kill_workload_cpu(workload_cpu_pid)
        print("Done!")
        print(">>>>Generating GPU Workload")
        
        workload_gpu_pid = generate_gpu_workload(8000)
        time.sleep(10)
        wk_load = check_gpu(workload_gpu_pid)
        # print("Done!")
        print("GPU WOrkload:{}; done!".format(wk_load))
        kill_workload_gpu(workload_gpu_pid)

        print("Workload testing done!")
    except:
        sys.exit("Workload not working properly, kill the workloads manually: Failure")
        
    #test Final wrapper funtion
    try:
        print(">>>> Final wrapper function test!")
        exe_time, wk_g = yolo_execution_time(3, 3, 2000, 40, 5000)
        print("Done! Time:{}s; GPU Workload: {}".format(exe_time, wk_g))
    except:
        sys.exit("Failure: wrapper function not working")
        
    print(">>>>Done System Test!")
else:
    print(">>>>Skipping System Checks!")




#==============================RUN CODE TO COLLECT DATA
print("\n>>>>Press 1 to continue or any other number to ABORT:")
check = int(input())
if check!=1:
    sys.exit("Bye! Please check the background processes once...")


#open CSV file to store data
fields = ['ram', 'cores', 'workload_cpu','workload_gpu', 'users', 'time'] #RAM in MB #cores in numbers #workload_cpu in percentage #users in numbers #time in seconds

#Create the CSV file
filename = 'yolo_data_base.csv'
if not os.path.exists(filename):
    print('>>>>Creating new CSV file')
    csvfile = open(filename, 'a')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
else:
    print('>>>>Opening the CSV file')
    csvfile = open(filename, 'a')
    csvwriter = csv.writer(csvfile)

#==================================Generate dataset
num_epoch = 1

ram_low = 2000
ram_high = 5000
ram_resolution = 1000

cores_high = 4
cores_low = 1

workload_high = 60
workload_low = 30
workload_resolution = 10

gpu_high = 7000
gpu_low = 1000
gpu_res = 2000

users_max = 2
users_min = 1
users_resolution = 1

for epoch in range(num_epoch):
    for ram in range(ram_low, ram_high, ram_resolution):
        for cores in range(cores_low, cores_high, 1):
            for workload_cpu in range(workload_low, workload_high, workload_resolution):
                for workload_gpu in range(gpu_low, gpu_high, gpu_res):
                    for users in range(users_min, users_max, users_resolution):
                        print("Running YOLO for>>>>RAM:{}, Cores:{}, Workload:{}%, GPU:{}, Users:{}".format(ram, cores, workload_cpu, workload_gpu, users))
                        #execute the yolo #num_of_users, num_of_cores, ram, workload
                        exe_time, wl_gpu = yolo_execution_time(users, cores, ram, workload_cpu, workload_gpu)
                        #store data to file
                        data_row = [ram, cores, workload_cpu, wl_gpu, users, exe_time] #format data for csv
                        csvwriter.writerow(data_row) #write the data

csvfile.close() #close the csv file
print(">>>>>>>>>>DONE!<<<<<<<<<<")
print("check background processes once!")



