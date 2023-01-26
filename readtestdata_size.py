### 
'''
Read test data and form into a 1-D data input
'''
"""
@author: czhao29
"""



import xlrd
import numpy as np
import sys
np.set_printoptions(precision=2)
np.set_printoptions(threshold=sys.maxsize)

def read_excel(file_name,sample_num,training_num,feature_num, test_num, validation_num):
    wb = xlrd.open_workbook(filename=file_name)
    sheet1 = wb.sheet_by_index(0)
    sheet2 = wb.sheet_by_index(1)
    sheet3 = wb.sheet_by_index(2)
    sheet4 = wb.sheet_by_index(3)
    x_sample = np.zeros((training_num,feature_num))
    y_sample = np.zeros((training_num))
    x_test = np.zeros((validation_num, feature_num))
    y_test = np.zeros((validation_num))
    Count_Sample = 0
    Count_Test = 0
    for i in range(sample_num):
        cycle_num = int(sheet1.cell(i+1,5).value)
        for j in range(cycle_num):
            row_index = i + 1       # except the title row
            column_start = 6
            column_index = column_start + j
            x_sample[Count_Sample+j, 0] = sheet1.cell(row_index,column_index).value
            x_sample[Count_Sample+j, 1] = sheet1.cell(row_index, 0).value
            x_sample[Count_Sample+j, 2] = sheet1.cell(row_index, 1).value
            x_sample[Count_Sample+j, 3] = sheet1.cell(row_index, 2).value
            x_sample[Count_Sample+j, 4] = sheet1.cell(row_index, 3).value
            x_sample[Count_Sample+j, 5] = sheet1.cell(row_index, 4).value
            
            y_sample[Count_Sample+j] = sheet2.cell(row_index,column_index).value
        Count_Sample = Count_Sample + cycle_num    
    for i in range(test_num):
        cycle_num = int(sheet3.cell(i+1,5).value)
        for j in range(cycle_num):
            row_index = i + 1       # except the title row
            column_start = 6
            column_index = column_start + j
            x_test[Count_Test+j, 0] = sheet3.cell(row_index,column_index).value
            x_test[Count_Test+j, 1] = sheet3.cell(row_index, 0).value
            x_test[Count_Test+j, 2] = sheet3.cell(row_index, 1).value
            x_test[Count_Test+j, 3] = sheet3.cell(row_index, 2).value
            x_test[Count_Test+j, 4] = sheet3.cell(row_index, 3).value
            x_test[Count_Test+j, 5] = sheet3.cell(row_index, 4).value
            
            y_test[Count_Test+j] = sheet4.cell(row_index,column_index).value
        Count_Test = Count_Test + cycle_num           
    return x_sample,y_sample,x_test,y_test

def filter_regularize(file_name, x_sample,y_sample,x_test,sample_num,test_num):
    wb = xlrd.open_workbook(filename=file_name)
    sheet1 = wb.sheet_by_index(0)
    sheet3 = wb.sheet_by_index(2)
    Temp_max = np.amax(x_sample[:, 1])
    DISC_max = np.amax(x_sample[:, 2])
    SOCL_max = np.amax(x_sample[:, 3])	
    SOCH_max = np.amax(x_sample[:, 3])
    Type_max = np.amax(x_sample[:, 5])
    Capacity_max = np.amax(y_sample)

    Temp_min = np.amin(x_sample[:, 1])
    DISC_min = np.amin(x_sample[:, 2])
    SOCL_min = np.amin(x_sample[:, 3])
    SOCH_min = np.amin(x_sample[:, 4])
    Type_min = np.amin(x_sample[:, 5])
    Capacity_min = np.amin(y_sample)

    #print(Temp_min, DISC_min, SOCL_min, SOCH_min, Type_min, Capacity_min)
    #print(Temp_max, DISC_max, SOCL_max, SOCH_max, Type_max, Capacity_max)
    Count_Sample = 0
    Count_Test = 0
    for i in range(sample_num):
        cycle_num = int(sheet1.cell(i+1,5).value)
        for j in range(cycle_num):
            #x_sample[i, j, 0] = x_sample[i, j, 0]/ (cycle_num-1)
            x_sample[Count_Sample+j, 1] = x_sample[Count_Sample+j, 1]/Temp_max
            x_sample[Count_Sample+j, 2] = x_sample[Count_Sample+j, 2]/DISC_max
            x_sample[Count_Sample+j, 3] = x_sample[Count_Sample+j, 3]/SOCH_max
            x_sample[Count_Sample+j, 4] = x_sample[Count_Sample+j, 4]/SOCH_max
            x_sample[Count_Sample+j, 5] = x_sample[Count_Sample+j, 5]/Type_max
            #y_sample[i,j] = y_sample[i,j]/Capacity_max
        Count_Sample = Count_Sample + cycle_num     
    for i in range(test_num):
        cycle_num = int(sheet3.cell(i+1,5).value)
        for j in range(cycle_num):
            #x_sample[i, j, 0] = x_sample[i, j, 0]/ (cycle_num-1)
            x_test[Count_Test+j, 1] = x_test[Count_Test+j, 1]/Temp_max
            x_test[Count_Test+j, 2] = x_test[Count_Test+j, 2]/DISC_max
            x_test[Count_Test+j, 3] = x_test[Count_Test+j, 3]/SOCH_max
            x_test[Count_Test+j, 4] = x_test[Count_Test+j, 4]/SOCH_max
            x_test[Count_Test+j, 5] = x_test[Count_Test+j, 5]/Type_max
            #y_sample[i,j] = y_sample[i,j]/Capacity_max
        Count_Test = Count_Test + cycle_num   
    return x_sample,y_sample,x_test,y_test





def data_shuffle(x_sample,y_sample,sample_num,cycle_num,feature_num):
    indices = np.arange(sample_num)
    np.random.shuffle(indices)
    x_sample_rd = np.zeros((sample_num, cycle_num, feature_num))
    y_sample_rd = np.zeros((sample_num, cycle_num))
    sample_index = 0
    for i in indices:
        x_sample_rd[sample_index,:,:] = x_sample[i,:,:]
        y_sample_rd[sample_index, :] = y_sample[i, :]
        sample_index += 1
    return x_sample_rd,y_sample_rd



sample_num = 765
feature_num = 6
test_num = 179
training_num = 3069608 #2949728 #1009578
validation_num = 748624# 728644 #249049


file = '10080 with extra 0.xlsx'
x_sample,y_sample,x_test,y_test = read_excel(file,sample_num,training_num,feature_num, test_num, validation_num)
x_sample,y_sample,x_test,y_test = filter_regularize(file, x_sample,y_sample,x_test,sample_num,test_num)
#print(y_sample)
#x_sample,y_sample = data_shuffle(x_sample,y_sample,sample_num,cycle_num,feature_num)