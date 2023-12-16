import os
import numpy as np
import pandas as pd

class data_process():
    def __init__(self) -> None:
        pass
    
    def Import_Airfoils(self, dataDir :str):
        fileLst = os.listdir(dataDir)
        airfoilData = []
        for fileName in fileLst:
            filepath = dataDir + fileName
            data = np.loadtxt(filepath, skiprows=1)
            airfoil = fileName.split('.')[0]

            df = pd.DataFrame(data).rename(columns={0:'x',1:'y'})
            num_left = 150 - len(df)
            point_l1 = []
            point_l2 = []
            point_l3 = []
            if num_left < len(df):
                for i in range(num_left):
                    point_l1.append([np.mean([df['x'][i],df['x'][i+1]]),
                                        np.mean([df['y'][i],df['y'][i+1]])])
                point = np.concatenate([data,point_l1])

            elif num_left//2 < len(df):
                num_d2 = num_left - num_left//2
                for i in range(num_left//2):
                    point_l1.append([np.mean([df['x'][i],df['x'][i+1]]),
                                        np.mean([df['y'][i],df['y'][i+1]])])
                df_l1 = pd.DataFrame(np.concatenate([data,point_l1])).rename(columns={0:'x',1:'y'})
                for i in range(num_d2):
                    point_l2.append([np.mean([df_l1['x'][i],df_l1['x'][i+1]]),
                                        np.mean([df_l1['y'][i],df_l1['y'][i+1]])])
                point = np.concatenate([data,point_l1,point_l2])

            elif num_left//3 < len(df):
                num_d3 = num_left - num_left//3
                for i in range(num_left//3):
                    point_l1.append([np.mean([df['x'][i],df['x'][i+1]]),
                                        np.mean([df['y'][i],df['y'][i+1]])])
                df_l1 = pd.DataFrame(np.concatenate([data,point_l1])).rename(columns={0:'x',1:'y'})
                for i in range(num_d3):
                    point_l2.append([np.mean([df_l1['x'][i],df_l1['x'][i+1]]),
                                        np.mean([df_l1['y'][i],df_l1['y'][i+1]])])
                df_l2 = pd.DataFrame(np.concatenate([data,point_l1,point_l2])).rename(columns={0:'x',1:'y'})
        
                point = np.concatenate([data,point_l1,point_l2])

            elif num_left//4 <len(df):
                for i in range(num_left//4):
                    point_l1.append([np.mean([df['x'][i],df['x'][i+1]]),
                                        np.mean([df['y'][i],df['y'][i+1]])])
                df_l1 = pd.DataFrame(np.concatenate([data,point_l1])).rename(columns={0:'x',1:'y'})
                num_l1 = num_left - num_left//4
                if num_l1 < len(df_l1):
                    for i in range(num_l1):
                        point_l2.append([np.mean([df_l1['x'][i],df_l1['x'][i+1]]),
                                            np.mean([df_l1['y'][i],df_l1['y'][i+1]])])
                    point = np.concatenate([data,point_l1,point_l2])
                else:
                    for i in range(num_l1//2):
                        point_l2.append([np.mean([df_l1['x'][i],df_l1['x'][i+1]]),
                                            np.mean([df_l1['y'][i],df_l1['y'][i+1]])])
                    num_l2 = num_l1 - num_l1//2
                    df_l2 = pd.DataFrame(np.concatenate([data,point_l1,point_l2])).rename(columns={0:'x',1:'y'})
                    for i in range(num_l2):
                        point_l3.append([np.mean([df_l2['x'][i],df_l2['x'][i+1]]),
                                            np.mean([df_l2['y'][i],df_l2['y'][i+1]])])

                point = np.concatenate([data,point_l1,point_l2,point_l3])
            airfoilData.append([airfoil, point])
        return airfoilData
    
    def load_data(self, dataDir : str, airfoilDir : str):
        # 將 airfoil 的 point 存進原始 dataFrame 中
        airfoil = self.Import_Airfoils(dataDir=airfoilDir)
        airfoil_df = pd.DataFrame(airfoil).rename(columns={0:'Airfoils',1:'point'})

        fileLst = os.listdir(dataDir)
        tmp = []
        for fileName in fileLst:
            filepath = dataDir + fileName
            data = np.load(filepath)
            airfoil = fileName.split('_')[0]
            tmp.append([airfoil,data['Vel'],
                        data['upper'][0],data['upper'][1], 
                        data['lower'][0],data['lower'][1]])
        
        data_df = pd.DataFrame(tmp).rename(columns={0:'Airfoils',1:'Velocity',2:'upperx',3:'uppercf',4:'lowerx',5:'lowercf'})
        merged_df = pd.merge(data_df,airfoil_df, on='Airfoils', how='left').set_index('Airfoils').dropna()
        # 分成 input 及 target
        input_list = []
        target_list = []
        for index, row in merged_df.iterrows():
            velocity = row['Velocity'].tolist()
            point = row['point'].tolist()
            point_to_list = [item for sublist in point for item in sublist]
            input_list.append(point_to_list + velocity)
            target_to_list = row[['uppercf','lowercf']].tolist()
            target_tmp = [item for sublist in target_to_list for item in sublist]
            target_list.append(target_tmp)
        model_target = np.asarray(target_list)
        model_input = np.asarray(input_list)

        return model_input, model_target
    