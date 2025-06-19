import numpy as np
from numpy import *
import random
import time
import sys
import copy
import math
from sklearn.svm import SVC,LinearSVC
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
import sklearn
from sklearn import metrics
from sklearn.metrics import plot_confusion_matrix, confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt
from sklearn import preprocessing as p
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
from imblearn.over_sampling import SMOTE


class Fir_stage:
    def __init__(self,DNA_SIZE,POP_SIZE,require,CROSS_RATE,MUTATION_RATE, N_GENERATIONS,Gamma_Bound,Cost_min_Bound,Cost_maj_Bound,split
                 ,demand_num,select_method_num,tourn_num):
        self.DNA_SIZE = DNA_SIZE  # DNA length
        self.POP_SIZE = POP_SIZE  # population size
        self.require_select = require
        self.CROSS_RATE = CROSS_RATE  # mating probability (DNA crossover)
        self.MUTATION_RATE = MUTATION_RATE  # mutation probability
        self.N_GENERATIONS = N_GENERATIONS
        self.demand_num = demand_num
        self.select_method_num = select_method_num
        self.tourn_num = tourn_num
        self.predicted =[]
        #------------------------------------------------------------------------------------------------
        print("set fir")
        self.Gamma_Bound = Gamma_Bound  # x upper and lower bounds
        self.Cost_min_Bound = Cost_min_Bound
        self.Cost_maj_Bound = Cost_maj_Bound
        self.split = split
        #------------------------------np.ndarray-------------------------------------------------------------------
        self.Gamma_gene_pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE))
        self.Cmin_gene_pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE))
        self.Cmaj_gene_pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE))
        #--------------------------------------------------------------------------------------------------
        self.best_fitness_in_generation=[]
        self.best_combination_in_generation=[]
        self.best_results_in_generation = []
        self.qualified_fitness = [] #used for constraint learning
        self.start_time = time.time()
        #--------------------------------------------------------------------------------------------------
        self.data = pd.read_csv('Yelp_total_data.csv', error_bad_lines=False, index_col=0);  # index_col remove unamed
        distance =self.data.drop(columns=['True_y'])
        label = self.data['True_y']

        self.x_train_1, self.x_test_1, self.y_train_1, self.y_test_1 = train_test_split(distance, label, test_size=self.split, random_state=1)
        train_data = self.x_train_1
        train_data['True_y'] = self.y_train_1
        train_data = train_data.reset_index()

        train_data = train_data.drop(columns=['index'])
        self.x_train_r = train_data.iloc[:, train_data.columns != 'True_y']
        self.y_train_r = train_data.iloc[:, -1]


    def Create_Gamma_Gene(self,pop):
        print('Gamma')
        G_pop = pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * self.Gamma_Bound[1]
        while G_pop.all() == 0:
            self.Gamma_gene_pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE))
            G_pop = self.Gamma_gene_pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2**self.DNA_SIZE-1) * self.Gamma_Bound[1]
        return np.around(G_pop, decimals=4)


    def Create_Cost_Min_Gene(self,pop):
        Cmin_pop = pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * self.Cost_min_Bound[1]
        while Cmin_pop.astype(int).all() == 0:
            self.Cmin_gene_pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE))
            Cmin_pop = self.Cmin_gene_pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * \
                       self.Cost_min_Bound[1]
        return Cmin_pop.astype(int)


    def Create_Cost_Maj_Gene(self,pop):
        Cmaj_pop = pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * self.Cost_maj_Bound[1]
        while Cmaj_pop.astype(int).all() == 0:
            self.Cmaj_gene_pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE))
            Cmaj_pop = self.Cmaj_gene_pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * self.Cost_maj_Bound[1]
        return Cmaj_pop.astype(int)


    def execute(self):
        G_pop = self.Create_Gamma_Gene(self.Gamma_gene_pop)
        Cmin_pop = self.Create_Cost_Min_Gene(self.Cmin_gene_pop)
        Cmaj_pop = self.Create_Cost_Maj_Gene(self.Cmaj_gene_pop)
        print("what")
        for i in range(0 , self.N_GENERATIONS):
            print("This is {0} generation%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(i+1))
            if i != 0:
                pop_1=[]
                pop_2=[]
                pop_3=[]
                for k in range(self.POP_SIZE):
                    pop_1.append(list(child_pops[k][0])) #gamma
                    pop_2.append(list(child_pops[k][1])) #min
                    pop_3.append(list(child_pops[k][2])) #maj
                G_pop =  self.Create_Gamma_Gene(np.array(pop_1))
                Cmin_pop = self.Create_Cost_Min_Gene(np.array(pop_2))
                Cmaj_pop = self.Create_Cost_Maj_Gene(np.array(pop_3))
            fitness_points=[]
            results = []
            F_points =[]
            R_points = []
            specificity_points =[]
            precision_points = []
            recall_points = []
            parents = []
            parents_nums = []
            child_pops = []
            for pop_num in range(0 , self.POP_SIZE):
                print("This is {0} pop in SVM+++++++++++++++++++++++++++++++++++++++++++++++++++".format(pop_num+1))
                fitness,result= self.get_fitness(G_pop[pop_num], Cmin_pop[pop_num], Cmaj_pop[pop_num])
                fitness_points.append(fitness)
                results.append(result)

            if self.demand_num == 1 :
                for pop_num in range(0, self.POP_SIZE):
                    F_points.append(fitness_points[pop_num][0])   #F_measure
                for pop_num in range(0, self.POP_SIZE):
                    precision_points.append(fitness_points[pop_num][1])
            elif self.demand_num == 2:
                for pop_num in range(0, self.POP_SIZE):
                    R_points.append(fitness_points[pop_num][0])   #recall
                for pop_num in range(0, self.POP_SIZE):
                    precision_points.append(fitness_points[pop_num][1])
            elif self.demand_num == 3:
                for pop_num in range(0, self.POP_SIZE):
                    specificity_points.append(fitness_points[pop_num][0])  # specificity
                for pop_num in range(0, self.POP_SIZE):
                    recall_points.append(fitness_points[pop_num][1])              # recall



            # if np.isnan(np.array(target_points)).any():
            #     id_nan = np.isnan(np.array(target_points))
            #     np.array(target_points)[id_nan] = 0
            # TODO -------------find best f_measure-----------------------------demand_num : 1
            if self.demand_num == 1:
                print("This is the best fitness_point(F_measure) : {0}".format(max(F_points)))
                self.best_fitness_in_generation.append(max(F_points))
                best_id = self.best_combination_each_generation(max(F_points),F_points)
                print("This members : Gamma->{0} Cmin->{1} Cmaj->{2}".format(G_pop[best_id],Cmin_pop[best_id],Cmaj_pop[best_id]))
                self.best_combination_in_generation.append([G_pop[best_id],Cmin_pop[best_id],Cmaj_pop[best_id]])
                if self.select_method_num == 1:
                     parents_nums = self.select(F_points)
                elif self.select_method_num == 2:
                     parents_nums = self.tournment_select(fitness_points,self.tourn_num)

                # TODO-----------find best recall constraint to precision[0.45, 0.55]------------------demand_num : 2
            elif self.demand_num == 2:
                print("find best f_measure constraint to precision[0.45, 0.55]")
                fitness_sets = copy.deepcopy(fitness_points)
                fitness_points.sort()
                results.sort()
                best_combination = self.best_combination_each_generation_constraint(fitness_points)
                best_results = self.best_combination_each_generation_constraint(results)
                print("This is the best fitness_point(Recall) : {0}".format(best_combination[0]))
                print("This members : Gamma->{0} Cmin->{1} Cmaj->{2}".format(best_combination[2], best_combination[3],best_combination[4]))
                self.best_fitness_in_generation.append(best_combination[0])
                self.best_combination_in_generation.append([best_combination[2], best_combination[3], best_combination[4]])
                self.best_results_in_generation.append([best_results[2],best_results[3],best_results[4],best_results[5]
                                                    ,best_results[6],best_results[7],best_results[8]])
                #[X,X,accuracy,precision,recall,specificity,F_measure,G_mean,auc]]

                targets = self.filter_combination_each_generation_constraint(fitness_points,R_points)
                if self.select_method_num == 1:
                    parents_nums = self.constraint_select(targets, fitness_sets)
                elif self.select_method_num == 2:
                    parents_nums = self.tournment_select(fitness_sets,self.tourn_num)

                # TODO-----------find best specifity constraint to recall[0.45, 0.55] by 史------------------demand_num : 3
            elif self.demand_num == 3:
                for pop_num in range(0, self.POP_SIZE):
                    specificity_points.append(fitness_points[pop_num][0])
                print("find best specifity constraint to recall[0.45, 0.55]")
                fitness_sets = copy.deepcopy(fitness_points)
                fitness_points.sort()
                results.sort()
                best_combination = self.best_combination_each_generation_constraint(fitness_points)
                best_results = self.best_combination_each_generation_constraint(results)
                print("This is the best fitness_point(specificity) : {0}".format(best_combination[0]))
                print("This members : Gamma->{0} Cmin->{1} Cmaj->{2}".format(best_combination[2], best_combination[3],
                                                                             best_combination[4]))
                self.best_fitness_in_generation.append(best_combination[0])
                self.best_combination_in_generation.append([best_combination[2], best_combination[3], best_combination[4]])
                self.best_results_in_generation.append(
                    [best_results[2], best_results[3], best_results[4], best_results[5]
                        , best_results[6], best_results[7], best_results[8]])
                targets = self.filter_combination_each_generation_constraint(fitness_points, specificity_points)
                if self.select_method_num == 1:
                     parents_nums = self.constraint_select(targets, fitness_sets)
                elif self.select_method_num == 2:
                    parents_nums = self.tournment_select(fitness_sets,self.tourn_num)






            for num in parents_nums:  # find the parents (high fitness pop)
                parents.append([self.Gamma_gene_pop[num]
                                            , self.Cmin_gene_pop[num]
                                            , self.Cmaj_gene_pop[num]])
            parents_copy = parents.copy()
            cross_num = 0

            for parent in parents:
                # parent = np.array(parent)
                p_range = self.POP_SIZE/(self.require_select*2)
                # print(p_range)
                for i in range(int(p_range)):
                    childs = self.crossover(parent,parents_copy,cross_num)
                    child_1 = self.mutate(childs[0])
                    print("child_1+++++++++++")
                    print(type(child_1))
                    print(child_1)
                    child_pops.append(child_1)
                    child_2 = self.mutate(childs[1])
                    print("child_2++++++++++++++")
                    print(type(child_2))
                    print(child_2)
                    child_pops.append(child_2)
                cross_num = cross_num + 1

    # fitness fir
    def get_fitness(self,gamma,min_c,maj_c):

        precision = 0
        recall = 0
        F_measure =0
        specificity = 0
        G_mean = 0
        cheeseburger = []
        gamma_pump = gamma
        if gamma == 0:
            gamma_pump = 'auto'
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("YOOOOOOO WTF gamma is {0} ,min_c is {1} , maj_C is {2}".format(gamma_pump, min_c, maj_c))
        clf = LinearSVC(class_weight={0: min_c, 1: maj_c}, dual=False)
        predicted = cross_val_predict(clf, self.x_train_r, self.y_train_r, cv=2)
        fir_matrix_train = confusion_matrix(self.y_train_r, predicted,labels = [0,1])
        print(fir_matrix_train)
        TP = fir_matrix_train[0][0]
        FN = fir_matrix_train[0][1]
        FP = fir_matrix_train[1][0]
        TN = fir_matrix_train[1][1]
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print("accuracy : {0}".format(accuracy))
        if TN != 0:
            specificity = TN /(TN+FP)
        if TP != 0:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F_measure = 2 * recall * precision / (recall + precision)
            G_mean =  math.sqrt(recall*precision)

        print("Precision : {0}".format(precision))
        print("Recall : {0}".format(recall))
        auc = roc_auc_score(self.y_train_r, predicted)
        if self.demand_num == 1 :
            cheeseburger = [[np.around(F_measure, decimals=4),np.around(precision ,decimals=4), gamma ,min_c ,maj_c],
                            [np.around(F_measure, decimals=4),np.around(precision ,decimals=4),accuracy,precision,recall,specificity,F_measure,G_mean,auc]]
        elif self.demand_num == 2:
            cheeseburger = [[np.around(recall, decimals = 4), np.around(precision ,decimals=4), gamma, min_c, maj_c],
                            [np.around(recall, decimals = 4),np.around(precision ,decimals=4),accuracy, precision, recall, specificity, F_measure, G_mean, auc]]
        elif self.demand_num == 3:
            cheeseburger = [[np.around(specificity, decimals = 4), np.around(recall , decimals =4), gamma, min_c, maj_c],
                            [np.around(specificity, decimals = 4), np.around(recall , decimals =4),accuracy, precision, recall, specificity, F_measure, G_mean, auc]]
        print(cheeseburger)
        return cheeseburger


    def best_combination_each_generation(self,best,target):
        id = 0
        for i in range(self.POP_SIZE):
            if best == target[i]:
                id = i
                break
        return id
    def best_combination_each_generation_constraint(self,fitness_sets):
        if self.demand_num == 2:
            for k in range(self.POP_SIZE):
                if 0.45<=fitness_sets[k][1] and fitness_sets[k][1]<=0.55:
                    print("prime precision")
                    print(fitness_sets[k][1])
                    return fitness_sets[k]
        elif self.demand_num == 3:
            for k in range(self.POP_SIZE):
                if  0.6<fitness_sets[k][1]:
                    print("prime recall")
                    print(fitness_sets[k][1])
                    return fitness_sets[k]
        return fitness_sets[0]
    def filter_combination_each_generation_constraint(self,fitness_sets,target_points):
        qualified_fitness = []
        if self.demand_num == 2:
            for k in range(self.POP_SIZE):
                if 0.45 <= fitness_sets[k][1] and fitness_sets[k][1] <= 0.55:
                    print(fitness_sets[k][1])
                    qualified_fitness.append(fitness_sets[k][0])
        elif self.demand_num == 3:
            for k in range(self.POP_SIZE):
                if 0.6<fitness_sets[k][1]:
                    print("boys")
                    print(fitness_sets[k][1])
                    qualified_fitness.append(fitness_sets[k][0])
        if len(qualified_fitness) == 0 :
            print("ak47")
            return target_points
        elif len(qualified_fitness) >= self.require_select:
            return qualified_fitness
        elif len(qualified_fitness) < self.require_select:
            print("type81")
            while  len(qualified_fitness) < self.require_select:
                for k in range(self.POP_SIZE):
                    if self.demand_num ==2:
                        if 0.45 <= fitness_sets[k][1] and fitness_sets[k][1] <= 0.55:
                            print(fitness_sets[k][1])
                            qualified_fitness.append(fitness_sets[k][0])
                    elif self.demand_num == 3:
                        if 0.6 < fitness_sets[k][1]:
                            qualified_fitness.append(fitness_sets[k][0])
                            print("chinese")
                    if len(qualified_fitness) == self.require_select:
                        break

            return qualified_fitness
    def tournment_select(self, fitness_points, tourn_num):
        orgy_party = []
        most_want = []
        probability = []
        num = np.arange(self.POP_SIZE)
        combine = []
        total_fitness_points = sum(fitness_points[i][0] for i in range(self.POP_SIZE))
        final = 0

        for i in range(self.POP_SIZE):
            probability.append(np.round(fitness_points[i][0] / total_fitness_points, decimals=4))
            combine = combine + [[num[i], probability[i]]]

        for k in range(self.require_select):
            print(k)
            for a in range(tourn_num):

                if a == (tourn_num - 1):
                    final = random.choices(combine, probability, k=int(self.POP_SIZE / tourn_num) * (tourn_num - a))
                    print("final")
                    print(final)
                    break
                ak = random.choices(combine, probability, k=int(self.POP_SIZE / tourn_num) * (tourn_num - a))
                combine = copy.deepcopy(ak)

                probability = [combine[i][1] for i in range(len(combine))]


            orgy_party.append(final[0][0])

        return orgy_party
    def constraint_select(self,target,fitness_points):  # nature selection wrt pop's fitness
        brazzer = []
        most_wanted = target[-(self.require_select):]
        for wanted in most_wanted:
            for pop_num in range(0, self.POP_SIZE):
                if fitness_points[pop_num][0] == wanted:
                    brazzer.append(pop_num)
                    break
        return brazzer
    def select(self,target):  # nature selection wrt pop's fitness
        brazzer = []
        TS = np.sort(target)
        most_wanted = TS[-(self.require_select):]
        for wanted in most_wanted:
            for pop_num in range(0, self.POP_SIZE):
                if target[pop_num] == wanted:
                    brazzer.append(pop_num)
                    break
        return brazzer
    def crossover(self,mate_1,mate_group,cross_num):  # mating process (genes crossover)
        child_1 = 0
        child_2 = 0
        cross_points_1 = 0
        cross_points_2 = 0
        pop_num = random.randrange(0,self.require_select)
        if np.random.rand() < self.CROSS_RATE:
            while pop_num == cross_num:
                pop_num = random.randrange(0,self.require_select)  # ａｖｏｉｄ　find　ｙｏｕｒｓｅｌｆ
            while (cross_points_1 >= cross_points_2) or abs(cross_points_2 - cross_points_1)<3 :
                cross_points_1 = np.random.randint(0, self.DNA_SIZE)  # choose crossover points
                cross_points_2 = np.random.randint(0, self.DNA_SIZE+1)  # choose crossover points
            child_1 = copy.deepcopy(mate_1)  #array
            child_2 = copy.deepcopy(mate_group[pop_num])
            for type in range(3):
                child_1[type][cross_points_1:cross_points_2] = mate_group[pop_num][type][cross_points_1:cross_points_2]
                child_2[type][cross_points_1:cross_points_2] = mate_1[type][cross_points_1:cross_points_2]
            return [list(child_1), child_2]
        print('------still virgin-----')
        return [mate_1,mate_group[pop_num]]

    def mutate(self,child):
        for type in range(3):
            for point in range(self.DNA_SIZE):
                if np.random.rand() < self.MUTATION_RATE:
                    if child[type][point] == 0:
                        child[type][point] = 1
                    else:
                        child[type][point] = 0

        return child

    def show_fir_result(self):
        precision = 0
        recall = 0
        F_measure = 0
        specificity = 0
        G_mean = 0

        self.min_c = 0
        self.maj_c = 0
        self.gamma_pump = 0
        print("best fitness")
        print(max(self.best_fitness_in_generation))
        for i in range(self.N_GENERATIONS):
            if self.best_fitness_in_generation[i] == max(self.best_fitness_in_generation):
                print("This members : Gamma->{0} Cmin->{1} Cmaj->{2}".format(self.best_combination_in_generation[i][0]
                                                                             ,self.best_combination_in_generation[i][1]
                                                                             ,self.best_combination_in_generation[i][2]))
                self.gamma_pump = self.best_combination_in_generation[i][0]
                self.min_c = self.best_combination_in_generation[i][1]
                self.maj_c = self.best_combination_in_generation[i][2]

                accuracy = self.best_results_in_generation[i][0]
                precision = self.best_results_in_generation[i][1]
                recall = self.best_results_in_generation[i][2]
                specificity = self.best_results_in_generation[i][3]
                F_measure = self.best_results_in_generation[i][4]
                G_mean = self.best_results_in_generation[i][5]
                auc = self.best_results_in_generation[i][6]
                print("Ing results")
                print("accuracy : {0}".format(accuracy))
                print("Precision : {0}".format(precision))
                print("Recall : {0}".format(recall))
                print("specifity : {0}".format(specificity))
                print("F_measure : {0}".format(F_measure))
                print("G_mean : {0}".format(G_mean))
                print("AUC : {0}".format(auc))

                if self.best_combination_in_generation[i][0] == 0:
                    self.gamma_pump = 'auto'
                break

        # sys.exit("Error message")

        clf = LinearSVC(class_weight={0: self.min_c,
                                1: self.maj_c}, dual=False)
        self.predicted = cross_val_predict(clf, self.x_train_r, self.y_train_r, cv=2)
        fir_matrix_train = confusion_matrix(self.y_train_r, self.predicted,labels = [0,1])
        print(fir_matrix_train)
        print("www")
        print(self.predicted)
        TP = fir_matrix_train[0][0]
        FN = fir_matrix_train[0][1]
        FP = fir_matrix_train[1][0]
        TN = fir_matrix_train[1][1]
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print("accuracy : {0}".format(accuracy))
        if TN != 0:
            specificity = TN / (TN + FP)
        if TP != 0:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F_measure = 2 * recall * precision / (recall + precision)
            G_mean = math.sqrt(recall * precision)

        print("Precision : {0}".format(precision))
        print("Recall : {0}".format(recall))
        print("specifity : {0}".format(specificity))
        print("F_measure : {0}".format(F_measure))
        print("G_mean : {0}".format(G_mean))
        auc = roc_auc_score(self.y_train_r, self.predicted)
        print("AUC : {0}".format(auc))
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def create(self):
        sec_data = self.x_train_r
        print("DWAdawdaw")
        print(self.predicted)
        sec_data['True_y'] = self.y_train_r
        sec_data['Pred_y'] = self.predicted
        f = (sec_data['Pred_y'] == 0)  # find the data which is classified as "label 0" at first stage
        sec_min = sec_data[f].reset_index()
        sec_min = sec_min.drop(columns=['index'])
        sec_min.to_csv("Sec_ready_data.csv")
        #-----------------------------------------------------------------------------
        sec_test = self.x_test_1
        sec_test['True_y'] = self.y_test_1
        sec_test = sec_test.reset_index()
        sec_test = sec_test.drop(columns=['index'])
        sec_test.to_csv("Sec_test_data.csv")

    def show_finish_time(self):
        print("----%s minutes----" %(round((time.time()-self.start_time)/60)))

class Sec_Stage:
    "the second stage"

    def __init__(self, DNA_SIZE, POP_SIZE, require, CROSS_RATE, MUTATION_RATE, N_GENERATIONS,gamma_bound,cost_bound,select_method_num,tourn_num):
        self.DNA_SIZE = DNA_SIZE  # DNA length
        self.POP_SIZE = POP_SIZE  # population size
        self.require_select = require
        self.CROSS_RATE = CROSS_RATE  # mating probability (DNA crossover)
        self.MUTATION_RATE = MUTATION_RATE  # mutation probability
        self.N_GENERATIONS = N_GENERATIONS
        self.select_method_num = select_method_num
        self.tourn_num = tourn_num
        # ------------------------------------------------------------------------------------------------
        print("set sec")
        self.Gamma_Bound = gamma_bound # x upper and lower bounds
        self.Cost_Bound = cost_bound
        # ------------------------------np.ndarray-------------------------------------------------------------------
        self.Gamma_gene_pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE))
        self.C_gene_pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE))
        # --------------------------------------------------------------------------------------------------
        self.best_fitness_in_generation = []
        self.best_combination_in_generation = []
        self.best_results_in_generation = []
        self.start_time = time.time()
        # --------------------------------------------------------------------------------------------------
        self.data = pd.read_csv('Sec_ready_data.csv', error_bad_lines=False, index_col=0);  # index_col remove unamed
        self.distance = self.data.drop(columns=['True_y'])
        self.distance = self.distance.drop(columns=['Pred_y'])

        self.label = self.data['True_y']
        #----------------------------------------------------------------------------------------------------
        self.test_data = pd.read_csv('Sec_test_data.csv', error_bad_lines=False, index_col=0);
        self.test_x = self.test_data.drop(columns=['True_y'])
        self.test_y = self.test_data['True_y']




    def Create_Gamma_Gene(self, pop):
        print('Gamma')
        print(pop)
        G_pop = pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * self.Gamma_Bound[1]
        while G_pop.all() == 0:
            self.Gamma_gene_pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE))
            G_pop = self.Gamma_gene_pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * \
                    self.Gamma_Bound[1]
        print(G_pop)
        return np.around(G_pop, decimals=4)

    def Create_Cost_Gene(self, pop):
        print('Cost')
        print(pop)
        C_pop = pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * self.Cost_Bound[1]
        while C_pop.astype(int).all() == 0:
            self.C_gene_pop = np.random.randint(2, size=(self.POP_SIZE, self.DNA_SIZE))
            C_pop = self.C_gene_pop.dot(2 ** np.arange(self.DNA_SIZE)[::-1]) / float(2 ** self.DNA_SIZE - 1) * \
                       self.Cost_Bound[1]
        print(C_pop)
        return C_pop.astype(int)

    def execute(self):
        G_pop = self.Create_Gamma_Gene(self.Gamma_gene_pop)
        C_pop = self.Create_Cost_Gene(self.C_gene_pop)
        for i in range(0, self.N_GENERATIONS):
            print("This is {0} generation%%%%%%%in sec stage%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%".format(i + 1))
            if i != 0:
                print("child_pops")
                print(child_pops)
                pop_1 = []
                pop_2 = []
                for k in range(self.POP_SIZE):
                    print(k)
                    pop_1.append(list(child_pops[k][0]))  # gamma
                    pop_2.append(list(child_pops[k][1]))  # Cost
                G_pop = self.Create_Gamma_Gene(np.array(pop_1))
                C_pop = self.Create_Cost_Gene(np.array(pop_2))
            fitness_points = []
            target_points = []
            precision_points = []
            parents = []
            child_pops = []
            results=[]
            for pop_num in range(0, self.POP_SIZE):
                print("This is {0} pop in SVM+++++++++++++++++++++++++++++++++++++++++++++++++++".format(pop_num + 1))
                fitness, result = self.get_fitness(G_pop[pop_num], C_pop[pop_num])
                fitness_points.append(fitness)
                results.append(result)

            for pop_num in range(0, self.POP_SIZE):
                target_points.append(fitness_points[pop_num][0])  # F_measure


            print("This is the best fitness_point(F_measure) : {0}".format(max(target_points)))
            self.best_fitness_in_generation.append(max(target_points))
            best_id = self.best_combination_each_generation(max(target_points), target_points)
            self.best_results_in_generation.append(results[best_id])
            print("This members : Gamma->{0} C->{1} ".format(G_pop[best_id], C_pop[best_id]))
            self.best_combination_in_generation.append([G_pop[best_id], C_pop[best_id]])
            if self.select_method_num == 1:
                parents_nums = self.select(target_points)
            elif self.select_method_num == 2:
                parents_nums = self.tournment_select(target_points, self.tourn_num)


            for num in parents_nums:  # find the parents (high fitness pop)
                parents.append([self.Gamma_gene_pop[num]
                                   , self.C_gene_pop[num]])

            parents_copy = parents.copy()
            cross_num = 0

            for parent in parents:
                p_range = self.POP_SIZE / (self.require_select * 2)
                print(p_range)
                for i in range(int(p_range)):
                    childs = self.crossover(parent, parents_copy, cross_num)
                    child_1 = self.mutate(childs[0])
                    child_pops.append(child_1)
                    child_2 = self.mutate(childs[1])
                    child_pops.append(child_2)
                cross_num = cross_num + 1

    def get_fitness(self, gamma, cost):
        precision = 0
        recall = 0
        F_measure = 0
        G_mean = 0
        specificity = 0
        cheeseburger = []
        gamma_pump = gamma
        if gamma == 0:
            gamma_pump = 'auto'
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print("YOOOOOOO WTF gamma is {0} ,Cost is {1}".format(gamma, cost))
        clf = SVC(C= cost, gamma=gamma_pump, kernel='rbf')
        #-----------------------------------------------------------------------
        predicted_cv = cross_val_predict(clf, self.distance, self.label, cv=10)
        # --------------------------------------------------------------------
        fir_matrix_train = confusion_matrix(self.label, predicted_cv,labels = [0,1])
        print("CV_matrix")
        print(fir_matrix_train)
        TP = fir_matrix_train[0][0]
        FN = fir_matrix_train[0][1]
        FP = fir_matrix_train[1][0]
        TN = fir_matrix_train[1][1]
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print("accuracy : {0}".format(accuracy))
        if TN != 0:
            specificity = TN / (TN + FP)
        if TP != 0:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F_measure = 2 * recall * precision / (recall + precision)
            G_mean = math.sqrt(recall * precision)

        print("CV_Precision : {0}".format(precision))
        print("CV_Recall : {0}".format(recall))
        print("CV_specifity : {0}".format(specificity))
        print("CV_F_measure : {0}".format(F_measure))
        print("CV_G_mean : {0}".format(G_mean))
        auc = roc_auc_score(self.label, predicted_cv)
        print("CV_AUC : {0}".format(auc))
        #-----------------------------------------------------------------------
        clf.fit(self.distance, self.label)
        predicted = clf.predict(self.test_x)
        print("Predict_matrix")
        fir_matrix_train = confusion_matrix(self.test_y, predicted,labels = [0,1])
        print(fir_matrix_train)
        TP = fir_matrix_train[0][0]
        FN = fir_matrix_train[0][1]
        FP = fir_matrix_train[1][0]
        TN = fir_matrix_train[1][1]
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print("accuracy : {0}".format(accuracy))
        if TN != 0:
            specificity = TN / (TN + FP)
        if TP != 0:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F_measure = 2 * recall * precision / (recall + precision)
            G_mean = math.sqrt(recall * precision)

        print("Precision : {0}".format(precision))
        print("Recall : {0}".format(recall))
        print("specifity : {0}".format(specificity))
        print("F_measure : {0}".format(F_measure))
        print("G_mean : {0}".format(G_mean))
        auc = roc_auc_score(self.test_y, predicted)
        print("AUC : {0}".format(auc))

        cheeseburger = [[F_measure,precision],[accuracy, precision, recall, specificity, F_measure, G_mean, auc]]
        print(cheeseburger)
        return cheeseburger


    def best_combination_each_generation(self, best, target):
        id = 0
        print(target)
        for i in range(self.POP_SIZE):
            print(i)
            if best == target[i]:
                id = i
                break
        return id


    def tournment_select(self, fitness_points, tourn_num):
        orgy_party = []
        most_want =[]
        probability = []
        num = np.arange(self.POP_SIZE)
        combine = []
        total_fitness_points = sum(fitness_points[0] for i in range(self.POP_SIZE))
        final = 0

        for i in range(self.POP_SIZE):
            probability.append(np.round(fitness_points[0] / total_fitness_points, decimals=4))
            combine = combine + [[num[i], probability[i]]]

        for k in range(self.require_select):
            print(k)
            for a in range(tourn_num):
                if a == (tourn_num - 1):
                    final = random.choices(combine, probability, k=int(self.POP_SIZE / tourn_num) * (tourn_num - a))
                    break
                ak = random.choices(combine, probability, k=int(self.POP_SIZE / tourn_num) * (tourn_num - a))
                combine = copy.deepcopy(ak)
                print('combine')
                print(combine)
                probability = [combine[i][1] for i in range(len(combine))]
                print('po')
                print(probability)

            orgy_party.append(final[0][0])

        return orgy_party

    def select(self, target):  # nature selection wrt pop's fitness
        brazzer = []
        TS = np.sort(target)
        most_wanted = TS[-(self.require_select):]
        for wanted in most_wanted:
            for pop_num in range(0, self.POP_SIZE):
                if target[pop_num] == wanted:
                    brazzer.append(pop_num)
                    break
        return brazzer

    def crossover(self, mate_1, mate_group, cross_num):  # mating process (genes crossover)
        child_1 = 0
        child_2 = 0
        cross_points_1 = 0
        cross_points_2 = 0
        pop_num = random.randrange(0, self.require_select)
        if np.random.rand() < self.CROSS_RATE:
            while pop_num == cross_num:
                pop_num = random.randrange(0, self.require_select)  # ａｖｏｉｄ　find　ｙｏｕｒｓｅｌｆ
            while (cross_points_1 >= cross_points_2) or abs(cross_points_2 - cross_points_1) < 3:
                cross_points_1 = np.random.randint(0, self.DNA_SIZE)  # choose crossover points
                cross_points_2 = np.random.randint(0, self.DNA_SIZE + 1)  # choose crossover points
            child_1 = copy.deepcopy(mate_1)  # array
            child_2 = copy.deepcopy(mate_group[pop_num])
            for type in range(2):
                child_1[type][cross_points_1:cross_points_2] = mate_group[pop_num][type][cross_points_1:cross_points_2]
                print('child_2s mate process')
                print(child_2[type][cross_points_1:cross_points_2])
                print(mate_1[type][cross_points_1:cross_points_2])
                child_2[type][cross_points_1:cross_points_2] = mate_1[type][cross_points_1:cross_points_2]
            return [list(child_1), child_2]
        print('------still virgin-----')
        return [mate_1, mate_group[pop_num]]


    def mutate(self, child):
        for type in range(2):
            for point in range(self.DNA_SIZE):
                if np.random.rand() < self.MUTATION_RATE:
                    # print("+++++++i just mutate+++++++")
                    if child[type][point] == 0:
                        child[type][point] = 1
                    else:
                        child[type][point] = 0
        return child

    def show_sec_result(self):
        specificity =0
        print("best fitness")
        print(max(self.best_fitness_in_generation))
        self.gamma_pump = 0
        self.Cost = 0
        for i in range(self.N_GENERATIONS):
            if self.best_fitness_in_generation[i] == max(self.best_fitness_in_generation):
                print("This members : Gamma->{0} Cost->{1}".format(self.best_combination_in_generation[i][0]
                                                                  ,self.best_combination_in_generation[i][1]))

                self.Cost = self.best_combination_in_generation[i][1]
                self.gamma_pump = self.best_combination_in_generation[i][0]

                accuracy = self.best_results_in_generation[i][0]
                precision = self.best_results_in_generation[i][1]
                recall = self.best_results_in_generation[i][2]
                specificity = self.best_results_in_generation[i][3]
                F_measure = self.best_results_in_generation[i][4]
                G_mean = self.best_results_in_generation[i][5]
                auc = self.best_results_in_generation[i][6]
                print("Ing results")
                print("accuracy : {0}".format(accuracy))
                print("Precision : {0}".format(precision))
                print("Recall : {0}".format(recall))
                print("specifity : {0}".format(specificity))
                print("F_measure : {0}".format(F_measure))
                print("G_mean : {0}".format(G_mean))
                print("AUC : {0}".format(auc))
                if self.best_combination_in_generation[i][0] == 0:
                    self.gamma_pump = 'auto'
                break

        clf = SVC(C = self.Cost,
                  gamma= self.gamma_pump, kernel='rbf')
        #--------------------------------------------------------------------
        predicted_cv = cross_val_predict(clf, self.distance, self.label, cv=10)
        #--------------------------------------------------------------------
        fir_matrix_train = confusion_matrix(self.label, predicted_cv,labels = [0,1])
        print(fir_matrix_train)
        TP = fir_matrix_train[0][0]
        FN = fir_matrix_train[0][1]
        FP = fir_matrix_train[1][0]
        TN = fir_matrix_train[1][1]
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print("accuracy : {0}".format(accuracy))
        if TN != 0:
            specificity = TN / (TN + FP)
        if TP != 0:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F_measure = 2 * recall * precision / (recall + precision)
            G_mean = math.sqrt(recall * precision)

        print("CV_Precision : {0}".format(precision))
        print("CV_Recall : {0}".format(recall))
        print("CV_specifity : {0}".format(specificity))
        print("CV_F_measure : {0}".format(F_measure))
        print("CV_G_mean : {0}".format(G_mean))
        auc = roc_auc_score(self.label, predicted_cv)
        print("CV_AUC : {0}".format(auc))

        #----------------------------------------------------------------------
        clf.fit(self.distance,self.label)
        predicted = clf.predict(self.test_x)
        #--------------------------------------------------------------------
        fir_matrix_train = confusion_matrix(self.test_y,predicted,labels = [0,1])
        print(fir_matrix_train)
        TP = fir_matrix_train[0][0]
        FN = fir_matrix_train[0][1]
        FP = fir_matrix_train[1][0]
        TN = fir_matrix_train[1][1]
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        print("accuracy : {0}".format(accuracy))
        if TN != 0:
            specificity = TN / (TN + FP)
        if TP != 0:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            F_measure = 2 * recall * precision / (recall + precision)
            G_mean = math.sqrt(recall * precision)

        print("Precision : {0}".format(precision))
        print("Recall : {0}".format(recall))
        print("specifity : {0}".format(specificity))
        print("F_measure : {0}".format(F_measure))
        print("G_mean : {0}".format(G_mean))
        auc = roc_auc_score(self.test_y, predicted)
        print("AUC : {0}".format(auc))
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # --------------------------------------------------------------------------------

    def show_finish_time(self):
        print("----%s minutes----" % (round((time.time() - self.start_time) / 60)))

if __name__ == "__main__":
   start_time = time.time()
    #TODO-------------first stage ---------------------
   fir_ga = Fir_stage(20      # DNA_SIZE 20
            ,100  # POP_SIZE 100
            ,5     #require 5
            ,0.9   # CROSS_RATE
            ,0.5   # MUTATION_RATE
            ,20    # N_GENERATIONS
            ,[0,0.01] #gamma bound
            ,[0,150]   #cost min bound
            ,[0,10]    #cost maj bound
            ,0.2       # split
            ,2    # demand_num 1:best fmeasure
                             # 2:best f_measure contraint precision
                             # 3:best specificity contraint recall
            ,2    # select_method 1: Truncation select(normal)
                                 #2: tournment select
            ,5     # if select is 2(tournment). it is his tournment amount
             )
   # ga.create_Gamma_Gene()
   fir_ga.execute()
   fir_ga.show_fir_result()
   fir_ga.create()
   # TODO--------------second stage--------------
   sec_ga = Sec_Stage(20  # DNA_SIZE 20
                     , 100  # POP_SIZE 100
                     , 5  # require
                     , 0.9  # CROSS_RATE
                     , 0.5  # MUTATION_RATE
                     , 50 # N_GENERATIONS
                     , [0, 0.01]  # gamma
                     , [0, 50]  # cost
                     , 2  # select_method 1: Truncation select(normal)
                     # 2: tournment select
                     , 5  # if select is 2(tournment). it is his tournment amount
                     )
   sec_ga.execute()
   fir_ga.show_fir_result()
   fir_ga.show_finish_time()
   sec_ga.show_sec_result()
   sec_ga.show_finish_time()

   print("Two stage execution time : {0}　minutes".format(round((time.time() - start_time)/60)))
