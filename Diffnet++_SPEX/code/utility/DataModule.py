from __future__ import division
from collections import defaultdict
import numpy as np

class DataModule():
    def __init__(self, conf, filename):
        self.conf = conf
        self.data_dict = {}
        self.terminal_flag = 1
        self.filename = filename
        self.index = 0

    #######  Initalize Procedures #######
    def prepareModelSupplement(self, model):
        data_dict = {}
        if 'CONSUMED_ITEMS_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedItemsSparseMatrix()
            data_dict['CONSUMED_ITEMS_INDICES_INPUT'] = self.consumed_items_indices_list
            data_dict['CONSUMED_ITEMS_VALUES_INPUT'] = self.consumed_items_values_list
            data_dict['CONSUMED_ITEMS_VALUES_WEIGHT_AVG_INPUT']  = self.consumed_items_values_weight_avg_list
            data_dict['CONSUMED_ITEMS_NUM_INPUT'] = self.consumed_item_num_list
            data_dict['CONSUMED_ITEMS_NUM_DICT_INPUT'] = self.user_item_num_dict
            data_dict['USER_ITEM_SPARSITY_DICT'] = self.user_item_sparsity_dict 
        if 'SOCIAL_NEIGHBORS_SPARSE_MATRIX' in model.supply_set:
            self.readSocialNeighbors()
            self.generateSocialNeighborsSparseMatrix()
            data_dict['SOCIAL_NEIGHBORS_INDICES_INPUT'] = self.social_neighbors_indices_list
            data_dict['SOCIAL_NEIGHBORS_VALUES_INPUT'] = self.social_neighbors_values_list
            data_dict['SOCIAL_NEIGHBORS_VALUES_WEIGHT_AVG_INPUT'] = self.social_neighbors_values_weight_avg_list
            data_dict['SOCIAL_NEIGHBORS_NUM_INPUT'] = self.social_neighbor_num_list
            data_dict['SOCIAL_NEIGHBORS_NUM_DICT_INPUT'] = self.social_neighbors_num_dict
            data_dict['USER_USER_SPARSITY_DICT']= self.user_user_sparsity_dict
        if 'ITEM_CUSTOMER_SPARSE_MATRIX' in model.supply_set:
            self.generateConsumedItemsSparseMatrixForItemUser()      
            data_dict['ITEM_CUSTOMER_INDICES_INPUT'] = self.item_customer_indices_list
            data_dict['ITEM_CUSTOMER_VALUES_INPUT'] = self.item_customer_values_list 
            data_dict['ITEM_CUSTOMER_VALUES_WEIGHT_AVG_INPUT'] = self.item_customer_values_weight_avg_list 
            data_dict['ITEM_CUSTOMER_NUM_INPUT'] = self.item_customer_num_list
            data_dict['ITEM_USER_NUM_DICT_INPUT'] = self.item_user_num_dict
        return data_dict

    def initializeRankingTrain(self):
        self.readTrainData()
        self.arrangePositiveData()
        self.arrangePositiveDataForItemUser() 

    def initializeRankingTest(self):
        self.readTestData()
        self.loadTestNegative()
        self.generateTestList()

    #######  Data Loading #######
    def readTrainData(self):
        f = open(self.filename) 
        total_user_list = set()
        total_item_list = set()
        hash_data = defaultdict(int)
        l = 0
        for _, line in enumerate(f):
            arr = line.split(" ")
            hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_list.add(int(arr[0]))
            total_item_list.add(int(arr[1]))
            l += 1
        self.total_user_list = list(total_user_list)
        self.hash_data = hash_data
        self.conf.num_users = max(total_user_list)+1
        self.conf.num_items = max(total_item_list)+1

    def readTestData(self):
        f = open(self.filename)
        total_user_list = set()
        hash_data = defaultdict(int)
        for _, line in enumerate(f):
            arr = line.split(" ")
            hash_data[(int(arr[0]), int(arr[1]))] = 1
            total_user_list.add(int(arr[0]))
        self.total_user_list = list(total_user_list)
        self.hash_data = hash_data

    def arrangePositiveData(self):
        positive_data = defaultdict(set)
        user_item_num_dict = defaultdict(list)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            positive_data[u].add(i)

        user_list = sorted(list(positive_data.keys()))

        for u in range(self.conf.num_users):
            user_item_num_dict[u] = len(positive_data[u])+1
        self.positive_data = positive_data  # {u:[pos_item,pos_item,...],...}
        self.user_item_num_dict = user_item_num_dict  # {u:pos_item_len,...}
        # self.user_item_num_for_sparsity_dict = self.user_item_num_for_sparsity_dict
        self.total_data = len(total_data)

    def Sparsity_analysis_for_user_item_network(self):
        hash_data_for_user_item = self.hash_data
        sparisty_user_item_dict = {}

    def arrangePositiveDataForItemUser(self):
        positive_data_for_item_user = defaultdict(set)
        item_user_num_dict = defaultdict(list)

        total_data_for_item_user = set()
        hash_data_for_item_user = self.hash_data
        for (u, i) in hash_data_for_item_user:
            total_data_for_item_user.add((i, u))
            positive_data_for_item_user[i].add(u)

        item_list = sorted(list(positive_data_for_item_user.keys()))

        for i in range(self.conf.num_items):
            item_user_num_dict[i] = len(positive_data_for_item_user[i])+1

        self.item_user_num_dict = item_user_num_dict
        self.positive_data_for_item_user = positive_data_for_item_user
        self.total_data_for_item_user = len(total_data_for_item_user)
    
    # This function designes for generating train/val/test negative
    def generateTrainNegative(self):
        num_items = self.conf.num_items
        num_negatives = self.conf.num_negatives
        negative_data = defaultdict(set)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            for _ in range(num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in hash_data:
                    j = np.random.randint(num_items)
                negative_data[u].add(j)
                total_data.add((u, j))
        self.negative_data = negative_data
        self.terminal_flag = 1

    def loadTestNegative(self):
        negative_filename = "%s%s/rec/%s.test.negative" % (self.conf.data_root,self.conf.dataset, self.conf.dataset)
        f = open(negative_filename)
        negative_data = defaultdict(list)

        for _, line in enumerate(f):
            arr = line.split(" ")
            u = arr[0]
            for j in arr[1:]:
                negative_data[int(u)].append(int(j))
        self.negative_data = negative_data

    def generateTestList(self):
        hash_data = self.hash_data
        negative_data = self.negative_data
        test_user_list = []
        test_item_list = []
        for (u, i) in hash_data:
            test_user_list.extend([u]*100)
            test_item_list.extend([int(i)])
            test_item_list.extend(negative_data[u])
        self.test_user_list = np.reshape(test_user_list, [-1, 1])
        self.test_item_list = np.reshape(test_item_list, [-1, 1])


    def parperTrainData(self):
        self.generateTrainNegative()
        positive_data = self.positive_data
        negative_data = self.negative_data
        total_user_list = self.total_user_list
        all_user_list, all_item_list, all_labels_list = [], [], []
        for u in total_user_list:
            all_user_list.extend([u] * len(positive_data[u]))
            all_item_list.extend(list(positive_data[u]))
            all_labels_list.extend([1] * len(positive_data[u]))
            all_user_list.extend([u] * len(negative_data[u]))
            all_item_list.extend(list(negative_data[u]))
            all_labels_list.extend([0] * len(negative_data[u]))
        # shuffled
        shuffled_arg = np.arange(len(all_user_list))
        np.random.shuffle(shuffled_arg)
        self.all_user_list = np.reshape(all_user_list, [-1, 1])[shuffled_arg]
        self.all_item_list = np.reshape(all_item_list, [-1, 1])[shuffled_arg]
        self.all_labels_list = np.reshape(all_labels_list, [-1, 1])[shuffled_arg]

    # ----------------------
    # This function designes for the training process
    def getTrainRankingBatch(self):
        index = self.index
        batch_size = self.conf.batch_size

        if index + batch_size < len(self.all_user_list):
            self.user_list = self.all_user_list[index:index + batch_size]
            self.item_list = self.all_item_list[index:index + batch_size]
            self.labels_list = self.all_labels_list[index:index + batch_size]
            self.index = index + batch_size
        else:
            self.user_list = self.all_user_list[index:len(self.all_user_list)]
            self.item_list = self.all_item_list[index:len(self.all_user_list)]
            self.labels_list = self.all_labels_list[index:len(self.all_user_list)]
            self.index = 0
            self.terminal_flag = 0

    # ----------------------
    # Read social network information
    def readSocialNeighbors(self, friends_flag=1):
        social_neighbors = defaultdict(set)
        social_neighbors_num_dict = defaultdict(list)

        links_file = open("%s%s/rec/%s" % (self.conf.data_root,self.conf.dataset, self.conf.links_filename))
        for _, line in enumerate(links_file):
            tmp = line.split(' ')
            u1, u2 = int(tmp[0]), int(tmp[1])
            social_neighbors[u1].add(u2)
            if friends_flag == 1:
                social_neighbors[u2].add(u1)
        user_list = sorted(list(social_neighbors.keys()))
        for u in range(self.conf.num_users):
            social_neighbors_num_dict[u] = len(social_neighbors[u])+1

        self.social_neighbors_num_dict = social_neighbors_num_dict
        self.social_neighbors = social_neighbors

    def arrangePositiveData(self):
        positive_data = defaultdict(set)
        user_item_num_dict = defaultdict(list)
        total_data = set()
        hash_data = self.hash_data
        for (u, i) in hash_data:
            total_data.add((u, i))
            positive_data[u].add(i)

        user_list = sorted(list(positive_data.keys()))
        for u in range(self.conf.num_users):
            user_item_num_dict[u] = len(positive_data[u])+1

        self.positive_data = positive_data
        self.user_item_num_dict = user_item_num_dict
        self.total_data = len(total_data)

    # ----------------------
    #Generate Social Neighbors Sparse Matrix Indices and Values
    def generateSocialNeighborsSparseMatrix(self):
        social_neighbors = self.social_neighbors
        social_neighbors_num_dict = self.social_neighbors_num_dict  #weight avg

        social_neighbors_indices_list = []
        social_neighbors_values_list = []
        social_neighbors_values_weight_avg_list = []
        social_neighbor_num_list = []
        social_neighbors_dict = defaultdict(list)

        user_user_num_for_sparsity_dict = defaultdict(set)
        user_user_sparsity_dict = {}

        user_user_sparsity_dict['0-4'] = []
        user_user_sparsity_dict['4-8'] = []
        user_user_sparsity_dict['8-16'] = []
        user_user_sparsity_dict['16-32'] = []
        user_user_sparsity_dict['32-64'] = []
        user_user_sparsity_dict['64-'] = []
  
        for u in range(self.conf.num_users):
            user_user_num_for_sparsity_dict[u] = len(social_neighbors[u])

        for u in social_neighbors:
            social_neighbors_dict[u] = sorted(social_neighbors[u])
            
        user_list = sorted(list(social_neighbors.keys()))

        #node att
        for user in range(self.conf.num_users):
            if user in social_neighbors_dict:
                social_neighbor_num_list.append(len(social_neighbors_dict[user]))
            else:
                social_neighbor_num_list.append(1)
        
        for user in user_list:
            for friend in social_neighbors_dict[user]:
                social_neighbors_indices_list.append([user, friend])
                social_neighbors_values_list.append(1.0/len(social_neighbors_dict[user]))
                social_neighbors_values_weight_avg_list.append(1.0/(np.sqrt(social_neighbors_num_dict[user])*np.sqrt(social_neighbors_num_dict[friend])))  #weight avg
   
        for u in range(self.conf.num_users):
            cur_user_neighbors_num = user_user_num_for_sparsity_dict[u]
            if( (cur_user_neighbors_num >=0) & (cur_user_neighbors_num<4) ):
                user_user_sparsity_dict['0-4'].append(u)
            elif( (cur_user_neighbors_num >=4) & (cur_user_neighbors_num<8) ):
                user_user_sparsity_dict['4-8'].append(u)
            elif( (cur_user_neighbors_num >=8) & (cur_user_neighbors_num<16) ):
                user_user_sparsity_dict['8-16'].append(u)
            elif( (cur_user_neighbors_num >=16) & (cur_user_neighbors_num<32) ):
                user_user_sparsity_dict['16-32'].append(u)
            elif( (cur_user_neighbors_num >=32) & (cur_user_neighbors_num<64) ):
                user_user_sparsity_dict['32-64'].append(u)                
            elif( cur_user_neighbors_num >=64):
                user_user_sparsity_dict['64-'].append(u)


        self.user_user_sparsity_dict = user_user_sparsity_dict
        self.social_neighbors_indices_list = np.array(social_neighbors_indices_list).astype(np.int64)
        self.social_neighbors_values_list = np.array(social_neighbors_values_list).astype(np.float32)
        self.social_neighbors_values_weight_avg_list = np.array(social_neighbors_values_weight_avg_list).astype(np.float32)   # weight avg
        self.social_neighbor_num_list = np.array(social_neighbor_num_list).astype(np.int64)
        #self.social_neighbors_values_list = tf.Variable(tf.random_normal([len(self.social_neighbors_indices_list)], stddev=0.01))

    # ----------------------
    #Generate Consumed Items Sparse Matrix Indices and Values
    def generateConsumedItemsSparseMatrix(self):
        positive_data = self.positive_data  
        consumed_items_indices_list = []
        consumed_items_values_list = []
        consumed_items_values_weight_avg_list = []
        consumed_item_num_list = []
        consumed_items_dict = defaultdict(list)
        user_item_num_for_sparsity_dict = defaultdict(set)
        user_item_sparsity_dict = {}

        user_item_sparsity_dict['0-4'] = []
        user_item_sparsity_dict['4-8'] = []
        user_item_sparsity_dict['8-16'] = []
        user_item_sparsity_dict['16-32'] = []
        user_item_sparsity_dict['32-64'] = []
        user_item_sparsity_dict['64-'] = []
        
        consumed_items_num_dict = self.user_item_num_dict   #weight avg
        #social_neighbors_num_dict = self.social_neighbors_num_dict  #weight avg
        item_user_num_dict = self.item_user_num_dict  #weight avg
        print("2===max item:", max(item_user_num_dict.keys()), len(item_user_num_dict))
        print("2===max user:", max(consumed_items_num_dict.keys()), len(consumed_items_num_dict))

        for u in positive_data:
            consumed_items_dict[u] = sorted(positive_data[u])

        user_list = sorted(list(positive_data.keys()))

        for u in range(self.conf.num_users):
            user_item_num_for_sparsity_dict[u] = len(positive_data[u])
        
        for user in range(self.conf.num_users):
            if user in consumed_items_dict:
                consumed_item_num_list.append(len(consumed_items_dict[user]))
            else:
                consumed_item_num_list.append(1)

        for u in user_list:
            for i in consumed_items_dict[u]:
                consumed_items_indices_list.append([u, i])
                consumed_items_values_list.append(1.0/len(consumed_items_dict[u]))
                consumed_items_values_weight_avg_list.append(1.0 / (np.sqrt(consumed_items_num_dict[u]) * np.sqrt(item_user_num_dict[i])) )  #weight avg

        for u in range(self.conf.num_users):
            cur_user_consumed_item_num = user_item_num_for_sparsity_dict[u]
            if( (cur_user_consumed_item_num >=0) & (cur_user_consumed_item_num<4) ):
                user_item_sparsity_dict['0-4'].append(u)
            elif( (cur_user_consumed_item_num >=4) & (cur_user_consumed_item_num<8) ):
                user_item_sparsity_dict['4-8'].append(u)
            elif( (cur_user_consumed_item_num >=8) & (cur_user_consumed_item_num<16) ):
                user_item_sparsity_dict['8-16'].append(u)
            elif( (cur_user_consumed_item_num >=16) & (cur_user_consumed_item_num<32) ):
                user_item_sparsity_dict['16-32'].append(u)
            elif( (cur_user_consumed_item_num >=32) & (cur_user_consumed_item_num<64) ):
                user_item_sparsity_dict['32-64'].append(u)
            elif( cur_user_consumed_item_num >=64):
                user_item_sparsity_dict['64-'].append(u)

        self.user_item_sparsity_dict = user_item_sparsity_dict
        self.consumed_items_indices_list = np.array(consumed_items_indices_list).astype(np.int64)
        self.consumed_items_values_list = np.array(consumed_items_values_list).astype(np.float32)
        self.consumed_items_values_weight_avg_list = np.array(consumed_items_values_weight_avg_list).astype(np.float32)   #weight avg
        self.consumed_item_num_list = np.array(consumed_item_num_list).astype(np.int64)

    def generateConsumedItemsSparseMatrixForItemUser(self):
        positive_data_for_item_user = self.positive_data_for_item_user  
        item_customer_indices_list = []
        item_customer_values_list = []
        item_customer_values_weight_avg_list = []
        item_customer_num_list = []
        item_customer_dict = defaultdict(list)

        consumed_items_num_dict = self.user_item_num_dict   #weight avg
        #social_neighbors_num_dict = self.social_neighbors_num_dict  #weight avg
        item_user_num_dict = self.item_user_num_dict  #weight avg

        for i in positive_data_for_item_user:
            item_customer_dict[i] = sorted(positive_data_for_item_user[i])
        item_list = sorted(list(positive_data_for_item_user.keys()))

        for item in range(self.conf.num_items):
            if item in item_customer_dict:
                item_customer_num_list.append(len(item_customer_dict[item]))
            else:
                item_customer_num_list.append(1)
        
        for i in item_list:
            for u in item_customer_dict[i]:
                item_customer_indices_list.append([i, u])
                item_customer_values_list.append(1.0/len(item_customer_dict[i]))
                item_customer_values_weight_avg_list.append(1.0/(np.sqrt(consumed_items_num_dict[u]) * np.sqrt(item_user_num_dict[i]) ))
      
        self.item_customer_indices_list = np.array(item_customer_indices_list).astype(np.int64)
        self.item_customer_values_list = np.array(item_customer_values_list).astype(np.float32)
        self.item_customer_num_list = np.array(item_customer_num_list).astype(np.int64)
        self.item_customer_values_weight_avg_list = np.array(item_customer_values_weight_avg_list).astype(np.float32)



