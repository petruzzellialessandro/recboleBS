config_dict = {
    #GENERAL RECOMMENDER
    'LightGCN': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'reg_weight': (1e-5, 1e-2)
        },
        "choice": {
            'n_layers': [1,2,3,4],
        }
    },
    'BPR': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3)
        }
    },
    'CDAE': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'corruption_ratio': (0, 0.9),
            'reg_weight_1': (0.0, 0.01),
            'reg_weight_2': (0.0, 0.01)
        },
        "choice": {
            'loss_type': ['BCE','MSE'],
        }
    },
    'ItemKNN': {
        "choice": {
            'k': [10,50,100,200,250,300,40],
            'shrink':[0.0,0.1,0.5,1,2]
        }
    },
    'DMF': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3)
        },
        "choice": {
            'user_hidden_size_list': ['[64, 64]'],
            'item_hidden_size_list': ['[64, 64]'],
        }
    },
    'EASE': {
        "uniform": {
            'reg_weight': (1, 2000)
        }
    },
    'ENMF': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'dropout_prob': (0, 0.9),
            'negative_weight': (0, 0.9),
        }
    },
    'FISM': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'alpha': (0, 0.9)
        },
        "choice": {
            'embedding_size': [16,32,64],
            'regs': ['[1e-7, 1e-7]', '[0, 0]'],
        }
    },
    'GCMC': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'dropout_prob': (0, 0.9),
        },
        "choice": {
            'gcn_output_dim': [128, 256, 512, 1024],
            'accum': ['stack','sum'],
        }
    }, 
    'LINE': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'second_order_loss_weight': (0, 1),
        },
        "choice": {
            'gcn_output_dim': [1,3,5],
        }
    }, 
    'MacridVAE': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
        },
        "choice": {
            'kafc': [1,3,5,10,20]
        }
    },
    'MultiDAE': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
        }
    },
    'MultiVAE': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
        }
    },
    'NAIS': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'alpha': (0, 0.9),
            'beta': (0, 0.9)
        },
        "choice": {
            'weight_size': [16, 32, 64],
            'reg_weights': ['[1e-7, 1e-7, 1e-5]', '[0, 0, 0]']
        }
    }, 
    'NCEPLRec': {
        "uniform": {
            'reg_weight': (1e-4, 1e4),
            'beta': (0, 1)
        },
        "choice": {
            'rank': [100, 200, 450],
        }
    },  
    'NCL': {
        "uniform": {
            'proto_reg': (1e-8, 1e-5),
            'ssl_reg': (1e-8, 1e-5),
            'ssl_temp': (1e-3, 1e-1),
        },
        "choice": {
            'num_clusters': [100,1000],
        }
    },  
    'NeuMF': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'dropout_prob': (0, 0.9),
        },
        "choice": {
            'mlp_hidden_size': ['[64,32,16]'],
        }
    },  
    'NGCF': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'reg_weight': (5e-5, 1e-3),
            'node_dropout': (0, 0.9),
            'node_dropout': (0, 0.9),
        },
        "choice": {
            'hidden_size_list': ['[64,64,64]','[128,128,128]'],
        }
    }, 
    'NNCF': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
        },
        "choice": {
            'neigh_num': [20,50,100],
            'neigh_embedding_size': [32,64],
            'num_conv_kernel': [32,64,128],
            'neigh_info_method': ['random','knn'],
        }
    },  
    'RaCT': {
        "uniform": {
            'dropout_prob': (0, 0.9),
            'anneal_cap': (0, 0.9),
        }
    },  
    'RecVAE': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
        },
    },    
    'SGL': {
        "uniform": {
            'dropout_prob': (0, 0.9),
            'ssl_weight': (0, 0.9),
            'ssl_tau': (0, 0.9),
        },
        "choice": {
            'type': ['ED'],
        }
    },   
    'SimpleX': {
        "uniform": {
            'gamma': (0, 0.9),
            'margin': (0, 0.9),
        },
        "choice": {
            'negative_weight': [1,10,50],
        }
    },  
    'SLIMElastic': {
        "uniform": {
            'alpha': (0, 0.9),
            'l1_ratio': (0, 0.9),
            'margin': (0, 0.9),
        },
        "choice": {
            'positive_only': [True],
            'hide_item': [True],
        }
    },  
    'SpectralCF': {
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'reg_weight': (0, 0.9),
        },
        "choice": {
            'n_layers': [1,2,3,4],
        }
    },

    #KNOWLEDGE AWARE  
    'CFKG':{
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'margin': (5e-1, 5e1)
        },
        "choice": {
            'loss_function': ['inner_product', 'transe'],
        }
    },
    'CKE':{
        "uniform": {
            'learning_rate': (5e-5, 1e-3)
        },
        "choice": {
            'kg_embedding_size': [16, 32, 64, 128],
            'reg_weights': ["[0.1,0.1]", "[0.01,0.01]", "[0.001,0.001]"],
        }
    },
    'KGAT':{
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'mess_dropout': (0, 0.9),
            'reg_weight': (5e-6, 1e-4),
        },
        "choice": {
            'layers': ['[64,32,16]','[64,64,64]','[128,64,32]'],
        }
    },
    'KGCN':{
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'l2_weight': (5e-8, 1e-4),
        },
        "choice": {
            'neighbor_sample_size': [1,2,3,4,6,10],
            'aggregator': ['sum','concat','neighbor'],
        }
    },
    'KGIN':{
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'node_dropout_rate': (0, 0.9),
            'mess_dropout_rate': (0, 0.9),
        },
        "choice": {
            'context_hops': [2,3,5,6],
            'n_factors': [4,8,12,16],
            'ind': ['cosine','distance'],
        }
    },
    'KGNNLS':{
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'l2_weight': (5e-8, 1e-4),
            'ls_weight': (0, 0.9),
        },
        "choice": {
            'n_iter': [1,2,3,5,6],
            'neighbor_sample_size': [4,8,12,16],
            'aggregator': ['sum'],
        }
    },
    'KTUP':{
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
        },
        "choice": {
            'train_kg_step': [0,1,2,3,5,6],
            'train_rec_step': [4,8,12,16],
            'L1_flag': [True, False],
            'use_st_gumbel': [True, False],
        }
    },
    'MCCLK':{
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'node_dropout_rate': (0, 0.9),
            'mess_dropout_rate': (0, 0.9),
        },
        "choice": {
            'build_graph_separately': [True, False],
            'loss_type': ['BPR'],
        }
    },
    'MKR':{
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
            'l2_weight': (5e-8, 1e-4),
        },
        "choice": {
            'kg_embedding_size': [16, 32, 64, 128],
            'low_layers_num': [1,2,4,8,12,16],
            'high_layers_num': [1,2,4,8,12,16],
        }
    },
    'RippleNet':{
        "uniform": {
            'learning_rate': (5e-5, 1e-3),
        },
        "choice": {
            'kg_embedding_size': [16, 32, 64, 128],
            'training_neg_sample_num ': [1,16,32,64,128],
        }
    },
}