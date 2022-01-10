params = {'img_root': 'data/mri',
          'meta_path': 'data/meta.csv',
          'log_path': 'run.log',
          'train_pct': 0.7,
          'batch_size': 32,
          'epochs': 50,
          'num_workers': 0,
          'learning_rate': 1e-3,
          'weight_decay': 1e-4,
          'model_dir': 'models/',  # saved models dir
          'random_state': 2,
          'device': 'auto',  # 'auto'or 'tpu'
          'master_addr': '127.0.0.1',  # DDP main node
          'master_port': '5555'
          }
