import torch.cuda
import torch
params = {'img_root': 'data/mri',
          'meta_path': 'data/meta.csv',
          'train_pct': 0.7,
          'batch_size': 32,
          'epochs': 50,
          'num_workers': 0,
          'learning_rate': 1e-3,
          'weight_decay': 1e-4,
          'model_dir': 'models/',
          'random_state': 2,
          'device': "cuda:0" if torch.cuda.is_available() else "cpu",
          }
