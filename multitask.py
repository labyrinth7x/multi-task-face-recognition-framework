from config_multi import get_config
import os
from Learner_multi import face_learner
from pathlib import Path
import argparse

# python train.py -net mobilefacenet -b 200 -w 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('-e', '--epochs', help='training epochs', default=20, type=int)
    parser.add_argument('-net', '--net_mode', help='which network, [ir, ir_se, mobilefacenet]',default='ir_se', type=str)
    parser.add_argument('-depth', '--net_depth', help='how many layers [50,100,152]', default=50, type=int)
    parser.add_argument('-lr','--lr',help='learning rate',default=1e-3, type=float)
    parser.add_argument('-b', '--batch_size', help='batch_size', default=100, type=int)
    parser.add_argument('-w', '--num_workers', help='workers number', default=3, type=int)
    parser.add_argument('-d', '--data_mode', help='use which database, [vgg, ms1m, emore, concat]',default='emore', type=str)
    parser.add_argument('-meta_file', type=str)
    parser.add_argument('-pseudo_folder', type=str)
    parser.add_argument('-remove_single', action='store_true')
    parser.add_argument('-resume', type=str, default=None)
    parser.add_argument('-device', type=int, default=None)
    args = parser.parse_args()

    conf = get_config()
    
    if args.net_mode == 'mobilefacenet':
        conf.use_mobilfacenet = True
    else:
        conf.net_mode = args.net_mode
        conf.net_depth = args.net_depth    
    
    conf.lr = args.lr
    conf.batch_size = args.batch_size
    conf.num_workers = args.num_workers
    conf.data_mode = args.data_mode
    conf.pseudo_folder = args.pseudo_folder
    conf.meta_file = args.meta_file
    conf.work_path = Path(conf.meta_file.replace('labels.txt', str(args.remove_single)))
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'log'
    conf.remove_single = args.remove_single
    conf.resume = args.resume
    conf.device = args.device


    learner = face_learner(conf)

    learner.train(conf, args.epochs)
