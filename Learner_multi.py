from data_pipe import de_preprocess, get_train_dataset, get_pseudo_dataset, get_train_loader, get_val_data
from model import Backbone, Arcface, MobileFaceNet, Am_softmax, l2_norm
from verification import evaluate
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from utils import get_time, gen_plot, hflip_batch, separate_bn_paras, GivenSizeSampler
from torchvision import transforms as trans
import math
import os
import torch.nn as nn

# split1,split1-2,...,split1-9
count = [0, 584013, 1164672, 1740301, 2314488, 2890517, 3465678, 4046365, 4628523, 5206761]


class face_learner(object):
    def __init__(self, conf, inference=False):
        print(conf)

        self.num_splits = int(conf.meta_file.split('_labels.txt')[0][-1])


        if conf.use_mobilfacenet:
            self.model = MobileFaceNet(conf.embedding_size)
            print('MobileFaceNet model generated')
        else:
            self.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
            print('{}_{} model generated'.format(conf.net_mode, conf.net_depth))
        
        if conf.device > 1:
            gpu_ids = list(range(0,min(torch.cuda.device_count(), conf.device)))
            self.model = nn.DataParallel(self.model, device_ids=gpu_ids).cuda()
        else:
            self.model = self.model.cuda()
        
        if not inference:
            self.milestones = conf.milestones

            if conf.remove_single is True:
                conf.meta_file = conf.meta_file.replace('.txt','_clean.txt')
            meta_file = open(conf.meta_file, 'r')
            meta = meta_file.readlines()
            pseudo_all = [int(item.split('\n')[0]) for item in meta]
            pseudo_classnum = set(pseudo_all)
            if -1 in pseudo_classnum:
                pseudo_classnum = len(pseudo_classnum) - 1
            else:
                pseudo_classnum = len(pseudo_classnum)
            print('classnum:{}'.format(pseudo_classnum))

            pseudo_classes = [pseudo_all[count[index]:count[index+1]] for index in range(self.num_splits)]
            meta_file.close()


            train_dataset = [get_train_dataset(conf.emore_folder)] + [get_pseudo_dataset([conf.pseudo_folder, index+1], pseudo_classes[index], conf.remove_single) for index in range(self.num_splits)]
            self.class_num = [num for _,num in train_dataset]
            print('Loading dataset done')
           
            train_longest_size = [len(item[0]) for item in train_dataset]
            temp = int(np.floor(conf.batch_size // (self.num_splits+1)))
            self.batch_size = [conf.batch_size - temp*self.num_splits] + [temp] * self.num_splits
            train_longest_size = max([int(np.floor(td / bs)) for td, bs in zip(train_longest_size, self.batch_size)])
            train_sampler = [GivenSizeSampler(td[0], total_size=train_longest_size * bs, rand_seed=None) for td, bs in zip(train_dataset, self.batch_size)]

            self.train_loader = [DataLoader(train_dataset[k][0], batch_size=self.batch_size[k], 
                shuffle=False, pin_memory=conf.pin_memory, num_workers=conf.num_workers, sampler=train_sampler[k]) for k in range(1+self.num_splits)]
            print('Loading loader done')
            

            self.writer = SummaryWriter(conf.log_path)
            self.step = 0
            self.head = [Arcface(embedding_size=conf.embedding_size, classnum=self.class_num[0]), Arcface(embedding_size=conf.embedding_size, classnum=pseudo_classnum)]
            
            if conf.device > 1:
                self.head = [nn.DataParallel(self.head[0],device_ids=gpu_ids).cuda(), nn.DataParallel(self.head[1],device_ids=gpu_ids).cuda()]
            else:
                self.head = [self.head[0].cuda(), self.head[1].cuda()]

            print('two model heads generated')

            paras_only_bn, paras_wo_bn = separate_bn_paras(self.model.module)
            
            if conf.use_mobilfacenet:
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn[:-1], 'weight_decay': 4e-5},
                                    {'params': [paras_wo_bn[-1]] + [self.head.parameters()], 'weight_decay': 4e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            else:
                params = [a.module.parameters() for a in self.head]
                params = list(params[0]) + list(params[1])
                #from IPython import embed;embed()
                self.optimizer = optim.SGD([
                                    {'params': paras_wo_bn + params, 'weight_decay': 5e-4},
                                    {'params': paras_only_bn}
                                ], lr = conf.lr, momentum = conf.momentum)
            print(self.optimizer)

            if conf.resume is not None:
                self.start_epoch = self.load_state(conf.resume)
            else:
                self.start_epoch = 0

            print('optimizers generated')    
            self.board_loss_every = len(self.train_loader[0])//10
            self.evaluate_every = len(self.train_loader[0])//5
            self.save_every = len(self.train_loader[0])//5
            self.agedb_30, self.cfp_fp, self.lfw, self.agedb_30_issame, self.cfp_fp_issame, self.lfw_issame = get_val_data(conf.eval_path)
        else:
            self.threshold = conf.threshold
    
    def save_state(self, conf, accuracy, e, to_save_folder=False, extra=None, model_only=False):
        if to_save_folder:
            save_path = conf.save_path
            if not os.path.exists(str(save_path)):
                os.makedirs(str(save_path))
        else:
            save_path = conf.model_path
            if not os.path.exists(str(save_path)):
                os.makedirs(str(save_path))
        if model_only:
            torch.save(
                self.model.state_dict(), os.path.join(str(save_path),
                ('model_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))
        else:
            save = {'optimizer':self.optimizer.state_dict(),
                'head':[self.head[0].state_dict(), self.head[1].state_dict()], 'model':self.model.state_dict(),
                'epoch':e}
            torch.save(save,
                os.path.join(str(save_path), ('accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, self.step, extra))))

    
    def load_state(self, save_path, from_save_folder=False, model_only=False):
        if model_only:
            self.model.load_state_dict(torch.load(save_path))
        else:
            state = torch.load(save_path)
            self.model.load_state_dict(state['model'])
            self.head[0].load_state_dict(state['head'][0])
            self.head[1].load_state_dict(state['head'][1])
            self.optimizer.load_state_dict(state['optimizer'])
        return state['epoch'] + 1

        
    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
        
    def evaluate(self, conf, carray, issame, nrof_folds = 5, tta = False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), conf.embedding_size])
        with torch.no_grad():
            while idx + conf.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + conf.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                    embeddings[idx:idx + conf.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + conf.batch_size] = self.model(batch.cuda()).cpu()
                idx += conf.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.cuda()) + self.model(fliped.cuda())
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.cuda()).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    def find_lr(self,
                conf,
                init_value=1e-8,
                final_value=10.,
                beta=0.98,
                bloding_scale=3.,
                num=None):
        if not num:
            num = len(self.loader)
        mult = (final_value / init_value)**(1 / num)
        lr = init_value
        for params in self.optimizer.param_groups:
            params['lr'] = lr
        self.model.train()
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []
        for i, (imgs, labels) in tqdm(enumerate(self.loader), total=num):

            imgs = imgs.cuda()
            labels = labels.cuda()
            batch_num += 1          

            self.optimizer.zero_grad()

            embeddings = self.model(imgs)
            thetas = self.head(embeddings, labels)
            loss = conf.ce_loss(thetas, labels)          
          
            #Compute the smoothed loss
            avg_loss = beta * avg_loss + (1 - beta) * loss.item()
            self.writer.add_scalar('avg_loss', avg_loss, batch_num)
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            self.writer.add_scalar('smoothed_loss', smoothed_loss,batch_num)
            #Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
                print('exited with best_loss at {}'.format(best_loss))
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses
            #Record the best loss
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            #Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            self.writer.add_scalar('log_lr', math.log10(lr), batch_num)
            #Do the SGD step
            #Update the lr for the next step

            loss.backward()
            self.optimizer.step()

            lr *= mult
            for params in self.optimizer.param_groups:
                params['lr'] = lr
            if batch_num > num:
                plt.plot(log_lrs[10:-5], losses[10:-5])
                return log_lrs, losses    

    def train(self, conf, epochs):
        self.model.train()
        running_loss = 0.            
        for e in range(self.start_epoch, epochs):
            print('epoch {} started'.format(e))
            if e == self.milestones[0]:
                self.schedule_lr()
            if e == self.milestones[1]:
                self.schedule_lr()      
            if e == self.milestones[2]:
                self.schedule_lr()                                 
            self.iters = [iter(loader) for loader in self.train_loader]
            for i in tqdm(range(len(self.train_loader[0]))):
                data = [self.iters[i].next() for i in range(len(self.iters))]
                imgs, labels = zip(*[data[k] for k in range(self.num_splits+1)])
                labeled_num = len(imgs[0])

                imgs = torch.cat(imgs, dim=0)
                labels = torch.cat(labels, dim=0)

                imgs = imgs.cuda()
                labels = labels.cuda()
                self.optimizer.zero_grad()
                embeddings = self.model(imgs)
                
                thetas = self.head[0](embeddings[:labeled_num], labels[:labeled_num])
                losses1 = conf.ce_loss(thetas, labels[:labeled_num])
                thetas = self.head[1](embeddings[labeled_num:], labels[labeled_num:])
                losses2 = conf.ce_loss(thetas, labels[labeled_num:])

                num_ratio = labeled_num / len(embeddings)
                loss = num_ratio * losses1 + (1-num_ratio) * losses2

                loss.backward()
                running_loss += loss.item()
                self.optimizer.step()
                
                if self.step % self.board_loss_every == 0 and self.step != 0:
                    loss_board = running_loss / self.board_loss_every
                    self.writer.add_scalar('train_loss', loss_board, self.step)
                    print('step:{}, train_loss:{}'.format(self.step, loss_board))
                    running_loss = 0.
                
                if self.step % self.evaluate_every == 0 and self.step != 0:
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.agedb_30, self.agedb_30_issame)
                    accuracy1 = accuracy
                    self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.lfw, self.lfw_issame)
                    accuracy2 = accuracy
                    self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor)
                    accuracy, best_threshold, roc_curve_tensor = self.evaluate(conf, self.cfp_fp, self.cfp_fp_issame)
                    accuracy3 = accuracy
                    self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor)
                    print('step:{}, agedb:{},lfw:{},cfp_fp:{}'.format(self.step, accuracy1, accuracy2, accuracy3))
                    self.model.train()
                if self.step % self.save_every == 0 and self.step != 0:
                    self.save_state(conf, accuracy, e)
                    
                self.step += 1
                
        self.save_state(conf, accuracy, e, to_save_folder=True, extra='final')

    def schedule_lr(self):
        for params in self.optimizer.param_groups:                 
            params['lr'] /= 10
        print(self.optimizer)
    
    def infer(self, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        for img in faces:
            if tta:
                mirror = trans.functional.hflip(img)
                emb = self.model(conf.test_transform(img).cuda().unsqueeze(0))
                emb_mirror = self.model(conf.test_transform(mirror).cuda().unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:                        
                embs.append(self.model(conf.test_transform(img).cuda().unsqueeze(0)))
        source_embs = torch.cat(embs)
        
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1,0).unsqueeze(0)
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1 # if no match, set idx to -1
        return min_idx, minimum               
