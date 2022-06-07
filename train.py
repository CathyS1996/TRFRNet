import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import datasets
from utils_TRFR.metrics import evaluate
from opt import opt
from utils_TRFR.comm import generate_model
from utils_TRFR.loss import BceDiceLoss
from utils_TRFR.metrics import Metrics
import os
import numpy as np

def prob_2_entropy(prob):
    _, c, _, _ = prob.size()
    # ent = -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)
    ent = -torch.mul(prob, torch.log2(prob + 1e-30)) 
    return ent

def valid(model, valid_dataloader, total_batch):

    model.eval()

    # Metrics_logger initialization
    metrics = Metrics(['recall', 'specificity', 'precision', 'F1', 'F2',
                       'ACC_overall', 'IoU_poly', 'IoU_bg', 'IoU_mean'])

    with torch.no_grad():
        bar = tqdm(enumerate(valid_dataloader), total=total_batch)
        for i, data in bar:
            img, gt = data['image'], data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            output = model(img)
            _recall, _specificity, _precision, _F1, _F2, \
            _ACC_overall, _IoU_poly, _IoU_bg, _IoU_mean = evaluate(output, gt)

            metrics.update(recall= _recall, specificity= _specificity, precision= _precision, 
                            F1= _F1, F2= _F2, ACC_overall= _ACC_overall, IoU_poly= _IoU_poly, 
                            IoU_bg= _IoU_bg, IoU_mean= _IoU_mean
                        )

    metrics_result = metrics.mean(total_batch)

    return metrics_result


def train():

    model = generate_model(opt,'TRFRNet')

    model_D1 = generate_model(opt, 'FCDiscriminator')

    # load data
    source_data = getattr(datasets, opt.sdataset)(opt.root, opt.strain_data_dir, mode='train')
    source_dataloader = DataLoader(source_data, opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    target_data = getattr(datasets, opt.tdataset)(opt.root, opt.ttrain_data_dir, mode='train')
    target_dataloader = DataLoader(target_data, opt.batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True)
    valid_data = getattr(datasets, opt.tdataset)(opt.root, opt.tvalid_data_dir, mode='valid')
    valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    val_total_batch = int(len(valid_data) / 1)
   

    # load optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.mt, weight_decay=opt.weight_decay)

    lr_lambda = lambda epoch: 1.0 - pow((epoch / opt.nEpoch), opt.power)
    scheduler = LambdaLR(optimizer, lr_lambda)

    optimizer_TRFR = torch.optim.SGD(model.TRFR.parameters(), lr=opt.lrD, momentum=opt.mt, weight_decay=opt.weight_decay)
    scheduler_TRFR = LambdaLR(optimizer_TRFR, lr_lambda)

    optimizer_D1 = torch.optim.SGD(model_D1.parameters(), lr=opt.lrD, momentum=opt.mt, weight_decay=opt.weight_decay)
    scheduler_D1 = LambdaLR(optimizer_D1, lr_lambda)

    # train
    print('Start training')
    print('---------------------------------\n')
    best_score = 0.0

    for epoch in range(opt.nEpoch):
        print('------ Epoch', epoch + 1)
        model.train()
        model_D1.train()
        total_batch = int(len(source_data) / opt.batch_size)
        bar = enumerate(source_dataloader)

        # labels for adversial training
        source_label = 0
        target_label = 1

        
        for i, data in bar:
            for param in model_D1.parameters():
                param.requires_grad = False
            
            # train with target
            img = data['image']
            gt = data['label']

            if opt.use_gpu:
                img = img.cuda()
                gt = gt.cuda()

            optimizer.zero_grad()
            optimizer_D1.zero_grad()
            optimizer_TRFR.zero_grad()
            output = model(img)

            loss_seg = BceDiceLoss()(torch.sigmoid(output), gt)
            loss_seg.backward()

            try:
                _, tdata = next(targetloader_iter)
            except:
                targetloader_iter = iter(enumerate(target_dataloader))
                _, tdata = next(targetloader_iter)
            # next()
            # train with target
            timg = tdata['image']
            if opt.use_gpu:
                timg = timg.cuda()
            toutput = model(timg)
            D_out1 = model_D1(torch.sigmoid(toutput))

            mask = ((torch.sigmoid(toutput)-0.5)>0).detach()
            
            if opt.use_gpu:
                D_out1_l = (torch.FloatTensor(D_out1.data.size()).fill_(source_label)).cuda()

            loss_adv_target1 = BceDiceLoss()(torch.mul(D_out1,mask), torch.mul(D_out1_l,mask)) + 0.5*BceDiceLoss()(torch.mul(D_out1,~mask), torch.mul(D_out1_l,~mask))
            loss_adv = 0.001*loss_adv_target1
            loss_adv.backward()
            optimizer.step()

            # train TRFR
            for param in model.encoder.parameters():
                param.requires_grad = False
            for param in model.decoder.parameters():
                param.requires_grad = False
            output_plus, output_DI = model.entropy(img)
            ent_DI = prob_2_entropy(torch.sigmoid(output_DI))
            ent_plus = prob_2_entropy(torch.sigmoid(output_plus))
            ent_loss = torch.nn.SoftMarginLoss()(ent_plus-ent_DI,torch.ones_like(ent_DI))
            toutput_plus, toutput_DI = model.entropy(timg)
            tent_DI = prob_2_entropy(torch.sigmoid(toutput_DI))
            tent_plus = prob_2_entropy(torch.sigmoid(toutput_plus))
            tent_loss = torch.nn.SoftMarginLoss()(tent_plus-tent_DI,torch.ones_like(tent_DI))
            loss_TRFR = 0.01*(ent_loss+tent_loss)
            loss_TRFR.backward()
            optimizer_TRFR.step()


            # train D
            for param in model_D1.parameters():
                param.requires_grad = True
            
            pred1 = output.detach()
            mask_ = ((torch.sigmoid(pred1)-0.5)>0).detach()
            D_out_pred1 = model_D1(torch.sigmoid(pred1))
            if opt.use_gpu:
                D_out_pred1_l = (torch.FloatTensor(D_out_pred1.data.size()).fill_(source_label)).cuda()
            loss_D1_pred1 =  BceDiceLoss()(torch.mul(D_out_pred1,mask_),torch.mul(D_out_pred1_l,mask_)) + 0.5*BceDiceLoss()(torch.mul(D_out_pred1,~mask_),torch.mul(D_out_pred1_l,~mask_))
            loss_D1_pred1.backward()

            tpred1 = toutput.detach()
            tmask_ = ((torch.sigmoid(tpred1)-0.5)>0).detach()
            D_out_tpred1 = model_D1(torch.sigmoid(tpred1))
            if opt.use_gpu:
                D_out_tpred1_l = (torch.FloatTensor(D_out_tpred1.data.size()).fill_(target_label)).cuda()
            loss_D1_tpred1 =  BceDiceLoss()(torch.mul(D_out_tpred1,tmask_),torch.mul(D_out_tpred1_l,tmask_)) + 0.5* BceDiceLoss()(torch.mul(D_out_tpred1,~tmask_),torch.mul(D_out_tpred1_l,~tmask_))
            loss_D1_tpred1.backward()
            optimizer_D1.step()

            print('iter =  %.3d/%.3d, loss_seg1 = %.3f, loss_adv1 = %.3f, loss_useful = %.3f, loss_D1_source = %.3f, loss_D1_target = %.3f'
                %(i, int(len(source_data) / opt.batch_size), loss_seg, loss_adv_target1, loss_TRFR, loss_D1_pred1, loss_D1_tpred1))

            

        scheduler.step()
        scheduler_TRFR.step()
        scheduler_D1.step()


        metrics_result = valid(model, valid_dataloader, val_total_batch)
        score = metrics_result['recall']+ metrics_result['specificity']+ metrics_result['precision']+metrics_result['F1']+ metrics_result['F2']+ metrics_result['ACC_overall']+metrics_result['IoU_poly']+ metrics_result['IoU_bg']+ metrics_result['IoU_mean']

        print("Valid Result:")
        print('recall: %.4f, specificity: %.4f, precision: %.4f, F1: %.4f,'
              ' F2: %.4f, ACC_overall: %.4f, IoU_poly: %.4f, IoU_bg: %.4f, IoU_mean: %.4f'
              % (metrics_result['recall'], metrics_result['specificity'], metrics_result['precision'],
                 metrics_result['F1'], metrics_result['F2'], metrics_result['ACC_overall'],
                 metrics_result['IoU_poly'], metrics_result['IoU_bg'], metrics_result['IoU_mean']))

        if not (os.path.exists('./checkpoints'+str(opt.A2B)+'/exp'+ str(opt.expID))):
            os.makedirs('./checkpoints'+str(opt.A2B)+'/exp'+ str(opt.expID))
        if ((epoch + 1) % opt.ckpt_period == 0) and (epoch>50): 
            torch.save(model.state_dict(), './checkpoints'+str(opt.A2B)+'/exp' + str(opt.expID)+"/ck_{}.pth".format(epoch + 1))
        if score >best_score:
            torch.save(model.state_dict(), './checkpoints'+str(opt.A2B)+'/exp' + str(opt.expID)+"/ck_best.pth")
            best_score = score
        


if __name__ == '__main__':

    if opt.mode == 'train':
        print('---PolpySeg Train---')
        train()

    print('Done')