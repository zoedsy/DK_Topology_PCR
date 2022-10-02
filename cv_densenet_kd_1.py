# dataset/best_model/summary writer root needs to be edited
# cv-densenet+kd
import logging
import os
import sys
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn import metrics
from torch import nn
import monai
from monai.data import ImageDataset
from monai.transforms import AddChannel, Compose, RandRotate90, Resize, ScaleIntensity, EnsureType
from Logger import make_print_to_file
from metric_helpper import classi_report
from metric_helpper import comp_auc
from AutomaticWeightedLoss import AutomaticWeightedLoss
import random
import time
from datetime import datetime, timedelta
import torch.nn.functional as F
import argparse


# 添加有topo特征相关的分支
class Addtopo(nn.Module):
    def __init__(self, model):
        super(Addtopo, self).__init__()
        #         head part of denselayer
        self.resnet_layer1 = nn.Sequential(*list(model.children())[0])
        #         tail part of denselayer without the end classifier
        self.resnet_layer2 = nn.Sequential(*list(model.children())[1][:-1])
        #     densenet 所有层
        self.densenet_ori = nn.Sequential(*list(model.children()))

        # dense出来的分支，做一个普通的分类
        self.liner_layer_aux = nn.Linear(1024, 762)
        # seg-betticurve，做一个普通的分类
        self.linear_layer_cls = nn.Linear(762, 2)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, inputs, segs):
        # 这是3D图像特征部分
        outputs = self.densenet_ori(inputs)
        #       densenet_head
        x0 = self.resnet_layer1(inputs)

        x0 = self.resnet_layer2(x0)
        x_aux = self.relu(self.liner_layer_aux(x0))
        x_dense_cls = self.relu(self.linear_layer_cls(x_aux))
        segs = segs.view(segs.size(0), -1)
        #         print("segs.size()",segs.size())
        x_betti_cls = self.relu(self.linear_layer_cls(segs))

        print("x_dense_cls", x_dense_cls)
        print("x_dense_cls.argmax(dim=1)", x_dense_cls.argmax(dim=1))
        return outputs, x_dense_cls, x_betti_cls


# 固定seed
def set_global_random_seed(seed):
    os.environ['PYTHONASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


#     torch.backends.cudnn.benchmark = True

#knowledge distillation
def distillation(y,teacher_scores,labels,T,alpha):
    #学生logits的softmax
    p = F.log_softmax(y/T,dim=1)
    #老师logits的softmax-t
    q = F.softmax(teacher_scores/T,dim=1)
    #学生老师之间的kl散度之类的
    l_kl = F.kl_div(p,q,size_average=False)*(T**2)/y.shape[0]
    #学生logits和标签的ce
    l_ce = F.cross_entropy(y,labels)
    return l_kl*alpha+l_ce*(1-alpha)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", help="k_fold", default="5", type=int)
    parser.add_argument("--bs", help="bs", default="64", type=int)
    parser.add_argument("--seed", help="seed", default="1", type=int)
    parser.add_argument("--w_loss1", help="w_loss1", default="1", type=float)
    parser.add_argument("--w_loss2", help="w_loss2", default="0.2", type=float)
    parser.add_argument("--weight_d", help="weight_d", default="0", type=float)
    parser.add_argument("--learning_r", help="learning_r", default="1e-4", type=float)
    parser.add_argument("--epochs", help="epochs", default="200", type=int)
    parser.add_argument("--T", help="Temp", default="1", type=float)
    parser.add_argument("--alpha", help="alpha", default="1", type=float)
    
    args = parser.parse_args()
    k = args.k
    bs = args.bs
    seed = args.seed
    w_loss1 = args.w_loss1
    w_loss2 = args.w_loss2
    epochs = args.epochs
    learning_r = args.learning_r
    weight_d = args.weight_d
    T=args.T
    alpha=args.alpha

    set_global_random_seed(seed)
    # 输出流
    monai.config.print_config()

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # read the data
    topo_root = '/dushiyi/CV/Test/TTEST/TTEST/Code3D/MONAI/data/norm_betticurve_npy'
    root = "/dushiyi/CV/tensorflow_projects/medical_image_analysis/Slicer_NII"
    os.chdir(root)

    # normal nonpcrs
    # abnormal pcr
    normal_scan_paths = []
    abnormal_scan_paths = []
    normal_topo_paths = []
    abnormal_topo_paths = []

    for file in glob.glob("*.nii"):
        if str(file.split("_")[1]) == "0.nii":
            normal_scan_paths.append(os.path.join(root, file))
        else:
            abnormal_scan_paths.append(os.path.join(root, file))
    os.chdir(topo_root)
    for file in glob.glob("*.npy"):
        if str(file.split("_")[1]) == "0.npy":
            normal_topo_paths.append(os.path.join(topo_root, file))
        else:
            abnormal_topo_paths.append(os.path.join(topo_root, file))

    print("nonpcr num:" + str(len(normal_scan_paths)))
    print("pcr num:" + str(len(abnormal_scan_paths)))

    print("nonpcrtopo num:" + str(len(normal_topo_paths)))
    print("pcrtopo num:" + str(len(abnormal_topo_paths)))

    scan_paths = normal_scan_paths + abnormal_scan_paths
    y0 = [0] * len(normal_scan_paths)
    y1 = [1] * len(abnormal_scan_paths)
    y = np.concatenate(([0] * len(normal_scan_paths), [1] * len(normal_scan_paths)), axis=0)
    y = np.array(y, dtype=np.int64)
    topo_paths = normal_topo_paths + abnormal_topo_paths

    print("before shuffle")


    state = np.random.get_state()
    np.random.shuffle(scan_paths)
    np.random.set_state(state)
    np.random.shuffle(y)
    np.random.set_state(state)
    np.random.shuffle(topo_paths)
    train_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 60)), RandRotate90(), EnsureType()])
    val_transforms = Compose([ScaleIntensity(), AddChannel(), Resize((96, 96, 60)), EnsureType()])

    sample_num = len(scan_paths)
    fold_size = sample_num // k


    for i in range(k):
        #         i=1
        val_acc_list = []
        val_auc_list = []
        val_sen_list = []
        val_spe_list = []
        val_f1score_list = []

        if i != k - 1:
            val_start = i * fold_size
            val_end = (i + 1) * fold_size
            scan_paths_val, topo_paths_val, y_val = scan_paths[val_start:val_end], topo_paths[val_start:val_end], y[
                                                                                                                  val_start:val_end]
            #             scan_paths_train,topo_paths_train,y_train=scan_paths[:val_start]+scan_paths[val_end:],topo_paths[:val_start]+topo_paths[val_end:],y[:val_start]+y[val_end:]

            s = scan_paths[0:val_start]
            s.extend(scan_paths[val_end:])
            t = topo_paths[0:val_start]
            t.extend(topo_paths[val_end:])
            yy = np.concatenate((y[0:val_start], y[val_end:]), axis=0)
            print("y--------------------------", y)

            scan_paths_train, topo_paths_train, y_train = s, t, yy
        else:  # 如果是最后一折交叉
            scan_paths_val, topo_paths_val, y_val = scan_paths[val_start:], topo_paths[val_start:], y[val_start:]
            scan_paths_train, topo_paths_train, y_train = scan_paths[0:val_start], topo_paths[0:val_start], y[
                                                                                                            0:val_start]

        train_ds = ImageDataset(image_files=scan_paths_train, seg_files=topo_paths_train,
                                labels=y_train,
                                transform=train_transforms)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=2,
                                  pin_memory=torch.cuda.is_available())
        # create a validation data loader
        val_ds = ImageDataset(image_files=scan_paths_val, seg_files=topo_paths_val,
                              labels=y_val,
                              transform=val_transforms)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2,
                                pin_memory=torch.cuda.is_available())

        print("bs:", bs)

        # Create DenseNet121, CrossEntropyLoss and Adam optimizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tb_order = 0

        tb_order += 1
        tb_path = time.strftime("%F") + "/train_val:5fold" + " /RATIO:" + str(
            w_loss1) + ":" + str(w_loss2) + "/LR:" + str(learning_r) + "/BS:" + str(bs) + "/WD:" + str(
            weight_d)
        a = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        best_model_root = '/dushiyi/CV/Test/TTEST/TTEST/Code3D/MONAI/best_model_kd/' + a + '.pth'

        # densenet121
        model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
        model = Addtopo(model).to(device)
        model = torch.nn.DataParallel(model).to(device)

        # 主任务的loss_fn
        loss_function = torch.nn.CrossEntropyLoss()
        # auxiliary任务的loss_fn_aux
        loss_function_aux = torch.nn.MSELoss()
        # optim
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_r, weight_decay=weight_d)
        print(model)

        # start a typical PyTorch training
        val_interval = 2
        best_metric = -1
        best_metric_epoch = -1


        epoch_loss_values = list()
        metric_values = list()
       
        writer = SummaryWriter("/dushiyi/CV/Test/TTEST/TTEST/Code3D/MONAI/runs_kd")

        for epoch in range(epochs):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{epochs}")
            model.train()

            epoch_loss = 0
            val_epoch_loss = 0

            step = 0
            val_step = 0

            for batch_data in train_loader:
                step += 1
                inputs, segs, labels = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
                outputs, x_dense_cls, x_betti_cls = model(inputs, segs)
                loss1 = loss_function(outputs, labels)
                print("train_outputs", outputs)
                print("train_labels", labels)
                #                                 loss2 = loss_function(x_betti_cls,x_dense_cls.argmax(dim=1))
                loss2 = distillation(x_betti_cls, x_dense_cls, labels, T, alpha)
                loss3 = loss_function(x_betti_cls, labels)
                print("x_betti_cls", x_betti_cls)
                print("x_dense_cls", x_dense_cls)
                #             loss =torch.tensor(awl(loss1,loss2),requires_grad=True).float().cuda()
                loss = w_loss1 * loss1 + w_loss2 * loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_len = len(train_ds) // train_loader.batch_size + 1

                print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
                writer.add_scalar(tb_path + "/train_loss", loss.item(), epoch_len * epoch + step)

            epoch_loss /= step
            epoch_loss_values.append(epoch_loss)
            print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    num_correct = 0.0
                    metric_count = 0
                    val_outputs_list = list()
                    val_labels_list = list()
                    for val_data in val_loader:
                        val_step += 1
                        val_images, val_segs, val_labels = val_data[0].to(device), val_data[1].to(device), val_data[2].to(device)
                        val_labels_list.extend(val_labels.cpu().numpy().tolist())
                        #将val_images,val_segs放入到模型中
                        val_outputs,val_x_dense_cls,val_x_betti_cls = model(val_images,val_segs)
                        val_outputs_list.extend(val_outputs.argmax(dim=1).cpu().numpy().tolist())
                        val_loss1 = loss_function(val_outputs, val_labels)
                        print("val_x_betti_cls.size",val_x_betti_cls.size())
                        print("val_x_dense_cls.size",val_x_dense_cls.size())

                        val_loss2 = distillation(val_x_betti_cls,val_x_dense_cls,val_labels,T,alpha)
                        #                     val_loss=awl(val_loss1,val_loss2).float().cuda()
                        val_loss3=loss_function(val_x_betti_cls,val_labels)
                        val_loss = w_loss1 * val_loss1 + w_loss2 * val_loss2
                        val_epoch_loss += val_loss.item()
                        val_epoch_len = len(val_ds) // val_loader.batch_size+1
                        print(f"{val_step}/{val_epoch_len},val_loss:{val_loss.item():.4f}")

                        value = torch.eq(val_outputs.argmax(dim=1), val_labels)
                        metric_count += len(value)
                        num_correct += value.sum().item()


                        # writer.add_scalar(tb_path + "/val_loss", val_loss.item(),
                        #                   val_epoch_len * epoch + step)

                    metric = num_correct / metric_count
                    metric_values.append(metric)
                    bclass_metric = classi_report(val_labels_list, val_outputs_list)
                    val_auc = comp_auc(val_labels_list, val_outputs_list)
                    val_spe = comp_specificity(val_labels_list, val_outputs_list)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1

                        torch.save(model.state_dict(), best_model_root)
                        print("saved new best metric model")
                    print(
                        "current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                            epoch + 1, metric, best_metric, best_metric_epoch
                        )
                    )

                    print(
                        "auc: {:.4f} precision: {:.4f} recall: {:.4f} f1score: {:.4f}".format(
                            val_auc, bclass_metric["weighted avg"]["precision"],
                            bclass_metric["weighted avg"]["recall"],
                            bclass_metric["weighted avg"]["f1-score"]
                        )
                    )
                    val_acc_list.append(metric)
                    val_auc_list.append(val_auc)
                    val_sen_list.append(bclass_metric["weighted avg"]["recall"])
                    val_spe_list.append(val_spe)
                    val_f1score_list.append(bclass_metric["weighted avg"]["f1-score"])

                    writer.add_scalar(tb_path + "/val_accuracy", metric, epoch + 1)
                    writer.add_scalar(tb_path + "/val_auc", val_auc, epoch + 1)
                    # writer.add_scalar(tb_path + "/precision", bclass_metric["weighted avg"]["precision"], epoch + 1)

                    writer.add_scalar(tb_path + "/sensitivity", bclass_metric["weighted avg"]["recall"], epoch + 1)
                    writer.add_scalar(tb_path+"/specificity",val_spe,epoch+1)
                    writer.add_scalar(tb_path + "/f1-score", bclass_metric["weighted avg"]["f1-score"], epoch + 1)

        print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    val_metrics_list={
        "val_acc":np.array(val_acc_list),
        "val_auc":np.array(val_auc_list),
        "val_sen":np.array(val_sen_list),
        "val_spe":np.array(val_spe_list),
        "val_f1score":val_f1score_list
    }
    writer.close()


if __name__ == "__main__":
    make_print_to_file()
    main()

