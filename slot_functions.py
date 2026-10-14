# coding = utf-8
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import numpy as np
from config import args
import torch 
import models.GCDLNet
import models.satten_RegNet
import models.denseRegNet
import models.finalnet
import models.denseNet_reg
import models.resnet_reg
import models.unet_reg
import models.UNet_3plus
import models.LadderNet
import models.ConvNext
import isolate
from matplotlib.colors import Normalize, ListedColormap
from matplotlib import colormaps


def _make_lut_colormap(white_width=0.06, n=256):
    """Display colormap: a narrow white band at the bottom fading into the original
    'nipy_spectral'; all colors above the white band keep their original positions."""
    base = colormaps['nipy_spectral']
    k = max(2, int(round(white_width * n)))
    lower = np.linspace(np.array([1.0, 1.0, 1.0, 1.0]), base(white_width), k)
    upper = base(np.linspace(white_width, 1.0, n - k + 1))
    cmap = ListedColormap(np.vstack([lower, upper[1:]]), name='nipy_spectral_white')
    cmap.set_bad('white')
    return cmap


LUT_CMAP = _make_lut_colormap()


class slot_functions():
    def __init__(self, ui):
        super(slot_functions, self).__init__()
        self.ui = ui

    def import_file_slot(self): 
        tmp_file_name, _ = QFileDialog.getOpenFileName(self.ui, "select a file", "C:/", "numpy Files (*.npy);")
        self.ui.working_dir = os.path.dirname(tmp_file_name)
        self.ui.filemodel.setRootPath(self.ui.working_dir)
        self.ui.working_tree.setRootIndex(self.ui.filemodel.index(self.ui.working_dir))
        messages = '--import file:' + tmp_file_name
        self.info_slot(messages)
        if tmp_file_name.endswith('.npy'):
            lut = np.load(tmp_file_name, allow_pickle=True)
            if lut.shape[0] != 256 or lut.shape[1] != 256:
                # QMessageBox
                return
            self.ui.lut = lut
        if os.path.exists(tmp_file_name.replace('imgs', 'peaks')):
            peak = np.load(tmp_file_name.replace('imgs', 'peaks'), allow_pickle=True)
            if  peak.shape[0] != 256 or peak.shape[1] != 2:
                self.ui.peak = None
                self.show(lut=lut)
                self.ui.file_name = tmp_file_name
                self.ui.vis = 1
                # QMessageBox
                return
            self.ui.peak = peak
            self.show(lut=lut, peaks=peak)
            self.ui.file_name = tmp_file_name
            self.ui.vis = 2
            return
                                   
    def import_folder_slot(self):
        self.ui.working_dir = QFileDialog.getExistingDirectory(None, 'select a folder', os.getcwd())
        self.ui.filemodel.setRootPath(self.ui.working_dir)
        self.ui.working_tree.setRootIndex(self.ui.filemodel.index(self.ui.working_dir))
        messages = '--import folder:' + self.ui.working_dir
        self.info_slot(messages)
    
    def save_slot(self):
        print("save slot function")
        
    def quit_slot(self):
        app = QApplication.instance()
        app.quit()

    def predict_slot(self):
        if self.ui.lut is None:
            self.info_slot('--Lacking input lut!')
            return
        if self.ui.net_name == 'ResNet18':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy() # + data_set.mean_cordinate
            predict_peak.resize([256, 2])
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'GCDLNet':
            with torch.no_grad():
                input = self.ui.lut.reshape(-1, 1, 256, 256) 
                input = torch.from_numpy(input)
                input = input.to(torch.float32).to(args.device)
                predict_peak = self.ui.net(torch.clone(input))
                predict_peak = predict_peak.cpu().detach().numpy() # + data_set.mean_cordinate
                predict_peak.resize([256, 2])
                predict_peak = np.round(predict_peak).astype(int)
                self.ui.peak = predict_peak
                self.show(lut=self.ui.lut, peaks=predict_peak)
                self.ui.vis = 2
        elif self.ui.net_name == 'ResNet34':
            self.ui.max_iter = 0
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak.resize([256, 2])
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'DenseNet121':
            self.ui.max_iter += 1
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input).to(torch.int)
            # tmp1 = isolate.AverageBasedFilterModule(1)
            # input= tmp1(input[:, 0, :, :].reshape(1, 1, 256,256))
            input = input[:, 0, :, :].unsqueeze(0)
            tmp = isolate.GravityGuidedRegressionModule() #max_iters = self.ui.max_iter)
            predict_peak1 = tmp(input, predict_peak.reshape(-1, 256, 2))
            tmp1 = isolate.PointAdjustmentModule(4)
            predict_peak = tmp1(input, predict_peak.reshape(-1, 256, 2), predict_peak1.reshape(-1, 256, 2))
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            # predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'DenseNet121_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak[0,0,:] + torch.tensor(self.ui.mean_cor).to(predict_peak.device)
            lut = isolate.generate_heatmap(input.shape, predict_peak.reshape(-1, 1, 256, 2))
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=lut.reshape(256, 256).cpu().detach().numpy(), peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'UNet':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            # predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'UNet_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak = predict_peak = predict_peak + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'UNet3+':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            # predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'UNet3+_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'Laddernet':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            # predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'Laddernet_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'Convnext':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            # predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'Convnext_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()# + data_set.mean_cordinate
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'finalNet_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(3, axis=1) # [1,3,256,256]
            input = input.repeat(2, axis=0) # [2,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'stRegNet_mean':
            self.ui.net.train()
            tmp_lut = (self.ui.lut - self.ui.lut.min())/(self.ui.lut.max()-self.ui.lut.min() + 1e-7)
            input = np.repeat(tmp_lut[None,None,...], 1, axis=0).repeat(1, axis=1) # [1,1,256,256]
            input = torch.from_numpy(input)
            input = input.to(torch.float32).to(args.device)
            predict_peak = self.ui.net(input)
            predict_peak = predict_peak.cpu().detach().numpy()
            predict_peak = predict_peak = predict_peak[0,0,:] + self.ui.mean_cor
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        elif self.ui.net_name == 'Groundtruth':
            predict_peak = self.ui.gt
            predict_peak.resize([256, 2]) #[256,2]
            predict_peak = np.round(predict_peak).astype(int)
            self.ui.peak = predict_peak
            self.show(lut=self.ui.lut, peaks=predict_peak)
            self.ui.vis = 2
        if self.ui.peak is not None:
            # Ensure the folder exists before saving
            current_dir = os.path.dirname(self.ui.file_name)
            file_name = os.path.basename(self.ui.file_name)

            # Get the parent directory
            parent_dir = os.path.dirname(current_dir)
            save_dir = parent_dir + '/pred_peaks'
            os.makedirs(save_dir, exist_ok=True)

            # Save the file
            np.save(save_dir + f'/{file_name}', self.ui.peak)
            self.info_slot('--Save predict peaks to:' + save_dir + f'/{file_name}')


    
    def visual_slot(self):
        index = 1
        if self.ui.lut is not None:
            index = index + 1
            if self.ui.peak is not None:
                index = index + 1
                if self.ui.binary_border is not None:
                    index = index + 1
                if self.ui.cluster_lut is not None:
                    index = index + 1
        self.ui.vis = (self.ui.vis+1)%index
        if self.ui.vis == 0:
            self.ui.canvas.axes.clear()
            self.show(None)
        elif self.ui.vis == 1:
            self.show(lut=self.ui.lut, peaks=None, boundary=None, clusters=None)
        elif self.ui.vis == 2:
            self.show(lut=self.ui.lut, peaks=self.ui.peak, boundary=None, clusters=None)
        elif self.ui.vis == 3:
            self.show(lut=self.ui.lut, peaks=self.ui.peak, boundary=self.ui.binary_border, clusters=None)
        elif self.ui.vis == 4:
            self.show(lut=self.ui.lut, peaks=self.ui.peak, boundary=self.ui.binary_border, clusters=self.ui.cluster_lut)
        
            
    def isolate_slot(self):
        if self.ui.peak is None or self.ui.lut is None:
            self.info_slot('--Lacking input lut!')
            return
        self.ui.cluster_lut, self.ui.binary_border, = isolate.clusterAndBoundary(self.ui.lut, self.ui.peak)
        self.show(lut=self.ui.lut, peaks=self.ui.peak, boundary=self.ui.binary_border, clusters=None)
        self.ui.vis = 3

    
    def info_slot(self, messages):
        self.ui.plainTextEdit.appendPlainText(messages + '\n')
        
    def get_file(self, Qmodelidx):
        # print(self.ui.model.filePath(Qmodelidx))  # Output the file path.
        # print(self.ui.model.fileName(Qmodelidx))  # Output the file name
        tmp_file_name = self.ui.filemodel.filePath(Qmodelidx)
        messages = '--Current direction:' + tmp_file_name
        self.info_slot(messages)
        if tmp_file_name.endswith('.npy'):
            lut = np.load(tmp_file_name, allow_pickle=True)
            # print(lut.shape)
            if lut.shape[0] == 256 and lut.shape[1] == 256:
                self.ui.file_name = tmp_file_name
                self.ui.lut = lut
                self.show(lut=lut)
                self.ui.vis = 1
                num = os.path.splitext(os.path.basename(tmp_file_name))[0].split('_')[0]
                direct = os.path.dirname(tmp_file_name)
                gt_peak_name = direct.replace('imgs', 'peaks') + f'/{num}.npy'
                if os.path.exists(gt_peak_name):
                    self.ui.peak = np.load(gt_peak_name, allow_pickle=True)
                    # random color per cell so that neighbouring cells stay visually distinct
                    self.ui.cluster_lut, self.ui.binary_border, = isolate.clusterAndBoundary(self.ui.lut, self.ui.peak)
                    self.show(lut=self.ui.lut, peaks=self.ui.peak, boundary=self.ui.binary_border, clusters=None)
                    self.ui.vis = 3
                    
                # if os.path.exists(tmp_file_name.replace('imgs', 'peaks')):
                #     peak = np.load(tmp_file_name.replace('imgs', 'peaks'), allow_pickle=True)
                #     if  peak.shape[0] != 256 or peak.shape[1] != 2:
                #         self.ui.peak = None
                #         self.show(lut=lut)
                #         self.ui.file_name = tmp_file_name
                #         self.ui.vis = 1
                #         # QMessageBox
                #         return
                #     tmp_cluster_lut, tmp_binary_border, = isolate.clusterAndBoundary(self.ui.lut, peak, list(range(1, 257)) )
                #     mask = (tmp_cluster_lut != self.ui.cluster_lut).astype(np.uint8) 
                #     self.show(lut=lut*mask, peaks=None)
                #     self.ui.file_name = tmp_file_name
                #     self.ui.vis = 1
                #     return
            if lut.shape[0] == 256 and lut.shape[1] == 2:
                self.ui.file_name = tmp_file_name
                self.ui.gt = lut
                self.ui.peak = lut
                self.show(lut=self.ui.lut, peaks=lut)
                self.ui.vis = 2
    
    def show(self, width=None, height=None, lut=None, peaks=None, boundary=None, clusters=None):
        rectItem = self.ui.main_widget.scene().itemsBoundingRect()
        rectView = self.ui.main_widget.rect()
        ratioView = rectView.height() / rectView.width()
        ratioItem = rectItem.height() / rectItem.width()
        if ratioView > ratioItem:
            rectItem.moveTop(rectItem.width()*ratioView - rectItem.height())
            rectItem.setHeight(rectItem.width()*ratioView)
            rectItem.setWidth(rectItem.width())
            rectItem.setHeight(rectItem.height())
        else:
            rectItem.moveLeft(rectItem.height()/ratioView - rectItem.width())
            rectItem.setWidth(rectItem.height()/ratioView)
            rectItem.setWidth(rectItem.width())
            rectItem.setHeight(rectItem.height())

        self.ui.main_widget.fitInView(rectItem, Qt.KeepAspectRatio)
        if width is not None and height is not None:
            self.ui.canvas.fig.set_figheight(height/self.ui.canvas.fig.dpi)
            self.ui.canvas.fig.set_figwidth(width/self.ui.canvas.fig.dpi)
        self.ui.canvas.axes.clear()
        self.ui.canvas.axes.set_facecolor('white')
        for spine in self.ui.canvas.axes.spines.values():
            spine.set_visible(False)  # remove the rectangular frame around the image
        # fig.plot_array_3d_surface(lut.reshape(256, 256))
        if lut is not None:
                ax = self.ui.canvas.axes
                ax.set_facecolor('white')  # white axes background
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(lut, #isolate.high_pass_by_region(lut)[1],
                        cmap= LUT_CMAP, #'jet', #
                        # interpolation='bilinear',
                        norm=Normalize(vmin=lut.min(), vmax=lut.max()))

                if peaks is not None:
                    # plot peaks
                    # self.ui.canvas.axes.plot(peaks, 'ro', markersize=30)
                    self.ui.canvas.axes.plot(peaks[0::4, 1], peaks[0::4, 0], 'ro', markersize=30)
                    self.ui.canvas.axes.plot(peaks[1::4, 1], peaks[1::4, 0], 'ro', markersize=30)
                    self.ui.canvas.axes.plot(peaks[2::4, 1], peaks[2::4, 0], 'ro', markersize=30)
                    self.ui.canvas.axes.plot(peaks[3::4, 1], peaks[3::4, 0], 'ro', markersize=30)
                    # self.ui.canvas.axes.plot(peaks[0::4, 1], peaks[0::4, 0], 'ro', markersize=30)
                    # self.ui.canvas.axes.plot(peaks[1::4, 1], peaks[1::4, 0], 'wo', markersize=30)
                    # self.ui.canvas.axes.plot(peaks[2::4, 1], peaks[2::4, 0], 'yo', markersize=30)
                    # self.ui.canvas.axes.plot(peaks[3::4, 1], peaks[3::4, 0], 'bo', markersize=30)
                    
                # connect each peak_i to peak_{i+1}, unless (i+1)%16 == 0
                if boundary is not None:
                    ax.plot(peaks[:, 1], peaks[:, 0], 'ko', markersize=30)
                    # for i in range(len(peaks) - 1):
                    #     if (i + 1) % 16 != 0:
                    #         y0, x0 = peaks[i]
                    #         y1, x1 = peaks[i + 1]
                    #         ax.plot([x0, x1], [y0, y1], 'k-', linewidth=10)
                    # boundary is 0/1 mask of shape (H, W)
                    H, W = boundary.shape
                    # build an RGBA image: white ([1,1,1]) where boundary==1, transparent elsewhere
                    overlay = np.zeros((H, W, 4), dtype=float)
                    overlay[..., :3] = 1.0            # R=G=B=1 (white)
                    overlay[..., 3] = boundary       # alpha = 1 on boundary, 0 elsewhere
                    ax.imshow(overlay, interpolation='nearest')
                    
        # if lut is not None:
        #     # self.ui.canvas.fig.savefig("./new_result.png", bbox_inches='tight')
        #     self.ui.canvas.axes.imshow(lut, cmap='jet', interpolation='bilinear', norm=Normalize(vmin=np.min(lut), vmax=np.max(lut)))
        #     if peaks is not None:
        #         self.ui.canvas.axes.plot(peaks[0::4, 1], peaks[0::4, 0], 'bo', markersize=20)
        #         self.ui.canvas.axes.plot(peaks[1::4, 1], peaks[1::4, 0], 'bo', markersize=20)
        #         self.ui.canvas.axes.plot(peaks[2::4, 1], peaks[2::4, 0], 'bo', markersize=20)
        #         self.ui.canvas.axes.plot(peaks[3::4, 1], peaks[3::4, 0], 'bo', markersize=20)
        #         if boundary is not None:
        #             boundary = 255 - boundary
        #             # lut = lut + boundary
        #             self.ui.canvas.axes.set_facecolor('white')
        #             self.ui.canvas.axes.imshow(boundary, cmap='gray')
                if clusters is not None:
                    self.ui.canvas.axes.imshow(clusters, cmap='jet')

        if peaks is not None and lut is None:
                # peaks only (no image loaded): draw them in image coordinates on a white canvas
                ax = self.ui.canvas.axes
                ax.set_facecolor('white')
                ax.plot(peaks[0::4, 1], peaks[0::4, 0], 'ro', markersize=30)
                ax.plot(peaks[1::4, 1], peaks[1::4, 0], 'ro', markersize=30)
                ax.plot(peaks[2::4, 1], peaks[2::4, 0], 'ro', markersize=30)
                ax.plot(peaks[3::4, 1], peaks[3::4, 0], 'ro', markersize=30)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlim(0, 255)
                ax.set_ylim(255, 0)
                ax.set_aspect('equal')

        if peaks is not None or lut is not None:
            self.ui.canvas.fig.canvas.draw()
         # Add double-click event monitoring
        def on_double_click(event):
            if event.dblclick:  # Detect a double-click event
                if event.xdata is not None and event.ydata is not None:  # Ensure coordinates are valid
                    x, y = int(event.xdata), int(event.ydata)  # Get image coordinates of the double-click
                    self.handle_double_click(x, y)

        # Make sure the event is bound only once
        if not hasattr(self, '_double_click_connected') or not self._double_click_connected:
            self.ui.canvas.fig.canvas.mpl_connect('button_press_event', on_double_click)
            self._double_click_connected = True
    
    
    def handle_double_click(self, x, y):
        self.info_slot(f"--Handling double-click at: ({y}, {x})")
        if self.ui.vis==2:
            peak = self.ui.peak.copy()
            self.ui.peak = isolate.replace_nearest_point(peak, y, x)
            self.ui.binary_border = None
            self.ui.cluster_lut = None
            self.show(lut=self.ui.lut, peaks=self.ui.peak, boundary=None, clusters=None)
            # Ensure the folder exists before saving
            current_dir = os.path.dirname(self.ui.file_name)
            file_name = os.path.basename(self.ui.file_name)
            # Get the parent directory
            parent_dir = os.path.dirname(current_dir)
            save_dir = parent_dir + '/pred_peaks'
            os.makedirs(save_dir, exist_ok=True)

            # Save the file
            np.save(save_dir + f'/{file_name}', self.ui.peak)
            self.info_slot('--Save new predict peaks to:' + save_dir + f'/{file_name}')
    
    
    def model_select_slot(self):
        if self.ui.model_select.currentText() == 'ResNet18':
            torch.cuda.empty_cache()
            self.ui.net_name = 'ResNet18'
            self.ui.net = models.resnet_reg.resnet18_reg().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.ResNet18, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'GCDLNet':
            self.ui.net_name = 'GCDLNet'
            torch.cuda.empty_cache()
            self.ui.net = models.GCDLNet.GCDLNet(args.MeanModel).to(args.device)
            self.ui.net.load_state_dict(torch.load(args.GCDLNet, map_location=torch.device('cpu')), strict=True)
        
        elif self.ui.model_select.currentText() == 'ResNet34':
            self.ui.net_name = 'ResNet34'
            torch.cuda.empty_cache()
            # self.ui.net = models.resnet_reg.resnet34_reg().to(args.device)
            # self.ui.net.load_state_dict(torch.load(args.ResNet34, map_location=torch.device('cpu')), strict=False)
            self.ui.net = models.denseNet_reg.denseNet121_reg().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.DenseNet121, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'DenseNet121':
            self.ui.net_name = 'DenseNet121'
            torch.cuda.empty_cache()
            self.ui.net = models.denseNet_reg.denseNet121_reg().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.DenseNet121, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'DenseNet121_mean':
            self.ui.net_name = 'DenseNet121_mean'
            torch.cuda.empty_cache()
            self.ui.net = models.denseNet_reg.denseNet121_reg().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.DenseNet121_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'UNet':
            self.ui.net_name = 'UNet'
            torch.cuda.empty_cache()
            self.ui.net = models.unet_reg.Unet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.UNet, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'UNet_mean':
            self.ui.net_name = 'UNet_mean'
            torch.cuda.empty_cache()
            self.ui.net = models.unet_reg.Unet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.UNet_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'UNet':
            self.ui.net_name = 'UNet'
            torch.cuda.empty_cache()
            self.ui.net = models.unet_reg.Unet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.UNet, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'UNet3+':
            self.ui.net_name = 'UNet3+'
            torch.cuda.empty_cache()
            self.ui.net = models.UNet_3plus.UNet3Plus().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.UNet3plus, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'UNet3+_mean':
            self.ui.net_name = 'UNet3+_mean'
            torch.cuda.empty_cache()
            self.ui.net = models.UNet_3plus.UNet3Plus().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.UNet3plus_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'Laddernet':
            self.ui.net_name = 'Laddernet'
            torch.cuda.empty_cache()
            self.ui.net = models.LadderNet.LadderNetv6().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.Laddernet, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'Laddernet_mean':
            self.ui.net_name = 'Laddernet_mean'
            torch.cuda.empty_cache()
            self.ui.net = models.LadderNet.LadderNetv6().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.Laddernet_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'Convnext':
            self.ui.net_name = 'Convnext'
            torch.cuda.empty_cache()
            self.ui.net = models.ConvNext.ConvNeXtV2().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.Convnext, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'Convnext_mean':
            self.ui.net_name = 'Convnext_mean'
            torch.cuda.empty_cache()
            self.ui.net = models.ConvNext.ConvNeXtV2().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.Convnext_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'finalNet_mean':
            torch.cuda.empty_cache()
            self.ui.net_name = 'finalNet_mean'
            self.ui.net = models.finalnet.finalnet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.finalNet_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'stRegNet_mean':
            torch.cuda.empty_cache()
            self.ui.net_name = 'stRegNet_mean'
            self.ui.net = models.satten_RegNet.st_RegNet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.stRegNet_mean, map_location=torch.device('cpu')), strict=False)
        elif self.ui.model_select.currentText() == 'stRegNet_mean':
            torch.cuda.empty_cache()
            self.ui.net_name = 'stRegNet_mean'
            self.ui.net = models.satten_RegNet.st_RegNet().to(args.device)
            self.ui.net.load_state_dict(torch.load(args.stRegNet_mean, map_location=torch.device('cpu')), strict=False)
            
            
