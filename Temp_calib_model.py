# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:38:06 2019

@author: hzhong
"""

import sys
from PyQt5 import  QtWidgets
from PyQt5.QtWidgets import (QMainWindow, QLabel, QGridLayout, QWidget, QPushButton,
        QLineEdit, QGroupBox,QVBoxLayout,QFileDialog,QTextEdit,QHBoxLayout, 
        QCheckBox, QComboBox)
from PyQt5.QtCore import QSize  
import pyqtgraph as pg
import os
import numpy as np
import matplotlib.pyplot as plt
from lmfit import models
from scipy import signal,interpolate
import pandas as pd
import peakutils
import random


def generate_model(spec):
    composite_model = None
    params = None
    x = spec['x']
    y = spec['y']
    x_min = np.min(x)
    x_max = np.max(x)
    x_range = x_max - x_min
    y_max = np.max(y)
    for i, basis_func in enumerate(spec['model']):
        prefix = f'm{i}_'
        model = getattr(models, basis_func['type'])(prefix=prefix)
        if basis_func['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel','PseudoVoigtModel']: # for now VoigtModel has gamma constrained to sigma
            model.set_param_hint(prefix+'sigma', min=1e-6, max=x_range)
            model.set_param_hint(prefix+'center', min=x_min, max=x_max)
            model.set_param_hint(prefix+'height', min=1e-6, max=2*y_max)
            model.set_param_hint(prefix+'amplitude', min=1e-6)
            model.set_param_hint(prefix+'fwhm', min=1e-6, max=x_range)
            # default guess is horrible!! do not use guess()
            default_params = {
                prefix+'center': x_min + x_range * random.random(),
                prefix+'height': y_max * random.random(),
                prefix+'sigma': x_range * random.random()
            }

        else:
            raise NotImplementedError(f'model {basis_func["type"]} not implemented yet')
        if 'help' in basis_func:  # allow override of settings in parameter
            for param, options in basis_func['help'].items():
                model.set_param_hint(param, **options)
        
        model_params = model.make_params(**default_params, **basis_func.get('params', {}))
        if params is None:
            params = model_params
        else:
            params.update(model_params)
        if composite_model is None:
            composite_model = model
        else:
            composite_model = composite_model + model
            
    return composite_model, params

def update_spec_from_peaks(spec, model_indicies, peak_widths=(10, 25), peak_min=200, **kwargs):
    x = spec['x']
    y = spec['y']
    x_range = np.max(x) - np.min(x)
    peak_indicies,_ = signal.find_peaks(y, height=peak_min)
#    N_peaks=len(peak_indicies)
    for peak_indicie, model_indicie in zip(peak_indicies.tolist(), model_indicies):
        
        model = spec['model'][model_indicie]
        if model['type'] in ['GaussianModel', 'LorentzianModel', 'VoigtModel','PseudoVoigtModel']:
            params = {
                'height': y[peak_indicie],
                'sigma': x_range / len(x) * np.min(peak_widths),
                'center': x[peak_indicie]
            }
            if 'params' in model:
                model.update(params)
            else:
                model['params'] = params
        else:
            raise NotImplementedError(f'model {["type"]} not implemented yet')
        
    return peak_indicies

def print_best_values(spec, output):
    model_params = {
        'GaussianModel':   ['amplitude', 'sigma'],
        'LorentzianModel': ['amplitude', 'sigma'],
        'VoigtModel':      ['amplitude', 'sigma', 'gamma'],
        'PseudoVoigtModel':['amplitude', 'fwhm']
    }
    best_values = output.best_values
    print('center    model   amplitude     sigma      gamma')
    for i, model in enumerate(spec['model']):
        prefix = f'm{i}_'
        values = ', '.join(f'{best_values[prefix+param]:8.3f}' for param in model_params[model["type"]])
        print(f'[{best_values[prefix+"center"]:3.3f}] {model["type"]:16}: {values}')        

     
class HelloWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):    
        self.setMinimumSize(QSize(100, 100))    
        self.setWindowTitle("Temperature Calibration Tool")
        
        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   
 
        gridLayout = QGridLayout(self)     
        centralWidget.setLayout(gridLayout)  
 
        
        menu = self.menuBar().addMenu('Action for quit')
        action = menu.addAction('Quit')
        action.triggered.connect(QtWidgets.QApplication.quit)
        menu_clear = self.menuBar().addMenu('Clear Plot')
        act_clearall = menu_clear.addAction('Clear All')
        act_clearall.triggered.connect(self.delete_plot)
        act_clearpeak = menu_clear.addAction('Clear Peak Plot')
        act_clearpeak.triggered.connect(self.delete_peakplot)
        act_clearfitting = menu_clear.addAction('Clear Fitting Plots')
        act_clearfitting.triggered.connect(self.delete_fittingplot)
        act_clearresult = menu_clear.addAction('Clear Result Field')
        act_clearresult.triggered.connect(self.delete_resultfield)        
        self.creategroup1()
        self.creategroup2()
        self.creategroup3()
        
        gridLayout.addWidget(self.layout1,0,0)
        gridLayout.addWidget(self.layout2,0,1,2,1)
        gridLayout.addWidget(self.layout3,1,0)

        self.setLayout(gridLayout)
        
    def creategroup1(self):
        self.layout1 = QGroupBox("Load File")
        
        self.loadfilebutton=QPushButton("Load File")
        self.loadfilebutton.setCheckable(True)
        self.loadfilebutton.clicked.connect(self.loadfile)
        self.loadfilebutton.setFixedSize(70,25)
        self.filenametext=QTextEdit('Result')

        self.savebutton = QPushButton("Save")
        self.savebutton.setCheckable(True)
        self.savebutton.clicked.connect(self.save_xlsx)
        self.savebutton.setFixedSize(70,25)        
        self.upbutton = QPushButton("->")
        self.upbutton.setCheckable(True)
        self.upbutton.clicked.connect(self.plot_next)
        self.downbutton = QPushButton("<-")
        self.downbutton.setCheckable(True)
        self.downbutton.clicked.connect(self.plot_back)  
        self.currentplotid=QLineEdit('0')
#        self.currentplotid.setFixedWidth(30)
        self.currentplotid.returnPressed.connect(self.plot_current)        
        self.creategroup4()
        self.creategroup5()
        self.creategroup6()
        self.creategroup7()
        
        layout=QGridLayout()
        layout.addWidget(self.loadfilebutton,0,0)
        layout.addWidget(self.savebutton,0,1) 
        layout.addWidget(self.filenametext,1,0,1,8)
        layout.addWidget(self.layout4,2,0,1,2)
        layout.addWidget(self.layout7,2,2,1,2)        
        layout.addWidget(self.layout5, 2,4,1,2)
        layout.addWidget(self.layout6,2,6,1,2)
        layout.addWidget(self.downbutton,3,5) 
        layout.addWidget(self.currentplotid,3,6)
        layout.addWidget(self.upbutton, 3,7)       
        self.layout1.setLayout(layout)

    def creategroup2(self):
        self.layout2=QGroupBox("Fittings")
        self.pwlist=list() 
        self.pwlist.append(pg.PlotWidget(name="peak1"))
        self.pwlist.append(pg.PlotWidget(name="peak2"))
        self.pwlist.append(pg.PlotWidget(name="peak3"))
        self.pwplot=pg.PlotWidget(name='peakcenter')
        layout=QVBoxLayout()
        for i in range(len(self.pwlist)):
            layout.addWidget(self.pwlist[i])
        layout.addWidget(self.pwplot)
        
        self.layout2.setLayout(layout)

    def creategroup3(self):
        self.layout3=QGroupBox("Plots")
        
        self.pw1 = pg.PlotWidget(name='full')
        self.pw2 = pg.PlotWidget(name='zoomed')

        
        layout=QVBoxLayout()
        layout.addWidget(self.pw1)
        layout.addWidget(self.pw2)

        self.layout3.setLayout(layout)
    
    def creategroup4(self):
        self.layout4 = QGroupBox("Basic Parameters")
        baselinelabel=QLabel("Baseline range")
       
        self.basemin=QLineEdit("1.8")
        self.basemin.returnPressed.connect(self.update_basemin)  
        self.basemax=QLineEdit("14.0")
        self.basemax.returnPressed.connect(self.update_basemin) 
        wavelengthlabel=QLabel('Wavelength')
        self.wavelength=QLineEdit('0.1949')  
        emptylabel=QLabel(' ')        
        self.showbaselinebutton = QPushButton("Baseline")
        self.showbaselinebutton.setCheckable(True)
        self.showbaselinebutton.clicked.connect(self.show_baseline)

        layout=QGridLayout()
        
        layout.addWidget(wavelengthlabel,0,0)
        layout.addWidget(self.wavelength,0,1) 
        layout.addWidget(baselinelabel,1,0,1,2)
        layout.addWidget(self.basemin, 2,0)
        layout.addWidget(self.basemax,2,1)
        layout.addWidget(emptylabel,3,0,2,1)
        layout.addWidget(self.showbaselinebutton,5,1)               
        self.layout4.setLayout(layout)   
        
    def creategroup7(self):
        self.layout7 = QGroupBox("ROI ")
        tthminlabel=QLabel("Min tth")
        tthmaxlabel=QLabel("Max tth")
        self.tthmin=QLineEdit("4.0")
        self.tthmin.returnPressed.connect(self.update_tthmin)  
        self.tthmax=QLineEdit("7.0")
        self.tthmax.returnPressed.connect(self.update_tthmin) 
        peaknumlabel=QLabel("Peaks")
        peakminlabel=QLabel("Peak Min")
        self.peaknum=QLineEdit("1")
        self.peakmin=QLineEdit("20.0")
        self.plotfitbutton = QPushButton("Find Peak")
        self.plotfitbutton.setCheckable(True)
        self.plotfitbutton.clicked.connect(self.plot_peak)
        emptylabel=QLabel(' ')
        
        layout=QGridLayout()
        layout.addWidget(tthminlabel,0,0)
        layout.addWidget(self.tthmin, 0,1)
        layout.addWidget(tthmaxlabel,1,0)
        layout.addWidget(self.tthmax,1,1)
        layout.addWidget(peaknumlabel,2,0)
        layout.addWidget(self.peaknum, 2,1)        
        layout.addWidget(peakminlabel,3,0)
        layout.addWidget(self.peakmin,3,1) 
        layout.addWidget(emptylabel, 4,0)
        layout.addWidget(self.plotfitbutton,5,1)               
        self.layout7.setLayout(layout)         
        
        
        
    def creategroup5(self):
        self.layout5 = QGroupBox("Fit Parameters")
        self.runfitbutton = QPushButton("Fit One")
        self.runfitbutton.setCheckable(True)
        self.runfitbutton.clicked.connect(self.run_one_fitting)
        self.runfitbutton.setFixedSize(70,25)
        self.runallfitbutton = QPushButton("Fit All")
        self.runallfitbutton.setCheckable(True)
        self.runallfitbutton.clicked.connect(self.run_fitting)
        self.runallfitbutton.setFixedSize(70,25)        
        sigmalabel=QLabel("Sigma Max")
        widstartlabel=QLabel("wid")
        self.sigmaend=QLineEdit("0.02 ")
        self.widstart=QLineEdit("30")  
        self.modelComboBox=QComboBox()
        model_list=['GaussianModel', 'LorentzianModel', 'VoigtModel','PseudoVoigtModel']
        self.modelComboBox.addItems(model_list)
        self.modelComboBox.currentIndexChanged.connect(self.changemodel)
        self.modelLabel=QLabel('Model: ')
        self.modelLabel.setBuddy(self.modelComboBox)
        self.emptylabel=QLabel(' ')
        layout=QGridLayout()

        layout.addWidget(self.modelLabel, 0,0)
        layout.addWidget(self.modelComboBox,1,0,1,2)
        layout.addWidget(self.emptylabel,4,0,1,2)
        layout.addWidget(widstartlabel,2,0)
        layout.addWidget(self.widstart,2,1)             
        layout.addWidget(sigmalabel,3,0)
        layout.addWidget(self.sigmaend,3,1)  
        layout.addWidget(self.runfitbutton,5,0)
        layout.addWidget(self.runallfitbutton,5,1)
        self.layout5.setLayout(layout)           
        
    def creategroup6(self):
        self.layout6=QGroupBox("Temperature Calibration")
        self.tempcalibbutton=QPushButton("Calib. All")
        self.tempcalibbutton.setFixedSize(70,25)
        self.tempcalibbutton.setCheckable(True)
        self.tempcalibbutton.clicked.connect(self.temp_calibrate)
        self.onetempcalibbutton=QPushButton("Calib. One")
        self.onetempcalibbutton.setCheckable(True)
        self.onetempcalibbutton.clicked.connect(self.one_temp)
        self.onetempcalibbutton.setFixedSize(70,25)
        self.loadroomtempfileButton=QPushButton('Load RT File')
        self.loadroomtempfileButton.setCheckable(True)
        self.loadroomtempfileButton.clicked.connect(self.load_roomtemp_file)
        self.loadroomtempfileButton.setFixedSize(70,25)        
        self.usenistdata=QCheckBox('Use Standard CeO2')
        self.standardDcombox=QComboBox()
        self.standard_list=['111 3.123442','200 2.705643','220 1.913411','311 1.631811','222 1.562221','400 1.353080',
                            '331 1.241490','420 1.210120','422 1.104780', '511 1.041491', '440 0.956620','531 0.914710',
                            '600 0.901900','620 0.855610','533 0.825180','622 0.815750']
        self.standardDcombox.addItems(self.standard_list)
        self.standardDcombox.currentIndexChanged.connect(self.add_standardD)
        self.standardD=QLineEdit('0')
        self.roomtemp=QLineEdit('293')
        self.roomtemp.setFixedSize(70,25)
        
        self.calibrateComboBox=QComboBox()
        calib_list=['CeO2', 'Pt']
        self.calibrateComboBox.addItems(calib_list)
        self.calibrateComboBox.currentIndexChanged.connect(self.changeCalibrate)
        self.calibrateLabel=QLabel('&Calibrator:')
        self.calibrateLabel.setBuddy(self.calibrateComboBox)
        self.calibrateComboBox.setFixedSize(140,25)
#        emptylabel=QLabel(' ')
        layout=QGridLayout()


        layout.addWidget(self.roomtemp, 2,1)
        layout.addWidget(self.loadroomtempfileButton, 2,0)
        layout.addWidget(self.calibrateLabel, 0,0)
        layout.addWidget(self.calibrateComboBox, 1,0,1,2)
        layout.addWidget(self.usenistdata,3,0,1,2)
        layout.addWidget(self.standardDcombox, 4,0)
        layout.addWidget(self.standardD, 4,1)
#        layout.addWidget(emptylabel,3,0,2,2)
        layout.addWidget(self.onetempcalibbutton, 5,0)
        layout.addWidget(self.tempcalibbutton, 5,1)        
        self.layout6.setLayout(layout)
    
    def changemodel(self):
        self.select_model=self.modelComboBox.currentText()        

    def add_standardD(self):
        standardD=self.standardDcombox.currentText()
        try:
            standardD=standardD.split(' ')[1]
        except IndexError:
            standardD=' '
        self.standardD.setText(standardD) 
        
        
    def show_baseline(self):
        current_id=int(self.currentplotid.text())
        filenames=self.filenames[0]
        current_filename=filenames[current_id]    
        xmin=float(self.tthmin.text())
        xmax=float(self.tthmax.text())        
        self.pw2.clear()
        xrange, yrange,yrange_base,yrange_baseline=self.loadxy_sp(current_filename,xmin, xmax)
        self.pw2.plot(xrange,yrange)
        self.pw2.plot(xrange, yrange_base,pen=5)
        self.pw2.plot(xrange, yrange_baseline, pen=3)
    
    
    def plot_next(self):
        current_id=int(self.currentplotid.text())
        filenames = self.filenames[0]
        
        if current_id < len(filenames)-2:
            current_id = current_id+1
        else: 
            current_id = current_id
            
        current_filename = filenames[current_id]
        self.currentplotid.setText(str(current_id))
        self.plotupdate(current_filename)

    def plot_back(self):
        current_id=int(self.currentplotid.text())
        filenames=self.filenames[0]
        if current_id >=1:
            current_id=current_id-1
        else: 
            current_id=current_id
        current_filename=filenames[current_id]
        self.currentplotid.setText(str(current_id))
        self.plotupdate(current_filename)
        
    def plot_current(self):
        current_id=int(self.currentplotid.text())
        filenames=self.filenames[0]
        if current_id >= len(filenames):
            current_id=len(filenames)-1
        current_filename=filenames[current_id]
        self.plotupdate(current_filename)        

    def update_tthmin(self):
        xmin = float(self.tthmin.text())
        xmax = float(self.tthmax.text())
        self.rois.setRegion([xmin, xmax])
        self.update()
       
    def update_basemin(self):
        self.baselinemin = float(self.basemin.text())
        self.baselinemax = float(self.basemax.text())
        current_id = int(self.currentplotid.text())
        filenames = self.filenames[0]
        current_filename = filenames[current_id]
        self.plotupdate(current_filename)        
        
    def plotupdate(self, filename):
        xmin=float(self.tthmin.text())
        xmax=float(self.tthmax.text())        
        self.pw1.clear()
        self.pw2.clear()
        x,y,y_base = self.loadxy_total(filename)
        self.pw1.plot(x,y)
        self.pw1.plot(x, y_base,pen=5)
        self.rois=pg.LinearRegionItem([xmin,xmax])
        self.pw1.addItem(self.rois)
        self.rois.sigRegionChanged.connect(self.update)  
        xrange, yrange,yrange_base,yrange_baseline=self.loadxy_sp(filename,xmin, xmax)
        self.pw2.plot(xrange,yrange)
        self.pw2.plot(xrange, yrange_base,pen=5)
        self.plot_peak()
        
    def changeCalibrate(self):
        self.select_calibrate=self.calibrateComboBox.currentText()
        self.usenistdata.setText('Use Standard '+self.select_calibrate)
    
    
    def load_roomtemp_file(self):
        filters = "xy files ( *.xy);;chi files(*.chi) ;;tth chi (*_tth.chi);; all files(*.*)"
        path = "C:/Users/admin/Documents/"
        roomtempfilename = QFileDialog.getOpenFileName(self, 'Open file',
               path ,filters)
        if roomtempfilename:
            self.roomtempfilename = roomtempfilename
        
    def temp_calibrate_function(self, dspace):

        peak_sum=int(self.peaknum.text())
        current_calibrate=self.calibrateComboBox.currentText()
        scanid=range(len(dspace))

        if current_calibrate == 'CeO2':
            dd_calib=[-0.17,-0.085,0,0.106,0.21,0.321,0.437,0.558,0.683,0.813,0.946,1.083,1.223,1.365,1.51,1.657]
            temp_calib=[100,200,293,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600]
            f = interpolate.interp1d(dd_calib, temp_calib,fill_value="extrapolate")
            f2 = interpolate.interp1d(temp_calib, dd_calib,fill_value="extrapolate")

            if self.usenistdata.isChecked()==True:
                d_room = float(self.standardD.text())
                dd_room=f2(299)/100.
                d_20=d_room/(1+dd_room)               
            else:
                roomtempfile=self.roomtempfilename[0]
                output_room=self.run_one(roomtempfile)
                roomtemp=float(self.roomtemp.text())  
                dd_room=f2(roomtemp)/100.            
                d_room=np.asarray(output_room[2])
                d_20=d_room/(1+dd_room)
            deltd=(dspace-d_20)/d_20*100.
            temp_out=f(deltd)-273.15
            try:
                out1=[temp_out[:,i] for i in range(peak_sum)]
                for peakn in range(peak_sum):
                    plt.plot(scanid, out1[peakn], 'b-+')
                    plt.show()                
            except IndexError:
                out1 = temp_out

        elif current_calibrate == 'Pt':
            dd_calib=[-0.192,-0.191,-0.186,-0.157,-0.081,0,0.096,0.189,0.288,0.388,0.49,0.593,0.699,0.92,1.157,1.414,1.69,1.837]
            temp_calib=[5,25,50,100,200,293,400,500,600,700,800,900,1000,1200,1400,1600,1800,1900]
            f = interpolate.interp1d(dd_calib, temp_calib, fill_value='extrapolate')
            f2= interpolate.interp1d(temp_calib, dd_calib, fill_value="extrapolate")
  
            if self.usenistdata.isChecked()==True:
                d_room = float(self.standardD.text())
                dd_room=f2(299)/100.
                d_20=d_room/(1+dd_room)                 
            else:
                roomtemp=float(self.roomtemp.text())
                roomtempfile=self.roomtempfilename[0]
                output_room=self.run_one(roomtempfile)
                dd_room=f2(roomtemp)/100.                
                d_room=np.asarray(output_room[2])
                d_20=d_room/(1+dd_room)
            deltd=(dspace-d_20)/d_20*100.            
            temp_out=f(deltd)-273.15
            try:
                out1=[temp_out[:,i] for i in range(peak_sum)]              
                for peakn in range(peak_sum):
                    plt.plot(scanid, out1[peakn], 'b-+')
                    plt.show()                
            except IndexError:
                out1=temp_out
        return(temp_out)            
        
    def temp_calibrate(self):
        output=self.output
        dspace=np.asarray([output[2][i,:] for i in range(len(output[2]))])
        output_temp=self.temp_calibrate_function(dspace)
        self.output_temp=output_temp            

    def one_temp(self):
        output=self.output_one
        dspace=np.asarray(output[2])
        output_temp=self.temp_calibrate_function(dspace)
        output1=f'temperature:{output_temp}'
        self.filenametext.append(output1)
        
        
    def update(self):
        xrange=self.rois.getRegion()
        xmin=float("{0:.2f}".format(xrange[0]))
        xmax=float("{0:.2f}".format(xrange[1]))
        current_id=int(self.currentplotid.text())
        filenames=self.filenames[0]
        x, y, y_base,y_baseline=self.loadxy_sp(filenames[current_id],xrange[0],xrange[1])
      
        self.pw2.plot(x,y_base,clear=True)
        self.pw2.autoRange()
        self.tthmin.setText(str(xmin))
        self.tthmax.setText(str(xmax))

    def loadfile(self):
        self.filenametext.clear()
        filters="xy files ( *.xy);;chi files(*.chi) ;; tth chi(*_tth.chi);; all files(*.*)"
        path="C:/Users/admin/Documents/"
        filenames = QFileDialog.getOpenFileNames(self, 'Open file',
                path,filters)
        
        if filenames:
            self.filenames = filenames
            filenamelist=self.filenames[0]
            self.pw1.clear()
            self.pw2.clear()
            for i in range(len(self.pwlist)):
                self.pwlist[i].clear()

            x, y_basefree, y=self.loadxy_total(filenamelist[0])
            self.pw1.plot(x,y)
            self.pw1.plot(x,y_basefree,pen=4)
            self.rois=pg.LinearRegionItem([4.0,7.0])
            self.pw1.addItem(self.rois)
            self.rois.sigRegionChanged.connect(self.update)


    def loadxy_total(self, filename):
        x,y= np.loadtxt(filename, skiprows=10,unpack=True)
        xrange=x
        yrange=y
        baseline_y=peakutils.baseline(yrange)
        yrange_basefree=yrange-baseline_y
        return xrange, yrange_basefree, yrange  
      

    def loadxy(self,filename,xmin,xmax):
        x,y= np.loadtxt(filename, skiprows=10,unpack=True)
        baseline_y=peakutils.baseline(y)
        basefree_y=y-baseline_y
        xrange_index=np.where((x>xmin)&(x<xmax))
        xrange=x[xrange_index]
        yrange=y[xrange_index]
        yrange_basefree=basefree_y[xrange_index]
        return xrange, yrange_basefree, yrange

    def loadxy_sp(self, filename,xmin,xmax):
        baselinemin=float(self.basemin.text())
        baselinemax=float(self.basemax.text())
        x,y= np.loadtxt(filename, skiprows=10,unpack=True)
        xrange_base_index=np.where((x>baselinemin)&(x<baselinemax))
        x_base=x[xrange_base_index]
        y_base=y[xrange_base_index]
        y_baseline=peakutils.baseline(y_base)
        basefree_y=y_base-y_baseline
        xrange_index=np.where((x_base>xmin)&(x_base<xmax))
        xrange=x_base[xrange_index]
        yrange=y_base[xrange_index]
        yrange_basefree=basefree_y[xrange_index]
        yrange_baseline=y_baseline[xrange_index]
#        baseline_y=peakutils.baseline(yrange)
#        yrange_basefree=yrange-baseline_y
        return xrange, yrange_basefree, yrange, yrange_baseline
    
    def create_specs(self, xrange, yrange, peaknum, model_type):
        models=[{'type': model_type}]
        if peaknum > 1:
            for n in range(peaknum-1):
                models.append({'type':model_type})
        spec= {
                'x': xrange,
                'y': yrange,
                'model':models,
            }
        return spec

    def run_one_fitting(self):
        current_id=int(self.currentplotid.text())
        filenames=self.filenames[0]
        filename=filenames[current_id]
        self.output_one = self.run_one(filename)
        
        
    def run_one(self, filename):
        wavelength=float(self.wavelength.text())
#        sigmamax=float(self.sigmaend.text())
        xmin=float(self.tthmin.text())
        xmax=float(self.tthmax.text())
        peakn=int(self.peaknum.text())
        model_type=self.modelComboBox.currentText()
        peakwidth=float(self.widstart.text())
        peakmin=float(self.peakmin.text())
        sform=peakn
        peak_list=np.zeros(sform)
        dspace_list=np.zeros(sform)
        sigma_list=np.zeros(sform)
        x, y, y_base,y_baseline=self.loadxy_sp(filename, xmin, xmax)
        peak_indicies,_ = signal.find_peaks(y, height=peakmin)            
        d_spacing=np.zeros(peakn)            
        center=np.zeros(peakn)
        sigma=np.zeros(peakn)
        spec = self.create_specs(x, y, peakn, model_type)        
        center, sigma,d_spacing=self.peak_fittings(spec, peakn, wavelength, 
                center, sigma, d_spacing, peakwidth, peakmin)
                    
        for ind, peak_center in enumerate(center):
            peak_list[ind]=peak_center
            dspace_list[ind]=d_spacing[ind]
            sigma_list[ind]=sigma[ind]            
            
        output1=f'center:{center}; d:{d_spacing}'
        self.filenametext.append(output1)
        output_one=[peak_list, sigma_list, dspace_list]
        return output_one


    def run_fitting(self):
        
        wavelength=float(self.wavelength.text())
        filenames=self.filenames[0]
        filename=os.path.basename(filenames[0])
        timestart=filename.find('2019')
        xmin=float(self.tthmin.text())
        xmax=float(self.tthmax.text())
#        sigmamax=float(self.sigmaend.text())        
        peakn=int(self.peaknum.text())
        model_types=self.modelComboBox.currentText()        
        peakwidth=float(self.widstart.text())
        peakmin=float(self.peakmin.text())
        timelist=['' for i in range(len(filenames))]
        file_len=len(filenames)
        sform=(file_len, peakn)    
        peak_list=np.zeros(sform)
        dspace_list=np.zeros(sform)
        sigma_list=np.zeros(sform)

        for i,filename in enumerate(filenames):
            filename_only=os.path.basename(filename)
            timestart=filename_only.find('2019')
            timelist[i]=filename_only[timestart:timestart+15]
            scanid=list(range(1,i+2))
            x, y, y_base,y_baseline=self.loadxy_sp(filename, xmin, xmax)
            peak_indicies,_ = signal.find_peaks(y, height=peakmin)
            peakn=len(peak_indicies)
            d_spacing=np.zeros(peakn)            
            center=np.zeros(peakn)
            sigma=np.zeros(peakn)
            spec = self.create_specs(x, y, peakn, model_types)
            center, sigma,d_spacing=self.peak_fittings(spec, peakn, wavelength, 
                       center, sigma, d_spacing, peakwidth, peakmin)
            
            try:
                for ind, peak_center in enumerate(center):
#                    d_spacing=wavelength/2/np.sin(peak_center*np.pi/180./2.)
                    peak_list[i][ind]=peak_center
                    dspace_list[i][ind]=d_spacing[ind]
                    sigma_list[i][ind]=sigma[ind]
                    self.pwplot.plot(scanid,peak_list[0:i+1,ind],pen=(ind+1,7))
                    pg.QtGui.QApplication.processEvents()                     
            except IndexError:
                print(f'scan :{i} has more peaks than others')
            output1=f'center:{center};d: {d_spacing}'
            self.filenametext.append(output1)
        self.output=[timelist, peak_list, dspace_list, sigma_list]
        self.peak_sum=peakn
        return self.output
        
    def save_xlsx(self): 
        filenames=self.filenames[0]
        path = os.path.dirname(os.path.realpath(filenames[0]))
        save_name= QFileDialog.getSaveFileName(self, 'Save file', 
                                            path,"excel files (*.xlsx)")
        save_file=save_name[0]
        writer = pd.ExcelWriter(save_file, engine='xlsxwriter')        
        output=self.output

        file_num=len(filenames)
        peaksum=self.peak_sum
        peakcenter=[output[1][i,:] for i in range(file_num)]
        dspace=[output[2][i,:] for i in range(file_num)]
        sigma=[output[3][i,:] for i in range(file_num)]
        DBcen=pd.DataFrame(peakcenter,columns=[f'tth_{i}' for i in range(peaksum)])
        DBdspace=pd.DataFrame(dspace,columns=[f'd_{i}' for i in range(peaksum)])
        DBsigma=pd.DataFrame(sigma,columns=[f'sigma_{i}' for i in range(peaksum)])         
        DBcen.to_excel(writer, sheet_name='tth')
        DBdspace.to_excel(writer, sheet_name='d_spacing')
        DBsigma.to_excel(writer, sheet_name='sigma')        
        try:
            output_temp=self.output_temp            
            temp_calib=[output_temp[i,:] for i in range(file_num)]    
            DBtemp=pd.DataFrame(temp_calib,columns=[f'temp_{i}' for i in range(peaksum)])         
            DBtemp.to_excel(writer,sheet_name='temperature')
        except AttributeError:
            print('No temperature output')
        writer.save()


    def peak_fittings(self, spec, peakn, wavelength, center, sigma, d_spacing, peak_widths=0.2, peak_min=200):
#        model_types=self.modelComboBox.currentText()
        model_list=list(range(peakn))
        x=spec['x']
        peak_indicies=update_spec_from_peaks(spec, model_list,peak_width=(peak_widths, peak_widths), peak_min=peak_min)
        if len(peak_indicies) > 0:
            model, params= generate_model(spec)    
            self.pw2.plot(spec['x'],spec['y'])
            for i in peak_indicies:
                line=pg.InfiniteLine(angle=90)
                line.setValue(x[i])
                self.pw2.addItem(line,markers=('.'))
            output = model.fit(spec['y'], params, x=spec['x'])
            components=output.eval_components(x=spec['x'])
            best_values = output.best_values
            for j, model_type in enumerate(spec['model']):
                center[j]=best_values[f'm{j}_'+"center"]
                sigma[j] =best_values[f'm{j}_'+"sigma"]
                d_spacing[j]=wavelength/2/np.sin(center[j]*np.pi/180./2.)
            zipped=list(zip(center, sigma,d_spacing))
            result = sorted(zipped, key = lambda x: x[0])
            center=[num[0] for num in result]
            sigma =[num[1] for num in result]
            d_spacing=[num[2] for num in result]
#        fig,axes = plt.subplots(peakn)
#        fig.suptitle(f'{model_types}')
            for j, model_type in enumerate(spec['model']):
                self.pwlist[0].plot(spec['x'],components[f'm{j}_'],pen=(j,7))
                self.pwlist[0].plot(spec['x'], spec['y'])
                pg.QtGui.QApplication.processEvents()
#            axes[j].plot(spec['x'],components[f'm{j}_'], spec['x'], spec['y'],'.')
#        plt.show() 
        else:
            print('NO Peak is found')
        return center, sigma, d_spacing
                 

    def plot_peak(self):
        current_id=int(self.currentplotid.text())
        filenames=self.filenames[0]
        xmin=float(self.tthmin.text())
        xmax=float(self.tthmax.text())
        peakmin=float(self.peakmin.text())
        filename=filenames[current_id]
        x, y, y_base,y_baseline=self.loadxy_sp(filename, xmin, xmax)
        peak_indicies,_ = signal.find_peaks(y, height=peakmin)
        peakn=len(peak_indicies)
        self.peaknum.setText(str(peakn))
        self.pw2.plot(x,y)
        for i in peak_indicies:
            line=pg.InfiniteLine(angle=90)
            line.setValue(x[i])
            self.pw2.addItem(line,markers=('.'))  
        return None

    def delete_plot(self):
        self.pw2.clear()
        self.pwplot.clear()
        for i in range(len(self.pwlist)):
            self.pwlist[i].clear() 
        self.filenametext.clear()            
        return None
    
    def delete_peakplot(self):
        self.pw2.clear()
        return None

    def delete_fittingplot(self):
        self.pwplot.clear()
        for i in range(len(self.pwlist)):
            self.pwlist[i].clear()        
        return None    

    def delete_resultfield(self):
        self.filenametext.clear()
        return None
    
def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))


if __name__ == "__main__":
    def run_app():
        app = QtWidgets.QApplication(sys.argv)
        mainWin = HelloWindow()
        mainWin.show()
        app.exec_()
    run_app()