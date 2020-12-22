import cv2
import dlib
import threading
import multiprocessing
import faceRecognition
import face_recognition
import pickle
import os
import shutil
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def __init__(self):
        self.capSource = None
        self.updateFrameIsRunning = False

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(882, 798)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.DisplayFrame = QtWidgets.QFrame(self.centralwidget)
        self.DisplayFrame.setGeometry(QtCore.QRect(210, 10, 661, 531))
        self.DisplayFrame.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.DisplayFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.DisplayFrame.setObjectName("DisplayFrame")
        self.DisplayLabel = QtWidgets.QLabel(self.DisplayFrame)
        self.DisplayLabel.setGeometry(QtCore.QRect(10, 10, 640, 480))
        self.DisplayLabel.setStyleSheet("QLabel {\n""    background:rgb(29, 29, 29);\n""}")
        self.DisplayLabel.setScaledContents(True)
        self.DisplayLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.DisplayLabel.setObjectName("DisplayLabel")
        self.layoutWidget = QtWidgets.QWidget(self.DisplayFrame)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 500, 641, 25))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.PreviewFrameCB = QtWidgets.QCheckBox(self.layoutWidget)
        self.PreviewFrameCB.setObjectName("PreviewFrameCB")
        self.horizontalLayout_3.addWidget(self.PreviewFrameCB)
        spacerItem = QtWidgets.QSpacerItem(268, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem)
        self.StartCaptureBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.StartCaptureBtn.setObjectName("StartCaptureBtn")
        self.horizontalLayout_3.addWidget(self.StartCaptureBtn)
        self.StopCaptureBtn = QtWidgets.QPushButton(self.layoutWidget)
        self.StopCaptureBtn.setObjectName("StopCaptureBtn")
        self.horizontalLayout_3.addWidget(self.StopCaptureBtn)
        self.SelectInputButton = QtWidgets.QPushButton(self.layoutWidget)
        self.SelectInputButton.setObjectName("SelectInputButton")
        self.horizontalLayout_3.addWidget(self.SelectInputButton)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 10, 191, 531))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.LeftPanel = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.LeftPanel.setContentsMargins(0, 0, 0, 0)
        self.LeftPanel.setObjectName("LeftPanel")
        self.FaceRecogFrame = QtWidgets.QFrame(self.layoutWidget1)
        self.FaceRecogFrame.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.FaceRecogFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.FaceRecogFrame.setObjectName("FaceRecogFrame")
        self.FaceRecogLabel = QtWidgets.QLabel(self.FaceRecogFrame)
        self.FaceRecogLabel.setGeometry(QtCore.QRect(10, 0, 91, 31))
        self.FaceRecogLabel.setObjectName("FaceRecogLabel")
        self.layoutWidget2 = QtWidgets.QWidget(self.FaceRecogFrame)
        self.layoutWidget2.setGeometry(QtCore.QRect(10, 30, 171, 35))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.FrEnableLabel = QtWidgets.QLabel(self.layoutWidget2)
        self.FrEnableLabel.setObjectName("FrEnableLabel")
        self.horizontalLayout.addWidget(self.FrEnableLabel)
        self.FrEnableCB = QtWidgets.QCheckBox(self.layoutWidget2)
        self.FrEnableCB.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.FrEnableCB.setText("")
        self.FrEnableCB.setObjectName("FrEnableCB")
        self.horizontalLayout.addWidget(self.FrEnableCB)
        self.layoutWidget3 = QtWidgets.QWidget(self.FaceRecogFrame)
        self.layoutWidget3.setGeometry(QtCore.QRect(10, 60, 171, 35))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.ToleranceLabel = QtWidgets.QLabel(self.layoutWidget3)
        self.ToleranceLabel.setObjectName("ToleranceLabel")
        self.horizontalLayout_2.addWidget(self.ToleranceLabel)
        self.ToleranceSlider = QtWidgets.QSlider(self.layoutWidget3)
        self.ToleranceSlider.setMaximum(100)
        self.ToleranceSlider.setProperty("value", 55)
        self.ToleranceSlider.setOrientation(QtCore.Qt.Horizontal)
        self.ToleranceSlider.setInvertedAppearance(False)
        self.ToleranceSlider.setInvertedControls(False)
        self.ToleranceSlider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.ToleranceSlider.setObjectName("ToleranceSlider")
        self.horizontalLayout_2.addWidget(self.ToleranceSlider)
        self.line = QtWidgets.QFrame(self.FaceRecogFrame)
        self.line.setGeometry(QtCore.QRect(0, 20, 191, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.TrainFaceGroup = QtWidgets.QGroupBox(self.FaceRecogFrame)
        self.TrainFaceGroup.setGeometry(QtCore.QRect(10, 100, 171, 121))
        self.TrainFaceGroup.setObjectName("TrainFaceGroup")
        self.layoutWidget4 = QtWidgets.QWidget(self.TrainFaceGroup)
        self.layoutWidget4.setGeometry(QtCore.QRect(0, 20, 171, 91))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget4)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.TrainFromFileBtn = QtWidgets.QPushButton(self.layoutWidget4)
        self.TrainFromFileBtn.setObjectName("TrainFromFileBtn")
        self.verticalLayout.addWidget(self.TrainFromFileBtn)
        self.ReTrainBtn = QtWidgets.QPushButton(self.layoutWidget4)
        self.ReTrainBtn.setObjectName("ReTrainBtn")
        self.verticalLayout.addWidget(self.ReTrainBtn)
        self.TrainFromFrameBtn = QtWidgets.QPushButton(self.layoutWidget4)
        self.TrainFromFrameBtn.setObjectName("TrainFromFrameBtn")
        self.verticalLayout.addWidget(self.TrainFromFrameBtn)
        self.KnownFaceListBtn = QtWidgets.QPushButton(self.FaceRecogFrame)
        self.KnownFaceListBtn.setGeometry(QtCore.QRect(14, 230, 161, 23))
        self.KnownFaceListBtn.setObjectName("KnownFaceListBtn")
        self.LeftPanel.addWidget(self.FaceRecogFrame)
        self.ObjDetFrame = QtWidgets.QFrame(self.layoutWidget1)
        self.ObjDetFrame.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.ObjDetFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.ObjDetFrame.setObjectName("ObjDetFrame")
        self.ObjDetLabel = QtWidgets.QLabel(self.ObjDetFrame)
        self.ObjDetLabel.setGeometry(QtCore.QRect(10, 0, 91, 31))
        self.ObjDetLabel.setObjectName("ObjDetLabel")
        self.layoutWidget_4 = QtWidgets.QWidget(self.ObjDetFrame)
        self.layoutWidget_4.setGeometry(QtCore.QRect(10, 39, 171, 35))
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.layoutWidget_4)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.OdEnableLabel = QtWidgets.QLabel(self.layoutWidget_4)
        self.OdEnableLabel.setObjectName("OdEnableLabel")
        self.horizontalLayout_7.addWidget(self.OdEnableLabel)
        self.OdEnableCB = QtWidgets.QCheckBox(self.layoutWidget_4)
        self.OdEnableCB.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.OdEnableCB.setText("")
        self.OdEnableCB.setObjectName("OdEnableCB")
        self.horizontalLayout_7.addWidget(self.OdEnableCB)
        self.line_3 = QtWidgets.QFrame(self.ObjDetFrame)
        self.line_3.setGeometry(QtCore.QRect(0, 20, 191, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.LeftPanel.addWidget(self.ObjDetFrame)
        self.StatusGroup = QtWidgets.QGroupBox(self.centralwidget)
        self.StatusGroup.setGeometry(QtCore.QRect(10, 560, 861, 191))
        self.StatusGroup.setObjectName("StatusGroup")
        self.StatusText = QtWidgets.QTextBrowser(self.StatusGroup)
        self.StatusText.setGeometry(QtCore.QRect(15, 20, 831, 141))
        self.StatusText.setStyleSheet("QTextBrowser {\n""    background: rgb(29, 29, 29);\n""    color: rgb(255, 255, 255);\n""}")
        self.StatusText.setObjectName("StatusText")
        self.widget = QtWidgets.QWidget(self.StatusGroup)
        self.widget.setGeometry(QtCore.QRect(20, 165, 344, 22))
        self.widget.setObjectName("widget")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.DlibGPUacceleration = QtWidgets.QLabel(self.widget)
        self.DlibGPUacceleration.setObjectName("DlibGPUacceleration")
        self.horizontalLayout_8.addWidget(self.DlibGPUacceleration)
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_8.addItem(spacerItem1)
        self.OpenCVGPUacceleration = QtWidgets.QLabel(self.widget)
        self.OpenCVGPUacceleration.setObjectName("OpenCVGPUacceleration")
        self.horizontalLayout_8.addWidget(self.OpenCVGPUacceleration)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 882, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")
        self.menuFace_Recognition = QtWidgets.QMenu(self.menuTools)
        self.menuFace_Recognition.setObjectName("menuFace_Recognition")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionTrain_From_File = QtWidgets.QAction(MainWindow)
        self.actionTrain_From_File.setObjectName("actionTrain_From_File")
        self.actionTrain_From_Frame = QtWidgets.QAction(MainWindow)
        self.actionTrain_From_Frame.setObjectName("actionTrain_From_Frame")
        self.actionKnown_Faces_List = QtWidgets.QAction(MainWindow)
        self.actionKnown_Faces_List.setObjectName("actionKnown_Faces_List")
        self.actionObject_Detection = QtWidgets.QAction(MainWindow)
        self.actionObject_Detection.setObjectName("actionObject_Detection")
        self.menuFace_Recognition.addAction(self.actionTrain_From_File)
        self.menuFace_Recognition.addAction(self.actionTrain_From_Frame)
        self.menuFace_Recognition.addSeparator()
        self.menuFace_Recognition.addAction(self.actionKnown_Faces_List)
        self.menuFace_Recognition.addSeparator()
        self.menuTools.addAction(self.menuFace_Recognition.menuAction())
        self.menuTools.addAction(self.actionObject_Detection)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        app.aboutToQuit.connect(self.closeEvent)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "ISS"))
        self.PreviewFrameCB.setText(_translate("MainWindow", "Preview"))
        self.PreviewFrameCB.setChecked(True)
        self.StartCaptureBtn.setText(_translate("MainWindow", "Start Capture"))
        self.StopCaptureBtn.setText(_translate("MainWindow", "Stop Capture"))
        self.SelectInputButton.setText(_translate("MainWindow", "Select Input Source"))
        self.FaceRecogLabel.setText(_translate("MainWindow", "Face Recognition"))
        self.FrEnableLabel.setText(_translate("MainWindow", "Enable"))
        self.ToleranceLabel.setText(_translate("MainWindow", f'Tolerance({self.ToleranceSlider.value() / 100})'))
        self.TrainFaceGroup.setTitle(_translate("MainWindow", "Train Face"))
        self.TrainFromFileBtn.setText(_translate("MainWindow", "Train From New Files"))
        self.ReTrainBtn.setText(_translate("MainWindow", "Re-Train from Existing Files"))
        self.TrainFromFrameBtn.setText(_translate("MainWindow", "Train From Frame"))
        self.KnownFaceListBtn.setText(_translate("MainWindow", "Known Face List"))
        self.ObjDetLabel.setText(_translate("MainWindow", "Object Detection"))
        self.OdEnableLabel.setText(_translate("MainWindow", "Enable"))
        self.StatusGroup.setTitle(_translate("MainWindow", "Status"))
        self.DlibGPUacceleration.setText(_translate("MainWindow", "DLIB GPU Acceleration: True"))
        self.OpenCVGPUacceleration.setText(_translate("MainWindow", "OPENCV GPU Acceleration: True"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuTools.setTitle(_translate("MainWindow", "Tools"))
        self.menuFace_Recognition.setTitle(_translate("MainWindow", "Face Recognition"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionTrain_From_File.setText(_translate("MainWindow", "Train From File"))
        self.actionTrain_From_Frame.setText(_translate("MainWindow", "Train From Frame"))
        self.actionKnown_Faces_List.setText(_translate("MainWindow", "Known Faces List"))
        self.actionObject_Detection.setText(_translate("MainWindow", "Object Detection"))

        ######################## It starts here ###########################

        ##### Display Frame #####
        self.SelectInputButton.clicked.connect(self.callSelectInputSource) # <--------select input source
        self.StartCaptureBtn.clicked.connect(self.callStartCapture) # <--------start capture from input source
        self.StopCaptureBtn.clicked.connect(self.callStopCapture) # <----------stop capture

        ###### Face Recognition ######
        self.face_recog = faceRecognition.Face_recognition() # <--------initialize FR object
        self.face_recog.read_learnt_encodings() 
        self.TOLERANCE = self.ToleranceSlider.value() / 100  # <--------tolearance for FR       
        self.ToleranceSlider.valueChanged.connect(self.updateTolerance)  # <---------When the tolerance slider is changed
        
        self.ReTrainBtn.clicked.connect(lambda: threading.Thread(target = self.face_recog.learnFace).start())
        self.TrainFromFileBtn.clicked.connect(self.callTrainFaceFromFile)
        self.TrainFromFrameBtn.clicked.connect(self.callTrainFaceFromFrame)

        self.KnownFaceListBtn.clicked.connect(self.callKnownFaceList)

        ##### Status Frame ######
        if dlib.DLIB_USE_CUDA == True:
            self.DlibGPUacceleration.setText("Dlib GPU Acceleration: True")
        else:
            self.DlibGPUacceleration.setText("Dlib GPU Acceleration: False")
        
        if cv2.cuda.getCudaEnabledDeviceCount() != 0:
            self.OpenCVGPUacceleration.setText("OpenCV GPU Acceleration: True")
        else:
            self.OpenCVGPUacceleration.setText("OpenCV GPU Acceleration: False")

    def callSelectInputSource(self):
        SelectInputSource = QtWidgets.QDialog()
        SelectInputSourceDialog = Ui_SelectInputSource()
        SelectInputSourceDialog.setupUi(SelectInputSource)
        SelectInputSource.show()
        SelectInputSource.exec_()
        if SelectInputSourceDialog.choice == 0:
            self.capSource = SelectInputSourceDialog.webCamId
        elif SelectInputSourceDialog.choice == 1:
            self.capSource = SelectInputSourceDialog.ipCamAdd
        elif SelectInputSourceDialog.choice == 2:
            self.capSource = SelectInputSourceDialog.fileName

    def callStartCapture(self):
        if self.capSource == None:
            self.callSelectInputSource()
        else:
            print('Call Update Frame Thread')
            if self.updateFrameIsRunning == False:
                self.updateFrameThread = threading.Thread(target = self.updateFrame)
                self.updateFrameThread.start()

    def callStopCapture(self):
        self.DisplayLabel.clear()
        self.shouldCaptureVideo = False
        self.updateFrameThread.join()

    def callTrainFaceFromFile(self):
        TrainFaceFromFile = QtWidgets.QDialog()
        TrainFaceFromFileDialog = Ui_TrainFaceFromFile()
        TrainFaceFromFileDialog.setupUi(TrainFaceFromFile)
        TrainFaceFromFile.show()
        TrainFaceFromFile.exec_()
        self.face_recog.read_learnt_encodings()

    def callTrainFaceFromFrame(self):
        TrainFaceFromFrame = QtWidgets.QDialog()
        TrainFaceFromFrameDialog = Ui_TrainFaceFromFrame()
        TrainFaceFromFrameDialog.setupUi(TrainFaceFromFrame)
        TrainFaceFromFrame.show()
        TrainFaceFromFrame.exec_()        

    def callKnownFaceList(self):
        KnownFaces = QtWidgets.QDialog()
        KnownFacesDialog = Ui_KnownFaces(self.face_recog.known_names)
        KnownFacesDialog.setupUi(KnownFaces)
        KnownFaces.show()
        KnownFaces.exec_()

    def draw_box(self, frame, LABEL, TOP_LEFT, BOTTOM_RIGHT, COLOR, FONT, FONT_SIZE, FONT_THICKNESS, FRAME_THICKNESS):
        i = 0
        while i < len(TOP_LEFT):
            cv2.rectangle(frame, TOP_LEFT[i], BOTTOM_RIGHT[i], COLOR, FRAME_THICKNESS)
            cv2.putText(frame, LABEL[i], TOP_LEFT[i] , FONT , FONT_SIZE , (255,255,255), FONT_THICKNESS)
            i = i + 1

    def updateTolerance(self):
        self.TOLERANCE = self.ToleranceSlider.value() / 100 
        self.ToleranceLabel.setText(f'Tolerance({self.TOLERANCE})')

    def updateFrame(self):
        self.updateFrameIsRunning = True
        self.shouldCaptureVideo = True
        self.cap = cv2.VideoCapture(self.capSource)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while self.cap.isOpened():
            self.capSuccess, self.capFrame = self.cap.read()
            if self.capSuccess:
                self.capFrame = cv2.cvtColor(self.capFrame,cv2.COLOR_BGR2RGB)

                if self.FrEnableCB.isChecked() == True:
                    self.face_recog.TOP_LEFT.clear()
                    self.face_recog.BOTTOM_RIGHT.clear()
                    self.face_recog.LABEL.clear()

                    self.face_recog.recognise_faces(self.capFrame, self.TOLERANCE)
                    self.draw_box(self.capFrame, self.face_recog.LABEL, self.face_recog.TOP_LEFT, self.face_recog.BOTTOM_RIGHT, (0,255,0), cv2.FONT_HERSHEY_SIMPLEX, 0.5,2,2)

                if self.PreviewFrameCB.isChecked():
                    qImage = QtGui.QImage(self.capFrame, self.capFrame.shape[1], self.capFrame.shape[0], self.capFrame.strides[0], QtGui.QImage.Format_RGB888)
                    self.DisplayLabel.setPixmap(QtGui.QPixmap.fromImage(qImage))
                else:
                    self.DisplayLabel.clear()
                    
                if self.shouldCaptureVideo == False:
                    self.updateFrameIsRunning = False
                    break
                    
            else:
                print('Capture read failed...')
                self.updateFrameIsRunning = False
                break

    def closeEvent(self):
        self.cap.release()
        self.updateFrameThread.join()

class Ui_SelectInputSource(Ui_MainWindow):
    def __init__(self):
        self.choice = None
        self.webCamId = None
        self.ipCamAdd = None
        self.fileName = None

        self.webCamCount = self.searchWebCam()

    def setupUi(self, SelectInputSource):
        SelectInputSource.setObjectName("SelectInputSource")
        SelectInputSource.resize(402, 361)
        SelectInputSource.setMinimumSize(QtCore.QSize(402, 361))
        SelectInputSource.setMaximumSize(QtCore.QSize(402, 361))
        self.WebCamGroup = QtWidgets.QGroupBox(SelectInputSource)
        self.WebCamGroup.setGeometry(QtCore.QRect(10, 10, 381, 101))
        self.WebCamGroup.setObjectName("WebCamGroup")
        self.label = QtWidgets.QLabel(self.WebCamGroup)
        self.label.setGeometry(QtCore.QRect(20, 30, 91, 20))
        self.label.setObjectName("label")
        self.WebCamComboBox = QtWidgets.QComboBox(self.WebCamGroup)
        self.WebCamComboBox.setGeometry(QtCore.QRect(130, 30, 231, 22))
        self.WebCamComboBox.setObjectName("WebCamComboBox")
        self.UseWebCamBtn = QtWidgets.QPushButton(self.WebCamGroup)
        self.UseWebCamBtn.setGeometry(QtCore.QRect(280, 60, 81, 23))
        self.UseWebCamBtn.setObjectName("UseWebCamBtn")
        self.IpCamGroup = QtWidgets.QGroupBox(SelectInputSource)
        self.IpCamGroup.setGeometry(QtCore.QRect(10, 120, 381, 111))
        self.IpCamGroup.setObjectName("IpCamGroup")
        self.label_2 = QtWidgets.QLabel(self.IpCamGroup)
        self.label_2.setGeometry(QtCore.QRect(20, 34, 91, 21))
        self.label_2.setObjectName("label_2")
        self.UseIPCamBtn = QtWidgets.QPushButton(self.IpCamGroup)
        self.UseIPCamBtn.setGeometry(QtCore.QRect(280, 70, 81, 23))
        self.UseIPCamBtn.setObjectName("UseIPCamBtn")
        self.IPAddTextEdit = QtWidgets.QTextEdit(self.IpCamGroup)
        self.IPAddTextEdit.setGeometry(QtCore.QRect(130, 30, 231, 31))
        self.IPAddTextEdit.setObjectName("IPAddTextEdit")
        self.FileGroup = QtWidgets.QGroupBox(SelectInputSource)
        self.FileGroup.setGeometry(QtCore.QRect(10, 240, 381, 111))
        self.FileGroup.setObjectName("FileGroup")
        self.label_3 = QtWidgets.QLabel(self.FileGroup)
        self.label_3.setGeometry(QtCore.QRect(20, 35, 91, 21))
        self.label_3.setObjectName("label_3")
        self.UseFileBtn = QtWidgets.QPushButton(self.FileGroup)
        self.UseFileBtn.setGeometry(QtCore.QRect(280, 70, 81, 23))
        self.UseFileBtn.setObjectName("UseFileBtn")
        self.FileNameTextBrowser = QtWidgets.QTextBrowser(self.FileGroup)
        self.FileNameTextBrowser.setGeometry(QtCore.QRect(130, 30, 231, 31))
        self.FileNameTextBrowser.setObjectName("FileNameTextBrowser")
        self.OpenFileBtn = QtWidgets.QPushButton(self.FileGroup)
        self.OpenFileBtn.setGeometry(QtCore.QRect(200, 70, 75, 23))
        self.OpenFileBtn.setObjectName("OpenFileBtn")

        self.retranslateUi(SelectInputSource)
        QtCore.QMetaObject.connectSlotsByName(SelectInputSource)

    def retranslateUi(self, SelectInputSource):
        _translate = QtCore.QCoreApplication.translate
        SelectInputSource.setWindowTitle(_translate("SelectInputSource", "Select Input Source"))
        self.WebCamGroup.setTitle(_translate("SelectInputSource", "WebCam"))
        self.label.setText(_translate("SelectInputSource", "Select Web Cam:"))
        self.UseWebCamBtn.setText(_translate("SelectInputSource", "Use WebCam"))
        self.IpCamGroup.setTitle(_translate("SelectInputSource", "IP Cam"))
        self.label_2.setText(_translate("SelectInputSource", "Enter IP address:"))
        self.UseIPCamBtn.setText(_translate("SelectInputSource", "Use IP Cam"))
        self.FileGroup.setTitle(_translate("SelectInputSource", "File"))
        self.label_3.setText(_translate("SelectInputSource", "File Name:"))
        self.UseFileBtn.setText(_translate("SelectInputSource", "Use File"))
        self.OpenFileBtn.setText(_translate("SelectInputSource", "Open"))

    ###########################################################################
        for _ in range(self.webCamCount):
            self.WebCamComboBox.addItem(f'Webcam {self.webCamCount}')
        
        self.WebCamComboBox.activated.connect(self.getWebCamId)

        self.UseWebCamBtn.clicked.connect(self.selectWebCam)
        self.UseIPCamBtn.clicked.connect(self.selectIPCam)
        self.OpenFileBtn.clicked.connect(self.openFile)
        self.UseFileBtn.clicked.connect(self.selectFile)
    
    def selectWebCam(self):
        self.choice = 0

    def selectIPCam(self):
        self.choice = 1
        self.ipCamAdd = self.IPAddTextEdit.toPlainText()
        if self.ipCamAdd == '':
            self.ipCamAdd = None

    def openFile(self):
        fileName,_ = QtWidgets.QFileDialog.getOpenFileName(None, "Open File", '', 'Video and Image files (*.jpg *.png *.jpeg *.mp4 *.avi)')
        self.FileNameTextBrowser.setText(fileName)
        self.fileName = fileName

    def selectFile(self):
        self.choice = 2

    def getWebCamId(self):
        self.webCamId = self.WebCamComboBox.currentIndex()

    def searchWebCam(self):
        count = 0
        while True:
            cap = cv2.VideoCapture(count)
            capSuccess,_ = cap.read()
            if capSuccess:
                count = count + 1
            else:
                break
        cap.release()
        return count

class Ui_TrainFaceFromFile(faceRecognition.Face_recognition):

    def __init__(self):
        super().__init__()

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(452, 296)
        Dialog.setMinimumSize(QtCore.QSize(452, 296))
        Dialog.setMaximumSize(QtCore.QSize(452, 296))
        self.layoutWidget = QtWidgets.QWidget(Dialog)
        self.layoutWidget.setGeometry(QtCore.QRect(26, 19, 401, 31))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.nameLabel = QtWidgets.QLabel(self.layoutWidget)
        self.nameLabel.setObjectName("nameLabel")
        self.horizontalLayout.addWidget(self.nameLabel)
        spacerItem = QtWidgets.QSpacerItem(28, 68, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.nameInput = QtWidgets.QTextEdit(self.layoutWidget)
        self.nameInput.setObjectName("nameInput")
        self.horizontalLayout.addWidget(self.nameInput)
        self.layoutWidget1 = QtWidgets.QWidget(Dialog)
        self.layoutWidget1.setGeometry(QtCore.QRect(23, 254, 411, 31))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem1 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.saveCopyCB = QtWidgets.QCheckBox(self.layoutWidget1)
        self.saveCopyCB.setObjectName("saveCopyCB")
        self.horizontalLayout_2.addWidget(self.saveCopyCB)
        self.trainFaceBtn = QtWidgets.QPushButton(self.layoutWidget1)
        self.trainFaceBtn.setObjectName("trainFaceBtn")
        self.horizontalLayout_2.addWidget(self.trainFaceBtn)
        self.cancelBtn = QtWidgets.QPushButton(self.layoutWidget1)
        self.cancelBtn.setObjectName("cancelBtn")
        self.horizontalLayout_2.addWidget(self.cancelBtn)
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(20, 60, 411, 151))
        self.groupBox.setObjectName("groupBox")
        self.fileListText = QtWidgets.QTextBrowser(self.groupBox)
        self.fileListText.setGeometry(QtCore.QRect(10, 20, 391, 121))
        self.fileListText.setObjectName("fileListText")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(22, 220, 411, 25))
        self.widget.setObjectName("widget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.openFileBtn = QtWidgets.QPushButton(self.widget)
        self.openFileBtn.setObjectName("openFileBtn")
        self.horizontalLayout_3.addWidget(self.openFileBtn)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Train From File"))
        self.nameLabel.setText(_translate("Dialog", "Name of the person: "))
        self.saveCopyCB.setText(_translate("Dialog", "Save a copy"))
        self.trainFaceBtn.setText(_translate("Dialog", "Train"))
        self.cancelBtn.setText(_translate("Dialog", "Cancel"))
        self.groupBox.setTitle(_translate("Dialog", "File List"))
        self.openFileBtn.setText(_translate("Dialog", "Open"))

        # ----------------------------------------------------------------------
        self.fileList = []
        self.personName = ''
        self.openFileBtn.clicked.connect(self.openFile)
        self.trainFaceBtn.clicked.connect(self.trainFace)

    def openFile(self):
        
        fileNames,_ = QtWidgets.QFileDialog.getOpenFileNames(None, "Open File", '', 'Image files (*.jpg *.png *.jpeg)')
        
        count = 1
        for fileName in fileNames:
            if fileName != '':
                self.fileListText.setText(self.fileListText.toPlainText() + str(count) + '. ' + fileName + '\n')
                self.fileList.append(fileName)
            count = count + 1
        if len(self.fileList) == 0:
            self.fileListText.clear()
        self.personName = self.nameInput.toPlainText()

    def trainFace(self):

        if len(self.fileList) != 0:
            if self.personName != '':
                for fileName in self.fileList:
                    image = face_recognition.load_image_file(fileName)
                    encoding = face_recognition.face_encodings(image)[0]
                    self.known_faces.append(encoding)
                    self.known_names.append(self.personName)
                print('Saving learnt faces... \n')

                filename = 'faceEncodings'
                outfile =  open(f'{self.KNOWN_FACES_DIR}/{filename}', 'wb')
                pickle.dump(self.known_faces, outfile)
                outfile.close()
                print(f'Encoded faces stored at: {self.KNOWN_FACES_DIR}/{filename}')
                
                filename = 'faceNames'
                outfile =  open(f'{self.KNOWN_FACES_DIR}/{filename}', 'wb')
                pickle.dump(self.known_names, outfile)
                outfile.close()
                print(f'Face names stored at: {self.KNOWN_FACES_DIR}/{filename}')

                print('Number of faces learnt: ' + str(len(self.known_names)))

                if self.saveCopyCB.isChecked() == True:
                    print('save a copy')
                    os.mkdir(f'{self.KNOWN_FACES_IMAGE_DIR}/{self.personName}')
                    try:
                        for fileName in self.fileList:
                            shutil.copy(fileName, f'{self.KNOWN_FACES_IMAGE_DIR}/{self.personName}/')
                    except IOError as e:
                        print("Unable to copy file. %s" % e)
                    except:
                        print("Unexpected error:", sys.exc_info())

            else:
                print('Person Name should not be empty')
        else:
            print('no file selected')

class Ui_TrainFaceFromFrame(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(340, 330)
        self.UnknownFaceDisplayLabel = QtWidgets.QLabel(Dialog)
        self.UnknownFaceDisplayLabel.setGeometry(QtCore.QRect(10, 10, 320, 240))
        self.UnknownFaceDisplayLabel.setStyleSheet("QLabel {\n""    background: rgb(29, 29, 29)\n""}")
        self.UnknownFaceDisplayLabel.setText("")
        self.UnknownFaceDisplayLabel.setObjectName("UnknownFaceDisplayLabel")
        self.AddFaceBtn = QtWidgets.QPushButton(Dialog)
        self.AddFaceBtn.setGeometry(QtCore.QRect(240, 300, 75, 23))
        self.AddFaceBtn.setObjectName("AddFaceBtn")
        self.SaveCopyCB = QtWidgets.QCheckBox(Dialog)
        self.SaveCopyCB.setGeometry(QtCore.QRect(150, 300, 81, 20))
        self.SaveCopyCB.setObjectName("SaveCopyCB")
        self.widget = QtWidgets.QWidget(Dialog)
        self.widget.setGeometry(QtCore.QRect(20, 270, 295, 21))
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.NameLabel = QtWidgets.QLabel(self.widget)
        self.NameLabel.setObjectName("NameLabel")
        self.horizontalLayout.addWidget(self.NameLabel)
        self.NameTextEdit = QtWidgets.QTextEdit(self.widget)
        self.NameTextEdit.setAcceptDrops(True)
        self.NameTextEdit.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.NameTextEdit.setObjectName("NameTextEdit")
        self.horizontalLayout.addWidget(self.NameTextEdit)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Train From Frame"))
        self.AddFaceBtn.setText(_translate("Dialog", "Add Face"))
        self.SaveCopyCB.setText(_translate("Dialog", "Save a copy"))
        self.NameLabel.setText(_translate("Dialog", "Name:"))

class Ui_KnownFaces():
    def __init__(self, knownNames):
        self.knownNames = knownNames

    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(297, 417)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setMinimumSize(QtCore.QSize(0, 13))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setStyleSheet("QLabel {\n""    background: rgb(212, 242, 255)\n""}")
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.listWidget = QtWidgets.QListWidget(Dialog)
        self.listWidget.setObjectName("listWidget")
        self.verticalLayout.addWidget(self.listWidget)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Known Faces"))
        self.label.setText(_translate("Dialog", "Known Faces List"))
        self.listWidget.setSortingEnabled(True)

        self.uniqueKnownNames = []
        for name in self.knownNames:
            if name not in self.uniqueKnownNames:
                self.uniqueKnownNames.append(name)

        for count,name in enumerate(self.uniqueKnownNames):        
            self.listWidget.addItem(f'{count+1}. {name}')
      

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
