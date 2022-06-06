# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'downloaddialog.ui'
#
# Created: Sun Jul 23 23:56:29 2006
#      by: PyQt4 UI code generator 4.0-snapshot-20060619
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt4 import QtCore, QtGui

class Ui_DownloadDialog(object):
    def setupUi(self, DownloadDialog):
        DownloadDialog.setObjectName("DownloadDialog")
        DownloadDialog.resize(QtCore.QSize(QtCore.QRect(0,0,512,320).size()).expandedTo(DownloadDialog.minimumSizeHint()))

        self.vboxlayout = QtGui.QVBoxLayout(DownloadDialog)
        self.vboxlayout.setMargin(9)
        self.vboxlayout.setSpacing(6)
        self.vboxlayout.setObjectName("vboxlayout")

        self.treeWidget = QtGui.QTreeWidget(DownloadDialog)
        self.treeWidget.setObjectName("treeWidget")
        self.vboxlayout.addWidget(self.treeWidget)

        self.progressBar = QtGui.QProgressBar(DownloadDialog)
        self.progressBar.setProperty("value",QtCore.QVariant(0))
        self.progressBar.setOrientation(QtCore.Qt.Horizontal)
        self.progressBar.setObjectName("progressBar")
        self.vboxlayout.addWidget(self.progressBar)

        self.hboxlayout = QtGui.QHBoxLayout()
        self.hboxlayout.setMargin(0)
        self.hboxlayout.setSpacing(6)
        self.hboxlayout.setObjectName("hboxlayout")

        spacerItem = QtGui.QSpacerItem(131,31,QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Minimum)
        self.hboxlayout.addItem(spacerItem)

        self.openDirButton = QtGui.QPushButton(DownloadDialog)
        self.openDirButton.setObjectName("openDirButton")
        self.hboxlayout.addWidget(self.openDirButton)

        self.stopButton = QtGui.QPushButton(DownloadDialog)
        self.stopButton.setObjectName("stopButton")
        self.hboxlayout.addWidget(self.stopButton)

        self.closeButton = QtGui.QPushButton(DownloadDialog)
        self.closeButton.setEnabled(False)
        self.closeButton.setObjectName("closeButton")
        self.hboxlayout.addWidget(self.closeButton)
        self.vboxlayout.addLayout(self.hboxlayout)

        self.retranslateUi(DownloadDialog)
        QtCore.QObject.connect(self.closeButton,QtCore.SIGNAL("clicked()"),DownloadDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(DownloadDialog)

    def tr(self, string):
        return QtGui.QApplication.translate("DownloadDialog", string, None, QtGui.QApplication.UnicodeUTF8)

    def retranslateUi(self, DownloadDialog):
        DownloadDialog.setWindowTitle(self.tr("Download Packages"))
        self.openDirButton.setText(self.tr("&Open Directory"))
        self.stopButton.setText(self.tr("&Stop"))
        self.stopButton.setShortcut(self.tr("Esc"))
        self.closeButton.setText(self.tr("&Close"))
