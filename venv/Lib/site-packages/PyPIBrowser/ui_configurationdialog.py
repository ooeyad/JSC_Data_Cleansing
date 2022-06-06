# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'configurationdialog.ui'
#
# Created: Sun Jul 30 17:13:28 2006
#      by: PyQt4 UI code generator 4.0.1
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt4 import QtCore, QtGui

class Ui_ConfigurationDialog(object):
    def setupUi(self, ConfigurationDialog):
        ConfigurationDialog.setObjectName("ConfigurationDialog")
        ConfigurationDialog.resize(QtCore.QSize(QtCore.QRect(0,0,291,323).size()).expandedTo(ConfigurationDialog.minimumSizeHint()))

        self.vboxlayout = QtGui.QVBoxLayout(ConfigurationDialog)
        self.vboxlayout.setMargin(9)
        self.vboxlayout.setSpacing(6)
        self.vboxlayout.setObjectName("vboxlayout")

        self.tabWidget = QtGui.QTabWidget(ConfigurationDialog)
        self.tabWidget.setObjectName("tabWidget")

        self.browserTab = QtGui.QWidget()
        self.browserTab.setObjectName("browserTab")

        self.vboxlayout1 = QtGui.QVBoxLayout(self.browserTab)
        self.vboxlayout1.setMargin(9)
        self.vboxlayout1.setSpacing(6)
        self.vboxlayout1.setObjectName("vboxlayout1")

        self.gridlayout = QtGui.QGridLayout()
        self.gridlayout.setMargin(0)
        self.gridlayout.setSpacing(6)
        self.gridlayout.setObjectName("gridlayout")

        self.packageIndexLineEdit = QtGui.QLineEdit(self.browserTab)
        self.packageIndexLineEdit.setReadOnly(False)
        self.packageIndexLineEdit.setObjectName("packageIndexLineEdit")
        self.gridlayout.addWidget(self.packageIndexLineEdit,0,1,1,2)

        self.downloadLabel = QtGui.QLabel(self.browserTab)
        self.downloadLabel.setObjectName("downloadLabel")
        self.gridlayout.addWidget(self.downloadLabel,1,0,1,1)

        self.downloadLineEdit = QtGui.QLineEdit(self.browserTab)
        self.downloadLineEdit.setReadOnly(True)
        self.downloadLineEdit.setObjectName("downloadLineEdit")
        self.gridlayout.addWidget(self.downloadLineEdit,1,1,1,1)

        self.packageIndexLabel = QtGui.QLabel(self.browserTab)
        self.packageIndexLabel.setObjectName("packageIndexLabel")
        self.gridlayout.addWidget(self.packageIndexLabel,0,0,1,1)

        self.downloadButton = QtGui.QToolButton(self.browserTab)
        self.downloadButton.setObjectName("downloadButton")
        self.gridlayout.addWidget(self.downloadButton,1,2,1,1)
        self.vboxlayout1.addLayout(self.gridlayout)

        self.label = QtGui.QLabel(self.browserTab)
        self.label.setObjectName("label")
        self.vboxlayout1.addWidget(self.label)

        self.hboxlayout = QtGui.QHBoxLayout()
        self.hboxlayout.setMargin(0)
        self.hboxlayout.setSpacing(6)
        self.hboxlayout.setObjectName("hboxlayout")

        self.preferencesList = QtGui.QListWidget(self.browserTab)

        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Policy(7),QtGui.QSizePolicy.Policy(7))
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.preferencesList.sizePolicy().hasHeightForWidth())
        self.preferencesList.setSizePolicy(sizePolicy)
        self.preferencesList.setObjectName("preferencesList")
        self.hboxlayout.addWidget(self.preferencesList)

        self.vboxlayout2 = QtGui.QVBoxLayout()
        self.vboxlayout2.setMargin(0)
        self.vboxlayout2.setSpacing(6)
        self.vboxlayout2.setObjectName("vboxlayout2")

        self.upButton = QtGui.QPushButton(self.browserTab)
        self.upButton.setEnabled(False)
        self.upButton.setObjectName("upButton")
        self.vboxlayout2.addWidget(self.upButton)

        self.downButton = QtGui.QPushButton(self.browserTab)
        self.downButton.setEnabled(False)
        self.downButton.setObjectName("downButton")
        self.vboxlayout2.addWidget(self.downButton)

        self.hideButton = QtGui.QPushButton(self.browserTab)
        self.hideButton.setEnabled(False)
        self.hideButton.setObjectName("hideButton")
        self.vboxlayout2.addWidget(self.hideButton)

        spacerItem = QtGui.QSpacerItem(20,40,QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Expanding)
        self.vboxlayout2.addItem(spacerItem)
        self.hboxlayout.addLayout(self.vboxlayout2)
        self.vboxlayout1.addLayout(self.hboxlayout)
        self.tabWidget.addTab(self.browserTab, "")

        self.pythonTab = QtGui.QWidget()
        self.pythonTab.setObjectName("pythonTab")

        self.gridlayout1 = QtGui.QGridLayout(self.pythonTab)
        self.gridlayout1.setMargin(9)
        self.gridlayout1.setSpacing(6)
        self.gridlayout1.setObjectName("gridlayout1")

        self.systemPathsList = QtGui.QListWidget(self.pythonTab)
        self.systemPathsList.setObjectName("systemPathsList")
        self.gridlayout1.addWidget(self.systemPathsList,3,0,1,2)

        self.platformLabel = QtGui.QLabel(self.pythonTab)
        self.platformLabel.setObjectName("platformLabel")
        self.gridlayout1.addWidget(self.platformLabel,1,0,1,1)

        self.platformPlaceholder = QtGui.QLabel(self.pythonTab)
        self.platformPlaceholder.setObjectName("platformPlaceholder")
        self.gridlayout1.addWidget(self.platformPlaceholder,1,1,1,1)

        self.sysPathsLabel = QtGui.QLabel(self.pythonTab)
        self.sysPathsLabel.setObjectName("sysPathsLabel")
        self.gridlayout1.addWidget(self.sysPathsLabel,2,0,1,1)

        spacerItem1 = QtGui.QSpacerItem(20,40,QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Expanding)
        self.gridlayout1.addItem(spacerItem1,4,0,1,1)

        self.versionPlaceholder = QtGui.QLabel(self.pythonTab)
        self.versionPlaceholder.setObjectName("versionPlaceholder")
        self.gridlayout1.addWidget(self.versionPlaceholder,0,1,1,1)

        self.versionLabel = QtGui.QLabel(self.pythonTab)
        self.versionLabel.setObjectName("versionLabel")
        self.gridlayout1.addWidget(self.versionLabel,0,0,1,1)
        self.tabWidget.addTab(self.pythonTab, "")
        self.vboxlayout.addWidget(self.tabWidget)

        spacerItem2 = QtGui.QSpacerItem(273,16,QtGui.QSizePolicy.Minimum,QtGui.QSizePolicy.Expanding)
        self.vboxlayout.addItem(spacerItem2)

        self.hboxlayout1 = QtGui.QHBoxLayout()
        self.hboxlayout1.setMargin(0)
        self.hboxlayout1.setSpacing(6)
        self.hboxlayout1.setObjectName("hboxlayout1")

        spacerItem3 = QtGui.QSpacerItem(131,31,QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Minimum)
        self.hboxlayout1.addItem(spacerItem3)

        self.okButton = QtGui.QPushButton(ConfigurationDialog)
        self.okButton.setObjectName("okButton")
        self.hboxlayout1.addWidget(self.okButton)

        self.cancelButton = QtGui.QPushButton(ConfigurationDialog)
        self.cancelButton.setObjectName("cancelButton")
        self.hboxlayout1.addWidget(self.cancelButton)
        self.vboxlayout.addLayout(self.hboxlayout1)
        self.downloadLabel.setBuddy(self.downloadLineEdit)
        self.packageIndexLabel.setBuddy(self.packageIndexLineEdit)
        self.label.setBuddy(self.preferencesList)

        self.retranslateUi(ConfigurationDialog)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QObject.connect(self.okButton,QtCore.SIGNAL("clicked()"),ConfigurationDialog.accept)
        QtCore.QObject.connect(self.cancelButton,QtCore.SIGNAL("clicked()"),ConfigurationDialog.reject)
        QtCore.QMetaObject.connectSlotsByName(ConfigurationDialog)

    def retranslateUi(self, ConfigurationDialog):
        ConfigurationDialog.setWindowTitle(QtGui.QApplication.translate("ConfigurationDialog", "Configure Browser", None, QtGui.QApplication.UnicodeUTF8))
        self.downloadLabel.setText(QtGui.QApplication.translate("ConfigurationDialog", "Download di&rectory:", None, QtGui.QApplication.UnicodeUTF8))
        self.packageIndexLabel.setText(QtGui.QApplication.translate("ConfigurationDialog", "Package &Index:", None, QtGui.QApplication.UnicodeUTF8))
        self.downloadButton.setText(QtGui.QApplication.translate("ConfigurationDialog", "...", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("ConfigurationDialog", "&Package preferences:", None, QtGui.QApplication.UnicodeUTF8))
        self.upButton.setText(QtGui.QApplication.translate("ConfigurationDialog", "Move &Up", None, QtGui.QApplication.UnicodeUTF8))
        self.downButton.setText(QtGui.QApplication.translate("ConfigurationDialog", "Move &Down", None, QtGui.QApplication.UnicodeUTF8))
        self.hideButton.setText(QtGui.QApplication.translate("ConfigurationDialog", "&Hide", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.browserTab), QtGui.QApplication.translate("ConfigurationDialog", "&Browser", None, QtGui.QApplication.UnicodeUTF8))
        self.platformLabel.setText(QtGui.QApplication.translate("ConfigurationDialog", "Platform:", None, QtGui.QApplication.UnicodeUTF8))
        self.sysPathsLabel.setText(QtGui.QApplication.translate("ConfigurationDialog", "System paths:", None, QtGui.QApplication.UnicodeUTF8))
        self.versionLabel.setText(QtGui.QApplication.translate("ConfigurationDialog", "Interpreter version:", None, QtGui.QApplication.UnicodeUTF8))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.pythonTab), QtGui.QApplication.translate("ConfigurationDialog", "&Python", None, QtGui.QApplication.UnicodeUTF8))
        self.okButton.setText(QtGui.QApplication.translate("ConfigurationDialog", "&OK", None, QtGui.QApplication.UnicodeUTF8))
        self.cancelButton.setText(QtGui.QApplication.translate("ConfigurationDialog", "&Cancel", None, QtGui.QApplication.UnicodeUTF8))
