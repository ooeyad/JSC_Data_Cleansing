# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'informationwindow.ui'
#
# Created: Sun Jul 23 21:17:15 2006
#      by: PyQt4 UI code generator 4.0-snapshot-20060619
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt4 import QtCore, QtGui

class Ui_InformationWindow(object):
    def setupUi(self, InformationWindow):
        InformationWindow.setObjectName("InformationWindow")
        InformationWindow.resize(QtCore.QSize(QtCore.QRect(0,0,400,300).size()).expandedTo(InformationWindow.minimumSizeHint()))

        self.vboxlayout = QtGui.QVBoxLayout(InformationWindow)
        self.vboxlayout.setMargin(9)
        self.vboxlayout.setSpacing(6)
        self.vboxlayout.setObjectName("vboxlayout")

        self.textBrowser = QtGui.QTextBrowser(InformationWindow)
        self.textBrowser.setObjectName("textBrowser")
        self.vboxlayout.addWidget(self.textBrowser)

        self.hboxlayout = QtGui.QHBoxLayout()
        self.hboxlayout.setMargin(0)
        self.hboxlayout.setSpacing(6)
        self.hboxlayout.setObjectName("hboxlayout")

        spacerItem = QtGui.QSpacerItem(40,20,QtGui.QSizePolicy.Expanding,QtGui.QSizePolicy.Minimum)
        self.hboxlayout.addItem(spacerItem)

        self.closeButton = QtGui.QPushButton(InformationWindow)
        self.closeButton.setObjectName("closeButton")
        self.hboxlayout.addWidget(self.closeButton)
        self.vboxlayout.addLayout(self.hboxlayout)

        self.closeAction = QtGui.QAction(InformationWindow)
        self.closeAction.setObjectName("closeAction")

        self.retranslateUi(InformationWindow)
        QtCore.QObject.connect(self.closeButton,QtCore.SIGNAL("clicked()"),InformationWindow.close)
        QtCore.QMetaObject.connectSlotsByName(InformationWindow)

    def tr(self, string):
        return QtGui.QApplication.translate("InformationWindow", string, None, QtGui.QApplication.UnicodeUTF8)

    def retranslateUi(self, InformationWindow):
        InformationWindow.setWindowTitle(self.tr("Information"))
        self.closeButton.setText(self.tr("&Close"))
        self.closeAction.setText(self.tr("Close"))
        self.closeAction.setShortcut(self.tr("Esc"))
