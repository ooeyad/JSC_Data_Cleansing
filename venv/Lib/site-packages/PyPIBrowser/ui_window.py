# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'PyPIBrowser/window.ui'
#
# Created: Tue Sep 26 01:11:54 2006
#      by: PyQt4 UI code generator 4.0.1
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt4 import QtCore, QtGui

class Ui_Window(object):
    def setupUi(self, Window):
        Window.setObjectName("Window")
        Window.resize(QtCore.QSize(QtCore.QRect(0,0,543,443).size()).expandedTo(Window.minimumSizeHint()))

        self.centralwidget = QtGui.QWidget(Window)
        self.centralwidget.setObjectName("centralwidget")

        self.vboxlayout = QtGui.QVBoxLayout(self.centralwidget)
        self.vboxlayout.setMargin(9)
        self.vboxlayout.setSpacing(6)
        self.vboxlayout.setObjectName("vboxlayout")

        self.treeView = QtGui.QTreeView(self.centralwidget)
        self.treeView.setEnabled(False)
        self.treeView.setObjectName("treeView")
        self.vboxlayout.addWidget(self.treeView)

        self.hboxlayout = QtGui.QHBoxLayout()
        self.hboxlayout.setMargin(0)
        self.hboxlayout.setSpacing(6)
        self.hboxlayout.setObjectName("hboxlayout")

        self.label = QtGui.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.hboxlayout.addWidget(self.label)

        self.fieldComboBox = QtGui.QComboBox(self.centralwidget)
        self.fieldComboBox.setEnabled(False)
        self.fieldComboBox.setObjectName("fieldComboBox")
        self.hboxlayout.addWidget(self.fieldComboBox)

        self.label_2 = QtGui.QLabel(self.centralwidget)
        self.label_2.setObjectName("label_2")
        self.hboxlayout.addWidget(self.label_2)

        self.searchLineEdit = QtGui.QLineEdit(self.centralwidget)
        self.searchLineEdit.setEnabled(False)
        self.searchLineEdit.setObjectName("searchLineEdit")
        self.hboxlayout.addWidget(self.searchLineEdit)

        self.searchButton = QtGui.QPushButton(self.centralwidget)
        self.searchButton.setEnabled(False)
        self.searchButton.setFocusPolicy(QtCore.Qt.NoFocus)
        self.searchButton.setObjectName("searchButton")
        self.hboxlayout.addWidget(self.searchButton)
        self.vboxlayout.addLayout(self.hboxlayout)
        Window.setCentralWidget(self.centralwidget)

        self.menubar = QtGui.QMenuBar(Window)
        self.menubar.setGeometry(QtCore.QRect(0,0,543,27))
        self.menubar.setObjectName("menubar")

        self.menu_File = QtGui.QMenu(self.menubar)
        self.menu_File.setObjectName("menu_File")

        self.menu_Packages = QtGui.QMenu(self.menubar)
        self.menu_Packages.setObjectName("menu_Packages")

        self.menu_Help = QtGui.QMenu(self.menubar)
        self.menu_Help.setObjectName("menu_Help")

        self.menu_Settings = QtGui.QMenu(self.menubar)
        self.menu_Settings.setObjectName("menu_Settings")
        Window.setMenuBar(self.menubar)

        self.statusbar = QtGui.QStatusBar(Window)
        self.statusbar.setObjectName("statusbar")
        Window.setStatusBar(self.statusbar)

        self.openAction = QtGui.QAction(Window)
        self.openAction.setObjectName("openAction")

        self.exitAction = QtGui.QAction(Window)
        self.exitAction.setObjectName("exitAction")

        self.reloadListAction = QtGui.QAction(Window)
        self.reloadListAction.setEnabled(False)
        self.reloadListAction.setObjectName("reloadListAction")

        self.downloadAction = QtGui.QAction(Window)
        self.downloadAction.setEnabled(False)
        self.downloadAction.setObjectName("downloadAction")

        self.filterMarkedAction = QtGui.QAction(Window)
        self.filterMarkedAction.setCheckable(True)
        self.filterMarkedAction.setEnabled(False)
        self.filterMarkedAction.setObjectName("filterMarkedAction")

        self.configureBrowserAction = QtGui.QAction(Window)
        self.configureBrowserAction.setObjectName("configureBrowserAction")

        self.filterNewAction = QtGui.QAction(Window)
        self.filterNewAction.setCheckable(True)
        self.filterNewAction.setEnabled(False)
        self.filterNewAction.setObjectName("filterNewAction")

        self.aboutAction = QtGui.QAction(Window)
        self.aboutAction.setObjectName("aboutAction")

        self.aboutQtAction = QtGui.QAction(Window)
        self.aboutQtAction.setIcon(QtGui.QIcon("../../../../../../../:/trolltech/formeditor/images/qtlogo.png"))
        self.aboutQtAction.setObjectName("aboutQtAction")

        self.openManualAction = QtGui.QAction(Window)
        self.openManualAction.setObjectName("openManualAction")

        self.editShortcutsAction = QtGui.QAction(Window)
        self.editShortcutsAction.setObjectName("editShortcutsAction")
        self.menu_File.addAction(self.openAction)
        self.menu_File.addAction(self.exitAction)
        self.menu_Packages.addAction(self.reloadListAction)
        self.menu_Packages.addAction(self.downloadAction)
        self.menu_Packages.addAction(self.filterMarkedAction)
        self.menu_Packages.addAction(self.filterNewAction)
        self.menu_Help.addAction(self.aboutAction)
        self.menu_Help.addAction(self.aboutQtAction)
        self.menu_Help.addAction(self.openManualAction)
        self.menu_Settings.addAction(self.configureBrowserAction)
        self.menu_Settings.addAction(self.editShortcutsAction)
        self.menubar.addAction(self.menu_File.menuAction())
        self.menubar.addAction(self.menu_Packages.menuAction())
        self.menubar.addAction(self.menu_Settings.menuAction())
        self.menubar.addAction(self.menu_Help.menuAction())
        self.label.setBuddy(self.fieldComboBox)
        self.label_2.setBuddy(self.searchLineEdit)

        self.retranslateUi(Window)
        QtCore.QObject.connect(self.exitAction,QtCore.SIGNAL("triggered()"),Window.close)
        QtCore.QMetaObject.connectSlotsByName(Window)

    def retranslateUi(self, Window):
        Window.setWindowTitle(QtGui.QApplication.translate("Window", "PyPI Browser", None, QtGui.QApplication.UnicodeUTF8))
        self.label.setText(QtGui.QApplication.translate("Window", "&Field:", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Name", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Version", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Author", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Summary", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Description", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Stable version", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Author e-mail", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Maintainer", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Maintainer e-mail", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "License", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Platform", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Download URL", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Home page", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Keywords", None, QtGui.QApplication.UnicodeUTF8))
        self.fieldComboBox.addItem(QtGui.QApplication.translate("Window", "Classifiers", None, QtGui.QApplication.UnicodeUTF8))
        self.label_2.setText(QtGui.QApplication.translate("Window", "Sea&rch for:", None, QtGui.QApplication.UnicodeUTF8))
        self.searchButton.setText(QtGui.QApplication.translate("Window", "&Search", None, QtGui.QApplication.UnicodeUTF8))
        self.menu_File.setTitle(QtGui.QApplication.translate("Window", "&File", None, QtGui.QApplication.UnicodeUTF8))
        self.menu_Packages.setTitle(QtGui.QApplication.translate("Window", "&Packages", None, QtGui.QApplication.UnicodeUTF8))
        self.menu_Help.setTitle(QtGui.QApplication.translate("Window", "&Help", None, QtGui.QApplication.UnicodeUTF8))
        self.menu_Settings.setTitle(QtGui.QApplication.translate("Window", "&Settings", None, QtGui.QApplication.UnicodeUTF8))
        self.openAction.setText(QtGui.QApplication.translate("Window", "&Open...", None, QtGui.QApplication.UnicodeUTF8))
        self.openAction.setShortcut(QtGui.QApplication.translate("Window", "Ctrl+O", None, QtGui.QApplication.UnicodeUTF8))
        self.exitAction.setText(QtGui.QApplication.translate("Window", "E&xit", None, QtGui.QApplication.UnicodeUTF8))
        self.exitAction.setShortcut(QtGui.QApplication.translate("Window", "Ctrl+Q", None, QtGui.QApplication.UnicodeUTF8))
        self.reloadListAction.setText(QtGui.QApplication.translate("Window", "&Reload List", None, QtGui.QApplication.UnicodeUTF8))
        self.reloadListAction.setShortcut(QtGui.QApplication.translate("Window", "Ctrl+R", None, QtGui.QApplication.UnicodeUTF8))
        self.downloadAction.setText(QtGui.QApplication.translate("Window", "Download...", None, QtGui.QApplication.UnicodeUTF8))
        self.downloadAction.setShortcut(QtGui.QApplication.translate("Window", "Ctrl+Return", None, QtGui.QApplication.UnicodeUTF8))
        self.filterMarkedAction.setText(QtGui.QApplication.translate("Window", "Filter Marked", None, QtGui.QApplication.UnicodeUTF8))
        self.configureBrowserAction.setText(QtGui.QApplication.translate("Window", "Configure Browser...", None, QtGui.QApplication.UnicodeUTF8))
        self.filterNewAction.setText(QtGui.QApplication.translate("Window", "Filter New", None, QtGui.QApplication.UnicodeUTF8))
        self.aboutAction.setText(QtGui.QApplication.translate("Window", "&About...", None, QtGui.QApplication.UnicodeUTF8))
        self.aboutQtAction.setText(QtGui.QApplication.translate("Window", "About Qt...", None, QtGui.QApplication.UnicodeUTF8))
        self.openManualAction.setText(QtGui.QApplication.translate("Window", "Open Manual", None, QtGui.QApplication.UnicodeUTF8))
        self.editShortcutsAction.setText(QtGui.QApplication.translate("Window", "Edit Shortcuts...", None, QtGui.QApplication.UnicodeUTF8))
