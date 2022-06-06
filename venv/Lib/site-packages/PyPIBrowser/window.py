#!/usr/bin/env python

"""
window.py

Copyright (C) 2006 David Boddie

This file is part of PyPI Browser, a GUI browser for the Python Package Index.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
"""

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from constants import __version__
import desktop
from dialogs import ActionEditorDialog, ConfigurationDialog, DownloadDialog, \
                    InformationWindow
from packagemodel import PackageModel
import os
import pypi
from searchmodel import SearchModel
import sys
from ui_window import Ui_Window
import urllib2


class Window(QMainWindow, Ui_Window):

    """Window(QMainWindow, Ui_Window)
    
    A class to provide the main application window and contain the
    infrastructure used by components to communicate with each other.
    """
    
    def __init__(self, parent = None):
    
        QMainWindow.__init__(self, parent)
        self.setupUi(self)

        # Create a settings object that will be shared between
        # application components.
        self.settings = QSettings("boddie.org.uk", "PyPI Browser")
        
        # We use two models: an underlying package model and a search
        # model that filters packages based on whether they match search
        # terms and whether they are marked. The tree view shows the
        # contents of the search model.
        self.package_server = pypi.AbstractServer()
        self.packageModel = PackageModel(self.package_server)
        self.searchModel = SearchModel(self.package_server)
        self.searchModel.setSourceModel(self.packageModel)
        self.treeView.setModel(self.searchModel)
        
        self.markedPackages = 0
        self.matchingPackages = 0
        self.windows = []
        
        # Set up signal-slot connections defined in the .ui files and
        # those providing higher-level functionality.
        QMetaObject.connectSlotsByName(self)
        
        # Load user-defined actions.
        ActionEditorDialog.loadSettings(self.settings, self.findChildren(QAction))
        
        self.connect(self.openAction, SIGNAL("triggered()"), self.openIndex)
        self.connect(self.downloadAction, SIGNAL("triggered()"), self.download)
        self.connect(self.reloadListAction, SIGNAL("triggered()"),
                     self.reloadPackages)
        self.connect(self.filterMarkedAction, SIGNAL("triggered(bool)"),
                     self.setMarkedFilter)
        self.connect(self.filterNewAction, SIGNAL("triggered(bool)"),
                     self.setNewFilter)
        self.connect(self.searchModel, SIGNAL("resultsFound(const QString &)"),
                     self.showResults)
        self.connect(self.searchModel, SIGNAL("markedChanged(bool)"),
                     self.downloadAction, SLOT("setEnabled(bool)"))
        
        self.connect(self.packageModel, SIGNAL("operationStarted()"),
                     self.showWaitCursor)
        self.connect(self.packageModel, SIGNAL("operationFinished()"),
                     self.unsetCursor)
        self.connect(self.searchModel, SIGNAL("operationStarted()"),
                     self.showWaitCursor)
        self.connect(self.searchModel, SIGNAL("operationFinished()"),
                     self.unsetCursor)
        
        self.connect(self.treeView, SIGNAL("activated(const QModelIndex &)"),
                     self.showInformation)
        self.connect(self.treeView, SIGNAL("expanded(const QModelIndex &)"),
                     self.resizeColumns)
        #self.connect(self.treeView, SIGNAL("collapsed(const QModelIndex &)"),
        #             self.resizeColumns)
        
        self.connect(self.fieldComboBox,
            SIGNAL("currentIndexChanged(int)"),
            self.searchModel.setSearchField)
        self.connect(self.searchLineEdit, SIGNAL("textChanged(const QString &)"),
                     self.searchModel.setSearchTerms)
        self.connect(self.searchLineEdit, SIGNAL("returnPressed()"),
                     self.searchModel.search)
        self.connect(self.searchButton, SIGNAL("clicked()"),
                     self.searchModel.search)
        
        self.connect(self.configureBrowserAction, SIGNAL("triggered()"),
                     self.configureBrowser)
        self.connect(self.editShortcutsAction, SIGNAL("triggered()"),
                     self.editShortcuts)
        
        self.connect(self.aboutAction, SIGNAL("triggered()"), self.about)
        self.connect(self.aboutQtAction, SIGNAL("triggered()"), self.aboutQt)
        self.connect(self.openManualAction, SIGNAL("triggered()"),
                     self.openManual)
    
    def about(self):
    
        QMessageBox.about(self,
            self.tr("About PyPI Browser %1").arg(__version__),
            self.tr("<qt><h3>About PyPI Browser %1</h3>"
                    "<p>PyPI Browser allows you to examine available "
                    "packages in the Python Package Index and other package "
                    "indexes that expose a compatible XML-RPC interface.</p>"
                    "<p>Uses desktop integration features provided by version "
                    "%2 of the <i>desktop</i> module (search the package "
                    "index for more information).</p></qt>").arg(__version__)
                    .arg(desktop.__version__))
    
    def aboutQt(self):
    
        QMessageBox.aboutQt(self)
    
    def changeServer(self):
    
        """changeModel(self, newModel)
        
        Replace the existing package model and set up the search model
        to filter the contents of the new model.
        """
        
        self.packageModel.setServer(self.package_server)
        self.searchModel.setServer(self.package_server)
    
    def closeEvent(self, event):
    
        """closeEvent(self, event)
        
        Checks for marked packages and accepts the close event only if
        there are either no marked packages or if the user discards them.
        """
        
        gen = self.marked()
        try:
            gen.next()
            if not self.confirmDiscard():
                event.ignore()
                return
        except StopIteration:
            pass
        
        self.packageModel.save(self.settings)
        for widget in qApp.topLevelWidgets():
            widget.close()
    
    def configureBrowser(self):
    
        """configureBrowser(self)
        
        Opens a configuration dialog to allow the user to change the
        behaviour of the application.
        """
        
        dialog = ConfigurationDialog(self.settings, self)
        if dialog.exec_() == QDialog.Accepted:
            dialog.saveSettings()
            self.settings.sync()
    
    def confirmDiscard(self):
    
        """confirmDiscard(self)
        
        Opens a message dialog asking whether the user wants to discard
        the current list of marked packages. Returns true if the user
        discards the packages; otherwise returns false.
        """
        
        answer = QMessageBox.warning(self, self.tr("Discard List"),
            self.tr("<qt>You have marked packages for download.\n"
                    "Click <b>OK</b> to discard this list.</qt>"),
                    QMessageBox.Ok, QMessageBox.Cancel)
        
        if answer == QMessageBox.Ok:
            return True
        else:
            return False
    
    def deleteWindow(self):
    
        self.windows.remove(self.sender())
    
    def download(self):
    
        """download(self)
        
        If a download directory has been configured, a download dialog is
        opened and the current list of marked packages is submitted for
        retrieval.
        
        If no valid download directory is configured, the user is asked
        to configure one in the configuration dialog.
        """
        
        if not self.settings.value("Download directory").isValid():
            QMessageBox.information(self, self.tr("Cannot Download Packages"),
                self.tr("<qt>You need to configure a download directory "
                        "before you can download packages. Open the "
                        "<b>Settings</b> menu and select "
                        "<b>Configure Browser...</b> to access the "
                        "browser's configuration."),
                QMessageBox.Ok)
        elif not self.settings.value("Package preferences").isValid():
            QMessageBox.information(self, self.tr("Cannot Download Packages"),
                self.tr("<qt>You need to configure preferences for the "
                        "types of packages you want to download. Open the "
                        "<b>Settings</b> menu and select "
                        "<b>Configure Browser...</b> to access the "
                        "browser's configuration."),
                QMessageBox.Ok)
        else:
            dialog = DownloadDialog(self.settings, self)
            dialog.show()
            dialog.execute(list(self.marked()))
    
    def editShortcuts(self):
    
        """editShortcuts(self)
        
        Opens a dialog to allow the user to edit the shortcuts used in the
        application.
        """
        
        actions = self.findChildren(QAction)
        dialog = ActionEditorDialog(actions, self)
        if dialog.exec_() == QDialog.Accepted:
            dialog.saveSettings(self.settings, actions)
            self.settings.sync()
    
    def listClassifiers(self, url):
    
        """listClassifiers(self, url)
        
        Returns true if the list of known classifiers from the current
        package index can be obtained; otherwise returns false. This
        test is used to check whether the URL used for the package index
        is valid. (It would be better if we could check for the presence
        of a usable XML-RPC server.)
        
        The URL used to obtain a list of classifiers is based on the Python
        Package Index URL:
        
        http://www.python.org/pypi?%3Aaction=list_classifiers
        
        The query may possibly be used with other package indexes.
        """
        
        try:
            u = urllib2.urlopen(url+u"?%3Aaction=list_classifiers")
            line = None
            while line != "":
                line = u.readline()
                qApp.processEvents()
            u.close()
        except:
            return False
        
        return True
    
    def marked(self):
    
        """marked(self)
        
        This generator returns marked packages one at a time.
        """
        
        for packageName in self.searchModel.markedPackages:
        
            package, versions = self.searchModel.markedPackages[packageName]
            
            for release in package.releases:
            
                if release.version in versions:
                
                    name = release.description.metadata["name"]
                    release_urls = release.description.metadata["release_urls"]
                    home_url = release.description.metadata["home_page"]
                    yield (name, release.version, release_urls, home_url)
    
    def openIndex(self):
    
        """openIndex(self)
        
        Opens a new package index specified by the user in an input dialog,
        checking first that the URL given corresponds to the location of
        a usable XML-RPC server.
        
        If the URL is invalid, the current package index is not replaced.
        """
        
        gen = self.marked()
        try:
            gen.next()
            if not self.confirmDiscard():
                return
        except StopIteration:
            pass
        
        url = unicode(self.settings.value("Package index").toString())
        if not url:
            url = u"http://cheeseshop.python.org/pypi"
        
        url, valid = QInputDialog.getText(self, self.tr("Open Index"),
            self.tr("Enter the URL of a package index."), QLineEdit.Normal, url)
        
        if not valid:
            return
        
        self.settings.setValue("Package index", QVariant(url))
        
        # Fetch a list of classifiers.
        if not self.listClassifiers(unicode(url)):
            QMessageBox.information(self, self.tr("Package Index Unavailable"),
                self.tr("The package index you specified is currently "
                        "unavailable.\n"
                        "(I failed to obtain a list of package classifiers.)"))
            return
        
        self.fieldComboBox.setEnabled(True)
        self.searchLineEdit.setEnabled(True)
        self.searchButton.setEnabled(True)
        self.reloadListAction.setEnabled(True)
        self.filterMarkedAction.setEnabled(True)
        self.filterNewAction.setEnabled(True)
        
        self.package_server = pypi.PackageServer(unicode(url))
        self.changeServer()
        self.searchModel.clear()
        self.reloadPackages()
        self.windows = []
        self.treeView.setEnabled(True)
        self.treeView.resizeColumnToContents(0)
        
        self.searchLineEdit.setFocus(Qt.OtherFocusReason)
        self.setWindowTitle(self.tr("PyPI Browser - %1").arg(url))
    
    def openManual(self):
    
        """openManual(self)
        
        Opens the manual supplied with this application in the user's
        web browser.
        """
        
        directory = os.path.join(os.path.split(__file__)[0], "Documents")
        desktop.open(os.path.join(directory, "Manual"+os.extsep+"html"))
    
    def reloadPackages(self):
    
        """reloadPackages(self)
        
        Reloads information about the packages in the current package index
        while retaining search and marked package information held by the
        search model.
        """
        
        # Clear the search model first to prevent old indexes from
        # being referenced when they are invalidated in the underlying
        # source model.
        self.searchModel.reset()
        self.packageModel.listPackages()
        self.packageModel.load(self.settings)
        self.searchModel.search()
    
    def resizeColumns(self):
    
        self.treeView.resizeColumnToContents(0)
    
    def setMarkedFilter(self, enable):
    
        self.searchModel.setMarkedFilter(enable)
        #self.treeView.resizeColumnToContents(0)
    
    def setNewFilter(self, enable):
    
        self.searchModel.setNewFilter(enable)
        #elf.treeView.resizeColumnToContents(0)
    
    def showInformation(self, index):
    
        if not index.isValid():
            return
        elif not index.parent().isValid():
            self.showPackageInformation(index)
        else:
            self.showReleaseInformation(index)
    
    def showPackageInformation(self, index):
    
        window = InformationWindow()
        window.setPackageInfo(index)
        window.show()
        self.windows.append(window)
        self.connect(window, SIGNAL("closed()"), self.deleteWindow)
    
    def showReleaseInformation(self, index):
    
        window = InformationWindow()
        window.setPackageInfo(index.parent(), index)
        window.show()
        self.windows.append(window)
        self.connect(window, SIGNAL("closed()"), self.deleteWindow)
    
    def showResults(self, message):
    
        self.statusBar().showMessage(message)
        self.treeView.resizeColumnToContents(0)
    
    def showWaitCursor(self):
    
        self.setCursor(Qt.WaitCursor)
