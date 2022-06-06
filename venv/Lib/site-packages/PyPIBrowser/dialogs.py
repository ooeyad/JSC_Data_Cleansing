#!/usr/bin/env python

"""
dialogs.py

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

from actioneditor import ActionEditorDialog
import distutils.command, distutils.util, os, sys, urllib2, urlparse
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from ui_configurationdialog import Ui_ConfigurationDialog
from ui_downloaddialog import Ui_DownloadDialog
from ui_informationwindow import Ui_InformationWindow
from delegates import ProgressDelegate
from packagemodel import PackageModel
import desktop


class ConfigurationDialog(QDialog, Ui_ConfigurationDialog):

    """ConfigurationDialog(QDialog, Ui_ConfigurationDialog)
    
    Provides a configuration dialog with basic functionality for changing
    settings used by various components of the browser.
    """
    
    def __init__(self, settings, parent = None):
    
        QDialog.__init__(self, parent)
        self.setupUi(self)
        
        self.settings = settings
        self.setPythonPlaceholders()
        
        directory = self.settings.value("Download directory")
        self.setDownloadDirectory(directory)
        
        self.getPackagePreferences()
        self.setPackagePreferences()
        
        self.getPackageIndex()
        
        self.connect(self.packageIndexLineEdit, SIGNAL("textChanged(const QString &)"),
                     self.validatePackageIndex)
        self.connect(self.downloadButton, SIGNAL("clicked()"),
                     self.getDownloadDirectory)
        self.connect(self.preferencesList, SIGNAL("itemSelectionChanged()"),
                     self.updatePreferencesButtons)
        self.connect(self.upButton, SIGNAL("clicked()"), self.movePreferenceUp)
        self.connect(self.downButton, SIGNAL("clicked()"), self.movePreferenceDown)
        self.connect(self.hideButton, SIGNAL("clicked()"), self.hidePreference)
        self.connect(self.systemPathsList, SIGNAL("itemActivated(QListWidgetItem *)"),
                     self.openSystemPath)
    
    def getDownloadDirectory(self):
    
        """getDownloadDirectory(self)
        
        Ask the user to specify an existing directory using a standard file
        dialog for the browser to use when saving downloaded packages.
        If the new directory path is not valid, the current directory is
        retained.
        """
        
        path = QFileDialog.getExistingDirectory(self,
            self.tr("Choose Download Directory"))
        
        if not path.isNull():
            self.setDownloadDirectory(path)
    
    def getPackageIndex(self):
    
        packageIndex = self.settings.value("Package index")
        if packageIndex.isValid():
            self.packageIndexLineEdit.setText(packageIndex.toString())
        else:
            self.packageIndexLineEdit.setText(u"http://cheeseshop.python.org/pypi")
        
        self.packageIndexLineEdit.setCursorPosition(0)
    
    def getPackagePreferences(self):
    
        preferences = self.settings.value("Package preferences")
        
        if preferences.isValid():
            commands = preferences.toStringList()
        else:
            commands = filter(lambda command: command.find("dist") != -1,
                              distutils.command.__all__)
            commands.insert(0, "default")
        
        for command in commands:
        
            if u"|" in command:
                text, state = unicode(command).split(u"|")
            else:
                text = command
                state = u"E"
            
            item = QListWidgetItem(self.preferencesList)
            item.setText(text)
            if state == u"H":
                item.setFlags(item.flags() ^ Qt.ItemIsEnabled)
    
    def movePreferenceUp(self):
    
        item = self.preferencesList.selectedItems()[0]
        row = self.preferencesList.row(item)
        if row > 0:
            item = self.preferencesList.takeItem(row)
            self.preferencesList.insertItem(row - 1, item)
            self.preferencesList.setCurrentRow(row - 1)
    
    def movePreferenceDown(self):
    
        item = self.preferencesList.selectedItems()[0]
        row = self.preferencesList.row(item)
        if row < self.preferencesList.count() - 1:
            item = self.preferencesList.takeItem(row)
            self.preferencesList.insertItem(row + 1, item)
            self.preferencesList.setCurrentRow(row + 1)
    
    def hidePreference(self):
    
        item = self.preferencesList.selectedItems()[0]
        row = self.preferencesList.row(item)
        item.setFlags(item.flags() ^ Qt.ItemIsEnabled)
        self.updatePreferencesButtons()
    
    def openSystemPath(self, item):
    
        desktop.open(unicode(item.text()))
    
    def saveSettings(self):
    
        self.settings.setValue("Download directory",
            QVariant(self.downloadLineEdit.text()))
        self.setPackagePreferences()
        self.setPackageIndex()
    
    def setDownloadDirectory(self, directory):
    
        """setDownloadDirectory(self, directory)
        
        Sets the directory used by the browser to hold downloaded packages
        and updates the relevant field in the dialog.
        
        The directory must be specified using either a string or a QVariant
        that holds a string.
        
        If the directory is not valid, the field in the dialog is disabled,
        indicating that package downloading is disabled.
        """
        
        if isinstance(directory, QVariant):
            directory = directory.toString()
        
        if directory.isNull():
            self.downloadLineEdit.setEnabled(False)
        else:
            self.downloadLineEdit.setEnabled(True)
            self.downloadLineEdit.setText(directory)
            self.downloadLineEdit.setCursorPosition(0)
    
    def setPackageIndex(self):
    
        url = self.packageIndexLineEdit.text()
        if self.validateURL(url):
            self.settings.setValue("Package index", QVariant(url))
    
    def setPackagePreferences(self):
    
        commands = []
        for row in range(self.preferencesList.count()):
            item = self.preferencesList.item(row)
            text = unicode(item.text())
            if item.flags() & Qt.ItemIsEnabled:
                state = u"E"
            else:
                state = u"H"
            commands.append(u"|".join((text, state)))
        
        self.settings.setValue("Package preferences", QVariant(commands))
    
    def setPythonPlaceholders(self):
    
        self.versionPlaceholder.setText(
            self.tr("%1.%2.%3").arg(sys.version_info[0]).arg(
                sys.version_info[1]).arg(sys.version_info[2]))
        
        self.platformPlaceholder.setText(distutils.util.get_platform())
        
        for path in sys.path:
            item = QListWidgetItem(self.systemPathsList)
            item.setText(path)
    
    def updatePreferencesButtons(self):
    
        enable = len(self.preferencesList.selectedItems()) > 0
        self.upButton.setEnabled(enable)
        self.downButton.setEnabled(enable)
        self.hideButton.setEnabled(enable)
        
        if enable:
            item = self.preferencesList.selectedItems()[0]
            if item.flags() & Qt.ItemIsEnabled:
                self.hideButton.setText(self.tr("&Hide"))
            else:
                self.hideButton.setText(self.tr("S&how"))
    
    def validateURL(self, text):
    
        pieces = filter(lambda piece: piece.strip() != u"",
                        urlparse.urlsplit(unicode(text))[:2])
        
        return len(pieces) == 2
    
    def validatePackageIndex(self, text):
    
        palette = self.packageIndexLineEdit.palette()
        
        if not self.validateURL(text):
            palette.setColor(QPalette.Text, Qt.red)
            palette.setColor(QPalette.Base, Qt.white)
        else:
            palette.setColor(QPalette.Text, QPalette().color(QPalette.Text))
            palette.setColor(QPalette.Base, QPalette().color(QPalette.Base))
        
        self.packageIndexLineEdit.setPalette(palette)


class DownloadDialog(QDialog, Ui_DownloadDialog):

    """DownloadDialog(QDialog, Ui_DownloadDialog)
    
    Provides a dialog that shows the progress of a series of download
    operations.
    
    Each package file that is successfully (or partially) downloaded is
    stored in a directory defined in the settings object specified when
    the dialog is created.
    """
    
    def __init__(self, settings, parent = None):
    
        QDialog.__init__(self, parent)
        self.setupUi(self)
        QMetaObject.connectSlotsByName(self)
        
        self.settings = settings
        self.stopped = False
        
        self.treeWidget.setColumnCount(3)
        self.treeWidget.setHeaderLabels(
            QStringList() << self.tr("Name") << self.tr("File name") << self.tr("Progress")
            )
        delegate = ProgressDelegate(self)
        self.treeWidget.setItemDelegate(delegate)
        self.treeWidget.setMouseTracking(True)
        self.treeWidget.mouseMoveEvent = self._mouseMoveEvent
        
        font = QFont()
        font.setUnderline(True)
        self.linkFont = QVariant(font)
        
        self.connect(self.stopButton, SIGNAL("clicked()"), self.stopDownload)
        self.connect(self.openDirButton, SIGNAL("clicked()"), self.openDirectory)
        self.connect(self.treeWidget,
            SIGNAL("itemEntered(QTreeWidgetItem *, int)"),
            self.changeCursor)
        self.connect(self.treeWidget,
            SIGNAL("itemClicked(QTreeWidgetItem *, int)"),
            self.launchBrowser)
    
    def _mouseMoveEvent(self, event):
    
        item = self.treeWidget.itemAt(event.pos())
        if not item:
            self.treeWidget.unsetCursor()
        QTreeWidget.mouseMoveEvent(self.treeWidget, event)
    
    def changeCursor(self, item, column):
    
        if column == 1 and item.data(column, Qt.UserRole+1).isValid():
            self.treeWidget.setCursor(Qt.PointingHandCursor)
        else:
            self.treeWidget.unsetCursor()
    
    def downloadPackage(self, item, directory, download_urls, completed, packages):
    
        for url in download_urls:
        
            try:
                path = urllib2.urlparse.urlsplit(url)[2]
                filename = path.split("/")[-1]
                savePath = os.path.join(directory, filename)
                item.setText(1, filename)
                
                f = open(savePath, "wb")

                u = urllib2.urlopen(url)
                info = u.info()
                length = int(info.getheader("Content-length"))
                digits = len("%i" % length)

                total = 0
                while True:

                    bytes = u.read(4096)
                    read = len(bytes)
                    f.write(bytes)
                    total += read
                    
                    if QT_VERSION & 0xffff00 < 0x40200:
                        text = self.tr("%1/%2 byte(s) (%3%)", "Total number of bytes")
                    else:
                        text = self.tr("%1/%2 byte(s) (%3%)", "Total number of bytes", length)
                    item.setText(2, text.arg(total, digits).arg(length)
                                        .arg(int(100*float(total)/length)))
                    item.setData(2, Qt.UserRole, QVariant(100*float(total)/length))
                    self.progressBar.setValue(100*(float(completed) + float(total)/length)/packages)
                    qApp.processEvents()

                    if read < 4096 or self.stopped:
                        break
                
                u.close()
                f.close()
                
                if self.stopped and total < length:
                    # Remove the file if it was only partially downloaded.
                    os.remove(savePath)
                
                # We successfully downloaded a package, or the download was
                # stopped, so break out of the loop.
                break

            except:
                # Try the next URL in the list.
                pass
        
        else:
            # All URLs were tried, but none could be used to obtain a package.
            return False
        
        return True
    
    def execute(self, packages):
    
        """execute(self, packages)
        
        Download each of the packages in a list to a directory specified
        in the application's settings.
        
        If no suitable download directory is defined in the settings, the
        method returns immediately.
        
        The event loop is run periodically, enabling the progress of the
        download operation to be reporting and allowing the user to cancel
        the operation if required.
        """
        
        if not self.settings.value("Download directory").isValid():
            return
        
        if not self.settings.value("Package preferences").isValid():
            return
        
        directory = unicode(self.settings.value("Download directory").toString())
        if not os.path.isdir(directory):
            return
        
        self.startDownload()
        qApp.processEvents()
        
        # Compile a dictionary of hidden package types from the package
        # preferences.
        hidden = {}
        ordered = []
        for command in self.settings.value("Package preferences").toStringList():
        
            text, state = unicode(command).split(u"|")
            if state == u"H":
                hidden[text] = None
            else:
                ordered.append(text)
        
        completed = 0
        for name, version, release_urls, home_url in packages:
        
            item = QTreeWidgetItem(self.treeWidget)
            item.setText(0, name)
            item.setText(2, self.tr("Fetching..."))
            item.setData(2, Qt.UserRole, QVariant(0))
            item.setFlags(Qt.ItemIsEnabled)
            
            qApp.processEvents()
            
            download_urls = {}
            for i in range(0, len(release_urls), 4):
                release_url = {}
                release_url[release_urls[i]] = release_urls[i+1]
                release_url[release_urls[i+2]] = release_urls[i+3]
                if release_url[u"packagetype"] not in hidden:
                    download_urls[release_url[u"packagetype"]] = release_url[u"url"]
            
            if download_urls:
                # Create an ordered list of URLs by filtering the ordered list
                # to include only keys that are available in the dictionary
                # supplied for this package, then use those keys to access the
                # dictionary.
                ordered_urls = map(lambda k: download_urls[k],
                                   filter(lambda k: k in download_urls, ordered))
                
                downloaded = self.downloadPackage(item, directory, ordered_urls,
                                                  completed, len(packages))
            else:
                downloaded = False
            
            if not downloaded:
                if not home_url or not urlparse.urlsplit(home_url)[0]:
                    item.setText(2, self.tr("Failed"))
                else:
                    item.setText(2, home_url)
                    item.setData(2, Qt.UserRole, QVariant())
                    item.setData(2, Qt.UserRole+1, QVariant(home_url))
                    item.setData(2, Qt.FontRole, self.linkFont)
            
            completed += 1
            
            self.progressBar.setValue(100*float(completed)/len(packages))
            
            if self.stopped:
                break
        
        self.stopDownload()
    
    def launchBrowser(self, item, column):
    
        if column == 1:
            variant = item.data(column, Qt.UserRole+1)
            if variant.isValid():
                home_url = unicode(variant.toString())
                desktop.open(home_url)
    
    def startDownload(self):
    
        """startDownload(self)
        
        Prepares the user interface for use during a download operation.
        """
        
        self.stopped = False
        self.stopButton.setEnabled(True)
        self.closeButton.setEnabled(False)
        self.treeWidget.clear()
    
    def openDirectory(self):
    
        desktop.open(unicode(self.settings.value("Download directory").toString()))
    
    def stopDownload(self):
    
        """stopDownload(self)
        
        Resets the user interface after a download operation.
        """
        
        self.stopped = True
        self.stopButton.setEnabled(False)
        self.closeButton.setEnabled(True)
    
    def reject(self):
    
        """reject(self)
        
        Stops any current download operation and rejects the dialog in the
        standard way.
        """
        
        self.stopDownload()
        QDialog.reject(self)


class InformationWindow(QWidget, Ui_InformationWindow):

    releaseFieldColumns = (1, 4, 2, 3, 5, 6, 7, 8, 10, 11)
    
    def __init__(self, parent = None):
    
        QWidget.__init__(self, parent)
        self.setupUi(self)
        
        self.connect(self.textBrowser, SIGNAL("anchorClicked(const QUrl &)"),
                     self.openURL)
        self.connect(self.closeAction, SIGNAL("triggered()"), self.close)
        self.addAction(self.closeAction)
        
        self.textCharFormat = QTextCharFormat()
        self.textCharFormat.setFont(QFont())
        
        self.textBlockFormat = QTextBlockFormat()
        self.textBlockFormat.setAlignment(Qt.AlignJustify)
        
        self.titleCharFormat = QTextCharFormat(self.textCharFormat)
        self.titleCharFormat.setFontWeight(QFont.Bold)
        self.titleCharFormat.setFontPointSize(self.titleCharFormat.fontPointSize()*2)
        
        self.titleBlockFormat = QTextBlockFormat()
        self.titleBlockFormat.setAlignment(Qt.AlignHCenter)
        
        self.subtitleCharFormat = QTextCharFormat(self.titleCharFormat)
        self.subtitleCharFormat.setFontPointSize(self.subtitleCharFormat.fontPointSize()*0.8)
        
        self.tableHeaderFormat = QTextCharFormat(self.textCharFormat)
        self.tableHeaderFormat.setFontWeight(QFont.Bold)
        
        self.anchorCharFormat = QTextCharFormat(self.textCharFormat)
        self.anchorCharFormat.setForeground(QBrush(Qt.blue))
        self.anchorCharFormat.setFontUnderline(True)
        
        self.releaseTableFormat = QTextTableFormat()
        self.releaseTableFormat.setAlignment(Qt.AlignLeft)
        self.releaseTableFormat.setBorder(0)
        self.releaseTableFormat.setCellSpacing(0)
        self.releaseTableFormat.setCellPadding(4)
        self.releaseTableFormat.setColumnWidthConstraints(
            [QTextLength(QTextLength.PercentageLength, 20),
             QTextLength(QTextLength.PercentageLength, 80)])
        
        self.markedReleaseFrameFormat = QTextFrameFormat()
        self.markedReleaseFrameFormat.setBackground(QBrush(QColor(238,224,224)))
        
        self.currentReleaseFrameFormat = QTextFrameFormat()
        self.currentReleaseFrameFormat.setBackground(QBrush(QColor(224,224,238)))
        self.currentReleaseFrameFormat.setBorder(1)
        
        self.evenRowBlockFormat = QTextBlockFormat()
        self.evenRowBlockFormat.setBackground(QBrush(QColor(238,238,238)))
        
        self.oddRowBlockFormat = QTextBlockFormat()
        self.oddRowBlockFormat.setBackground(QBrush(QColor(224,224,224)))
    
    def closeEvent(self, event):
    
        event.accept()
        self.emit(SIGNAL("closed()"))
    
    def openURL(self, url):
    
        desktop.open(unicode(url.toString()))
        self.textBrowser.setSource(QUrl())
    
    def setPackageInfo(self, index, releaseIndex = None):
    
        cursor = self.textBrowser.textCursor()
        cursor.insertBlock(self.titleBlockFormat)
        cursor.insertText(index.data().toString(), self.titleCharFormat)
        self.setWindowTitle(self.tr("Information about %1").arg(index.data().toString()))
        
        for row in range(index.model().rowCount(index)):
        
            version = index.child(row, 0).data().toString()
            cursor.insertBlock(self.textBlockFormat)
            cursor.insertText(version, self.subtitleCharFormat)
            
            topLevelCursor = QTextCursor(cursor)
            
            checkState, valid = index.child(row, 0).data(Qt.CheckStateRole).toInt()
            if valid and checkState == Qt.Checked:
                frame = cursor.insertFrame(self.markedReleaseFrameFormat)
            elif releaseIndex and releaseIndex.row() == row:
                frame = cursor.insertFrame(self.currentReleaseFrameFormat)
            else:
                frame = cursor.insertFrame(QTextFrameFormat())
            
            table = cursor.insertTable(len(self.releaseFieldColumns), 2,
                                       self.releaseTableFormat)
            
            tableRow = 0
            rowFormats = [self.evenRowBlockFormat, self.oddRowBlockFormat]
            
            for column in self.releaseFieldColumns:
            
                #rowFormat = rowFormats[tableRow % 2]
                
                fieldName = index.model().headerData(column, Qt.Horizontal, Qt.DisplayRole).toString()
                cell = table.cellAt(tableRow, 0)
                cursor = cell.firstCursorPosition()
                #cursor.setBlockFormat(rowFormat)
                cursor.insertText(fieldName, self.tableHeaderFormat)
                
                fieldValue = index.child(row, column).data().toString()
                cell = table.cellAt(tableRow, 1)
                cursor = cell.firstCursorPosition()
                #cursor.setBlockFormat(rowFormat)
                cursor.insertText(fieldValue)
                tableRow += 1
            
            # Add home page and download information.
            homePage = index.child(row, 9).data().toString()
            homePageURL = index.child(row, 0).data(PackageModel.HomePageRole)
            homePageHeader = index.model().headerData(9, Qt.Horizontal, Qt.DisplayRole).toString()
            
            if homePageURL.isValid():
            
                table.insertRows(table.rows(), 1)
                
                cell = table.cellAt(table.rows()-1, 0)
                cursor = cell.firstCursorPosition()
                cursor.insertText(homePageHeader, self.tableHeaderFormat)
                
                cell = table.cellAt(table.rows()-1, 1)
                cursor = cell.firstCursorPosition()
                anchorFormat = QTextCharFormat(self.anchorCharFormat)
                anchorFormat.setAnchorHref(homePageURL.toString())
                anchorFormat.setAnchor(True)
                cursor.insertText(homePage, anchorFormat)
            
            cursor = topLevelCursor
            cursor.movePosition(QTextCursor.End)
