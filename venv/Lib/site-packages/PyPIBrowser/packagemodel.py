#!/usr/bin/env python

"""
packagemodel.py

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

import base64
from PyQt4.QtCore import *
from PyQt4.QtGui import QAbstractProxyModel
import pypi


class PackageModel(QAbstractItemModel):

    """PackageModel(QAbstractItemModel)
    
    A model for obtaining package information from a package index.
    """
    
    headers = (
        "Package", "Author", "Summary", "Description", "Author e-mail",
        "Maintainer", "Maintainer e-mail", "License", "Platform",
        "Home page", "Keywords", "Stable version"
        )
    
    section_list = (
        "version", "author", "summary", "description", "author_email",
        "maintainer", "maintainer_email", "license", "platform",
        "home_page", "keywords", "stable_version"
        )
    
    DownloadRole = Qt.UserRole
    HomePageRole = Qt.UserRole + 1
    NewPackageRole = Qt.UserRole + 2
    UnusedRole = Qt.UserRole + 3
    
    def __init__(self, server, parent = None):
    
        QAbstractItemModel.__init__(self, parent)
        
        self.package_server = server
        self.listPackages()
    
    def listPackages(self):
    
        """listPackages(self)
        
        Returns a list of available packages and resets the model, informing
        other components that the underlying structure and data provided by
        the model has changed.
        """
    
        self.emit(SIGNAL("operationStarted()"))
        self.packages = self.package_server.list_packages()
        self.reverse = {}
        for i in range(len(self.packages)):
            self.reverse[self.packages[i]] = i
        self.reset()
        self.emit(SIGNAL("operationFinished()"))
    
    def hasChildren(self, parent):
    
        """hasChildren(self, parent)
        
        Returns true if the item corresponding to the parent index has
        child items; otherwise returns false.
        
        To begin with, we assume that all top-level items (packages) have
        children to reduce calls to the server. Once these items have been
        opened, more precise information will be provided by the rowCount()
        method.
        
        First level items (releases) have no children.
        """
        
        if not parent.isValid():
            # Top-level items
            return True
        
        parent_item = parent.internalPointer()
        
        if isinstance(parent_item, pypi.Package):
            # Items under packages
            return True
        else:
            return False
    
    def rowCount(self, parent):
    
        """rowCount(self, parent)
        
        Returns the number of rows containing child items corresponding to
        children of the given parent index.
        """
        
        if not parent.isValid():
            # Top-level items
            return len(self.packages)
        
        parent_item = parent.internalPointer()
        
        if isinstance(parent_item, pypi.Package):
            # Items under packages
            package = parent_item
            if package.releases is None:
                self.emit(SIGNAL("operationStarted()"))
                package.releases = self.package_server.package_releases(package)
                self.emit(SIGNAL("operationFinished()"))
            return len(package.releases)
        else:
            return 0
    
    def columnCount(self, parent):
    
        """columnCount(self, parent)
        
        Returns the number of columns in the model regardless of the number
        of columns containing items corresponding to children of the parent
        index.
        
        The number returned is based on the number of sections we want to
        expose to views.
        """
        
        return len(self.section_list)
    
    def flags(self, index):
    
        """flags(self, index)
        
        Returns the flags for the item corresponding to the given index.
        
        All items are enabled by default.
        """
        
        if not index.isValid():
            return QAbstractItemModel.flags(self, index)
        
        return Qt.ItemIsEnabled
    
    def index(self, row, column, parent):
    
        """index(self, row, column, parent)
        
        Returns the model index for the item whose parent item corresponds
        to the given parent index, and that resides in the specified row
        and column.
        """
        
        if not parent.isValid():
            # Top-level items
            parent_item = None
        else:
            parent_item = parent.internalPointer()
        
        if parent_item is None:
            try:
                package = self.packages[row]
            except IndexError:
                return QModelIndex()
            return self.createIndex(row, column, package)
        
        elif isinstance(parent_item, pypi.Package):
            # Items under packages
            package = parent_item
            try:
                release = package.releases[row]
            except IndexError:
                return QModelIndex()
            return self.createIndex(row, column, release)
        
        return QModelIndex()
    
    def parent(self, index):
    
        """parent(self, index)
        
        Returns the model index for the parent item of the item corresponding
        to the specified index.
        """
        
        if not index.isValid():
            return QModelIndex()
        
        item = index.internalPointer()
        
        if isinstance(item, pypi.Package):
            # Top-level packages have no parent.
            return QModelIndex()
        
        elif isinstance(item, pypi.Release):
            return self.createIndex(self.reverse[item.package], 0,
                                    item.package)
        else:
            return QModelIndex()
    
    def headerData(self, section, orientation, role = Qt.DisplayRole):
    
        """headerData(self, section, orientation, role = Qt.DisplayRole)
        
        Returns the header titles for each column in the model.
        """
        
        if orientation != Qt.Horizontal or role != Qt.DisplayRole:
            return QVariant()
        
        try:
            text = self.headers[section]
        except IndexError:
            return QVariant()
        
        return QVariant(self.tr(text))
    
    def data(self, index, role):
    
        """data(self, index, role)
        
        Returns the data described by the given role for the item
        corresponding to the specified index.
        
        For top-level items, this model only returns data for the DisplayRole
        in the first column since this corresponds to the name of each package.
        
        For first-level items, the version, author and a summary for each
        release is returned for the DisplayRole in each column. In the first
        column, the download URL is returned for the UserRole, and the home
        page URL is returned for UserRole+1.
        """
        
        if not index.isValid():
            return QVariant()
        elif role == Qt.DisplayRole:
            pass
        elif Qt.UserRole <= role < self.UnusedRole:
            pass
        else:
            return QVariant()
        
        row = index.row()
        if not 0 <= row < self.rowCount(index.parent()):
            return QVariant()
        
        column = index.column()
        
        item = index.internalPointer()
        
        if isinstance(item, pypi.Package):
        
            if column != 0:
                return QVariant()
            elif role == Qt.DisplayRole:
                return QVariant(item.name)
            elif role == self.NewPackageRole:
                return QVariant(item.new)
            else:
                return QVariant()
        
        elif isinstance(item, pypi.Release):
        
            if not 0 <= column < len(self.section_list):
                return QVariant()
            
            release = item
            if release.description is None:
                self.emit(SIGNAL("operationStarted()"))
                release.description = self.package_server.release_full_data(release)
                self.emit(SIGNAL("operationFinished()"))
            
            if column == 0:
                if role == Qt.DisplayRole:
                    return QVariant(release.version)
                elif role == self.DownloadRole:
                    value = release.description.metaData(u"release_urls")
                    if value:
                        return QVariant(value)
                    else:
                        return QVariant()
                elif role == self.HomePageRole:
                    value = release.description.metaData(u"home_page")
                    if value:
                        return QVariant(value)
                    else:
                        return QVariant()
                else:
                    return QVariant()
            
            if release.description:
                field = self.section_list[column]
                value = release.description.metaData(field)
            else:
                return QVariant()
            
            if value is None:
                return QVariant()
            elif role == Qt.DisplayRole:
                return QVariant(value)
            else:
                return QVariant()
        
        return QVariant()
    
    def load(self, settings):
    
        """load(self, settings)
        
        Loads information about the packages in the application's settings.
        """
        name = self.package_server.name()
        if not name:
            return
        
        settings.beginGroup("Servers")
        url = settings.value(name)
        settings.endGroup()
        
        if not url.isValid():
            return
        
        settings.beginGroup("Packages")
        settings.beginGroup(name)
        
        packageNames = {}
        for unique_string in settings.childKeys():
        
            packageName = unicode(base64.decodestring(str(unique_string)), "utf_8")
            releases = settings.value(packageName)
            
            # We don't use the release information at the moment.
            if releases.isValid():
                releases = unicode(releases).split(",")
                packageNames[unicode(packageName)] = releases
            else:
                packageNames[unicode(packageName)] = None
        
        for package in self.packages:
        
            if package.name in packageNames:
                package.new = False
        
        settings.endGroup()
        settings.endGroup()
    
    def save(self, settings):
    
        """save(self, settings)
        
        Saves information about the packages in the application's settings.
        """
        
        name = self.package_server.name()
        if not name:
            return
        
        settings.beginGroup("Servers")
        settings.setValue(name, QVariant(self.package_server.url))
        settings.endGroup()
        
        settings.beginGroup("Packages")
        settings.beginGroup(name)
        
        for package in self.packages:
        
            unique_string = base64.encodestring(package.name.encode("utf_8"))
            if package.releases:
                settings.setValue(unique_string, QVariant(
                    ",".join(map(lambda p: p.version, package.releases))))
            else:
                settings.setValue(unique_string, QVariant())
        
        settings.endGroup()
        settings.endGroup()
    
    def setServer(self, server):
    
        self.package_server = server


class ProxyModelMixIn:

    def getObject(self, index):
    
        """getObject(self, index)
        
        Returns the internal object in the source model corresponding to
        the model index previously issued by this model.
        """
        
        return self.sourceIndex(index).internalPointer()
    
    def sourceIndex(self, index):
    
        while isinstance(index.model(), QAbstractProxyModel):
            index = index.model().mapToSource(index)
        return index
