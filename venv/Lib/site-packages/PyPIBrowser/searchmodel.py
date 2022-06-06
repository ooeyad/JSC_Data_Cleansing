#!/usr/bin/env python

"""
searchmodel.py

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

import pypi
from PyQt4.QtCore import *
from PyQt4.QtGui import QFont, QSortFilterProxyModel
from packagemodel import PackageModel, ProxyModelMixIn


class SearchModel(QSortFilterProxyModel, ProxyModelMixIn):

    """SearchModel(QSortFilterProxyModel, ProxyModelMixIn)
    
    A model for providing search results from a package index based on
    searches for packages using standard fields and user-specified text.
    
    This model is a filtering proxy model, meaning that it filters
    communication between a source model and any other components, such
    as views and delegates. It performs two functions, providing filtering
    of items based on the results of package index searches and optional
    filtering of items based on whether they are checked or not.
    
    Internal Implementation:
    
    When the search() method is called, a dictionary is compiled that
    relates package names to dictionaries containing information about
    each matching package release. When the filterAcceptsRow() method
    is called to filter rows out of the information supplied by the
    package model, the data in the first column of the relevant row (the
    package name) is checked against the results dictionary, and the
    row is discarded if the package name is not in the dictionary.
    
    Similarly, when the setData() method is called with an index
    corresponding to a release and a value for the CheckStateRole, the
    state of the markedReleases dictionary is updated to include the
    relevant package and releases so that the filterAcceptsRow() method
    can discard packages that have no marked releases if the filter is
    enabled.
    
    Note that the filterAcceptsRow() method checks for marked packages
    before examining the results dictionary and checks for a special
    None value for the results to ensure that no filtering is performed
    if no valid search has been made.
    """
    
    fields = \
    (
        "name", "version", "author", "summary", "description",
        "stable_version", "author_email", "maintainer",
        "maintainer_email", "license", "platform", "download_url",
        "home_page", "keywords", "classifiers"
    )
    
    def __init__(self, server, parent = None):
    
        QSortFilterProxyModel.__init__(self, parent)
        
        self.package_server = server
        self.field = "name"
        self.terms = ""
        self.results = None
        self.markedPackages = {}
        self.markedFilter = False
        self.newFilter = False
        
        self.displayFields = \
        {
            "name": self.tr("Name"), "version": self.tr("Version"),
            "author": self.tr("Author"), "summary": self.tr("Summary"),
            "description": self.tr("Description"), 
            "stable_version": self.tr("Stable version"),
            "author_email": self.tr("Author e-mail"),
            "maintainer": self.tr("Maintainer"),
            "maintainer_email": self.tr("Maintainer e-mail"),
            "license": self.tr("License"), "platform": self.tr("Platform"),
            "download_url": self.tr("Download URL"),
            "home_page": self.tr("Home page"), 
            "keywords": self.tr("Keywords"),
            "classifiers": self.tr("Classifiers")
        }
        
        font = QFont()
        font.setBold(True)
        self.newPackageFont = QVariant(font)
    
    def clear(self):
    
        """clear(self)
        
        Clears any internal mappings defined by the model and additionally
        resets the results and marked packages dictionaries.
        """
        
        self.results = None
        self.markedPackages = {}
        QSortFilterProxyModel.clear(self)
    
    def columnCount(self, index):
    
        return 3
    
    def data(self, index, role):
    
        """data(self, index, role)
        
        Returns the data described by the given role for the item
        corresponding to the specified index.
        
        The method only provides information for indexes where the
        CheckStateRole is relevant. All other requests are passed to the
        base class which ensures that data from the source model is passed
        through correctly to views and delegates.
        """
        
        sourceIndex = self.mapToSource(index)
        parent = sourceIndex.parent()
        
        if role == Qt.CheckStateRole:
            if parent.isValid() and sourceIndex.column() == 0:
            
                # Quick test: If the package name is not in the marked
                # packages dictionary then return an unchecked state.
                # We need to check for invalid download roles here because
                # it's not enough to return the correct flags in the flags()
                # method.
                
                try:
                    packageName = unicode(parent.data().toString())
                    package, markedReleases = self.markedPackages[packageName]
                except KeyError:
                    if index.data(PackageModel.DownloadRole).isValid():
                        return QVariant(Qt.Unchecked)
                    else:
                        return QVariant()
                
                # If the release name is in the list of marked releases for
                # the package then return a checked state; otherwise return
                # an unchecked state. We need to check for invalid download
                # roles here because it's not enough to return the correct
                # flags in the flags() method.
                
                releaseName = unicode(sourceIndex.data().toString())
                if releaseName in markedReleases:
                    return QVariant(Qt.Checked)
                elif index.data(PackageModel.DownloadRole).isValid():
                    return QVariant(Qt.Unchecked)
                else:
                    return QVariant()
        
        elif role == Qt.FontRole and not parent.isValid() and \
            sourceIndex.data(PackageModel.NewPackageRole).toBool():
        
            return self.newPackageFont
        
        return QSortFilterProxyModel.data(self, index, role)
    
    def filterAcceptsRow(self, source_row, source_parent):
    
        """filterAcceptsRow(self, source_row, source_parent)
        
        Returns true if the model exposes the given source_row containing
        child items corresponding to child indexes of the source_parent
        model index; otherwise returns false.
        
        If the marked package filter is enabled, this method first filters
        out rows corresponding to packages if there are no marked releases
        for those packages.
        
        If a search has been performed, the first column of each unfiltered
        row (corresponding to a package name for a top-level item and
        a release version for a first-level item) is checked against the
        results dictionary. Those releases that are not in the dictionary
        and packages with no marked releases are discarded.
        """
        
        # If the marked or new package filters are being applied then filter
        # out top-level items that aren't in either the marked packages or
        # the new packages dictionaries.
        
        if not source_parent.isValid():
            index = self.sourceModel().index(source_row, 0, source_parent)
            if self.markedFilter:
                if unicode(index.data().toString()) not in self.markedPackages:
                    return False
            if self.newFilter:
                if not index.data(PackageModel.NewPackageRole).toBool():
                    return False
        
        if self.results is None:
            return True
        elif source_parent.isValid():
            return True
        
        index = self.sourceModel().index(source_row, 0, source_parent)
        name = unicode(index.data().toString())
        
        if name in self.results:
        
            # Add the information returned from the search to the item from
            # the source model.
            details = self.results[name]
            package = index.internalPointer()
            if package.releases is not None:
            
                for release in package.releases:
                    if release.version == details["version"]:
                        break
                else:
                    release = pypi.Release(package, details["version"])
                    package.releases.append(release)
                
                if not release.description:
                    release.description = pypi.Description(release, details)
            
            return True
        
        else:
            return False
    
    def flags(self, index):
    
        """flags(self, index)
        
        Returns the flags for the item corresponding to the specified index.
        
        All items are enabled, and the first column of rows containing
        release information are also checkable.
        """
        
        if not index.isValid():
            return QSortFilterProxyModel.flags(self, index)
        
        parent = index.parent()
        
        if not parent.isValid():
            # Top-level packages are enabled only.
            return Qt.ItemIsEnabled
        
        else:
            # The first column of release items are enabled and checkable.
            if index.column() == 0:
                if index.data(PackageModel.DownloadRole).isValid():
                    return Qt.ItemIsEnabled | Qt.ItemIsUserCheckable
                else:
                    return Qt.ItemIsEnabled
            else:
                return Qt.ItemIsEnabled
    
    def hasChildren(self, index):
    
        """hasChildren(self, index)
        
        Returns true if the item in the source model corresponding to the
        given index has child items; otherwise returns false.
        
        This reimplemented method is necessary to ensure that views do not
        call rowCount() for each top-level index and cause the source model
        to query the server for each package.
        """
        
        return self.sourceModel().hasChildren(self.mapToSource(index))
    
    def listPackages(self):
    
        """listPackages(self)
        
        Convenience method that resets the model and requests a new list of
        packages via the source model.
        """
        
        QSortFilterProxyModel.reset(self)
        self.sourceModel().listPackages()
    
    def search(self):
    
        """search(self)
        
        Searches the package index using the search field and terms, then
        resets the model to ensure that attached components are updated to
        take account of the new search results.
        
        The results are processed to make a dictionary mapping package names
        to the match for each package.
        """
        
        if self.terms.strip() == u"":
            self.results = None
            self.reset()
            self.sendMessage()
            return
        
        self.emit(SIGNAL("operationStarted()"))
        self.results = {}
        
        for result in self.package_server.search(
            {self.field: self.terms}):
            self.results[result["name"]] = result
        
        self.reset()
        self.sendMessage()
        self.emit(SIGNAL("operationFinished()"))
    
    def sendMessage(self):
    
        """sendMessage(self)
        
        Inform attached components about the results of a search or filtering
        operation.
        """
        
        if self.results is None:
            if self.markedFilter and self.newFilter:
                self.emit(SIGNAL("resultsFound(const QString &)"),
                          self.tr("Showing all new marked packages."))
            elif self.markedFilter:
                self.emit(SIGNAL("resultsFound(const QString &)"),
                          self.tr("Showing all marked packages."))
            elif self.newFilter:
                self.emit(SIGNAL("resultsFound(const QString &)"),
                          self.tr("Showing all new packages."))
            else:
                self.emit(SIGNAL("resultsFound(const QString &)"),
                          self.tr("Showing all packages."))
        else:
            if self.markedFilter and self.newFilter:
                self.emit(SIGNAL("resultsFound(const QString &)"),
                          self.tr("Showing new marked packages from a set of %1 with '%2' matching '%3'.").arg(
                          str(len(self.results)), self.displayFields[self.field],
                          self.terms))
            elif self.markedFilter:
                self.emit(SIGNAL("resultsFound(const QString &)"),
                          self.tr("Showing marked packages from a set of %1 with '%2' matching '%3'.").arg(
                          str(len(self.results)), self.displayFields[self.field],
                          self.terms))
            elif self.newFilter:
                self.emit(SIGNAL("resultsFound(const QString &)"),
                          self.tr("Showing new packages from a set of %1 with '%2' matching '%3'.").arg(
                          str(len(self.results)), self.displayFields[self.field],
                          self.terms))
            else:
                if QT_VERSION & 0xffff00 < 0x40200:
                    text = self.tr("Showing %1 package(s) with '%2' matching '%3'.", "Number of packages")
                else:
                    text = self.tr("Showing %1 package(s) with '%2' matching '%3'.", "Number of packages", len(self.results))
                
                self.emit(SIGNAL("resultsFound(const QString &)"),
                          text.arg(str(len(self.results)), self.displayFields[self.field],
                                   self.terms))
    
    def setData(self, index, value, role):
    
        """setData(self, index, value, role)
        
        Sets the data for the item corresponding to the given index and role
        to the specified value.
        
        This model only intercepts the CheckStateRole to ensure that, when
        releases are marked, the internal dictionary of marked packages
        is updated.
        
        For all other roles, the data is passed through to the source model.
        """
        
        if not index.isValid():
            return QSortFilterProxyModel.setData(self, index, value, role)
        elif role != Qt.CheckStateRole:
            return QSortFilterProxyModel.setData(self, index, value, role)
        
        # The dictionary contains rows of top-level items in the model,
        # so don't try and store rows of child items as it will only
        # lead to confusion.
        
        sourceIndex = self.mapToSource(index)
        parent = sourceIndex.parent()
        
        if parent.isValid():
        
            packageName = unicode(parent.data().toString())
            package, markedReleases = self.markedPackages.get(packageName,
                                      (parent.internalPointer(), {}))
            
            releaseName = unicode(sourceIndex.data().toString())
            if value.toBool():
                markedReleases[releaseName] = True
            elif releaseName in markedReleases:
                del markedReleases[releaseName]
            
            if markedReleases != {}:
                self.markedPackages[packageName] = package, markedReleases
            elif self.markedPackages.has_key(packageName):
                del self.markedPackages[packageName]
            
            self.emit(SIGNAL("markedChanged(bool)"), self.markedPackages != {})
            self.emit(SIGNAL("dataChanged(const QModelIndex &, const QModelIndex &)"),
                      index, index)
            return True
        
        return False
    
    def setMarkedFilter(self, enable):
    
        """setMarkedFilter(self, enable)
        
        Enables or disables the marked package filter.
        """
        
        self.markedFilter = enable
        self.reset()
        self.sendMessage()
    
    def setNewFilter(self, enable):
    
        """setNewFilter(self, enable)
        
        Enables or disables the new package filter.
        """
        
        self.newFilter = enable
        self.reset()
        self.sendMessage()
    
    def setSearchField(self, row):
    
        """setSearchField(self, row)
        
        Sets the search field to the text in the field list specified by the
        row and updates the model with the results of a new search.
        """
        
        self.field = self.fields[row]
        self.search()
    
    def setSearchTerms(self, text):
    
        """setSearchTerms(self, text)
        
        Sets the search terms to the text specified.
        
        No search is performed since the text supplied may be incomplete.
        When the text is complete, it is expected that another component
        will call the search() method when a suitable user action occurs.
        """
        
        self.terms = unicode(text)
    
    def setServer(self, server):
    
        self.package_server = server
