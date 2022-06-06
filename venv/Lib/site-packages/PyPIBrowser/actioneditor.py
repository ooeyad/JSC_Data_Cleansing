#!/usr/bin/env python

"""
actioneditor.py

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

from PyQt4.QtCore import QEvent, QRect, QString, Qt, QVariant, SIGNAL
from PyQt4.QtGui import qApp, QBrush, QColor, QDialog, QHBoxLayout, \
    QItemDelegate, QKeySequence, QLabel, QPainter, QPalette, QPen, \
    QPushButton, QStyle, QTableWidget, QTableWidgetItem, QVBoxLayout


class ActionEditorWidget(QLabel):

    # Redefine the tr() function for this class.
    def tr(self, text):
    
        return qApp.translate("ActionEditorWidget", text)

    def __init__(self, text, parent):
    
        QLabel.__init__(self, text, parent)
        self.key = ""
        self.modifiers = {}
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setBrush(palette.Base, palette.brush(palette.AlternateBase))
        self.setPalette(palette)
        self.valid = False
    
    def keyPressEvent(self, event):
    
        other = None
        
        if event.key() == Qt.Key_Shift:
            self.modifiers[Qt.Key_Shift] = u"Shift"
        elif event.key() == Qt.Key_Control:
            self.modifiers[Qt.Key_Control] = u"Ctrl"
        elif event.key() == Qt.Key_Meta:
            self.modifiers[Qt.Key_Meta] = u"Meta"
        elif event.key() == Qt.Key_Alt:
            self.modifiers[Qt.Key_Alt] = u"Alt"
        else:
            other = QString(QKeySequence(event.key()))
        
        if other:
            key_string = u"+".join(self.modifiers.values() + [unicode(other),])
            self.valid = True
        else:
            key_string = u"+".join(self.modifiers.values())
        
        self.setText(key_string)
    
    def keyReleaseEvent(self, event):
    
        if self.valid:
            return
        
        if event.key() == Qt.Key_Shift:
            if self.modifiers.has_key(Qt.Key_Shift):
                del self.modifiers[Qt.Key_Shift]
        elif event.key() == Qt.Key_Control:
            if self.modifiers.has_key(Qt.Key_Control):
                del self.modifiers[Qt.Key_Control]
        elif event.key() == Qt.Key_Meta:
            if self.modifiers.has_key(Qt.Key_Meta):
                del self.modifiers[Qt.Key_Meta]
        elif event.key() == Qt.Key_Alt:
            if self.modifiers.has_key(Qt.Key_Alt):
                del self.modifiers[Qt.Key_Alt]
        
        self.setText(u"+".join(self.modifiers.values()))
        
        if len(self.modifiers) == 0:
            self.releaseKeyboard()
    
    def mousePressEvent(self, event):
    
        if event.button() != Qt.LeftButton:
            return
        
        size = self.height() / 2.0
        rect = QRect(self.width() - size, size * 0.5, size, size)
        
        if rect.contains(event.pos()):
            self.clear()
            self.valid = True
            event.accept()
    
    def paintEvent(self, event):
    
        if not self.text().isEmpty():
        
            painter = QPainter()
            painter.begin(self)
            painter.setRenderHint(QPainter.Antialiasing)
            
            color = self.palette().color(QPalette.Highlight)
            color.setAlpha(127)
            painter.setBrush(QBrush(color))
            color = self.palette().color(QPalette.HighlightedText)
            color.setAlpha(127)
            painter.setPen(QPen(color))
            size = self.height() / 2.0
            
            painter.drawRect(self.width() - size, size * 0.5, size, size)
            painter.drawLine(self.width() - size * 0.75, size * 0.75,
                             self.width() - size * 0.25, size * 1.25)
            painter.drawLine(self.width() - size * 0.25, size * 0.75,
                             self.width() - size * 0.75, size * 1.25)
            painter.end()
        
        QLabel.paintEvent(self, event)
    
    def showEvent(self, event):
    
        self.grabKeyboard()


class ActionEditorDelegate(QItemDelegate):

    def __init__(self, parent = None):
    
        QItemDelegate.__init__(self, parent)
    
    def createEditor(self, parent, option, index):
    
        self.editor = ActionEditorWidget(index.data().toString(), parent)
        self.editor.installEventFilter(self)
        return self.editor
    
    def eventFilter(self, obj, event):
    
        if obj == self.editor:
            if event.type() == QEvent.KeyPress:
                obj.keyPressEvent(event)
                if obj.valid:
                    self.emit(SIGNAL("commitData(QWidget *)"), self.editor)
                    self.emit(SIGNAL("closeEditor(QWidget *, QAbstractItemDelegate::EndEditHint)"),
                              self.editor, QItemDelegate.NoHint)
                return True
            
            elif event.type() == QEvent.KeyRelease:
                obj.keyReleaseEvent(event)
                if obj.text().isEmpty():
                    self.emit(SIGNAL("closeEditor(QWidget *, QAbstractItemDelegate::EndEditHint)"),
                              self.editor, QItemDelegate.NoHint)
                return True
            
            elif event.type() == QEvent.MouseButtonPress:
                obj.mousePressEvent(event)
                if obj.valid:
                    self.emit(SIGNAL("commitData(QWidget *)"), self.editor)
                    self.emit(SIGNAL("closeEditor(QWidget *, QAbstractItemDelegate::EndEditHint)"),
                              self.editor, QItemDelegate.NoHint)
                return True
        
        return False
    
    def paint(self, painter, option, index):
    
        if index.column() != 0:
            QItemDelegate.paint(self, painter, option, index)
            return
        
        painter.fillRect(option.rect, option.palette.brush(QPalette.Base))
        painter.setPen(QPen(option.palette.color(QPalette.Text)))
        painter.drawText(option.rect.adjusted(4, 4, -4, -4),
            Qt.TextShowMnemonic | Qt.AlignLeft | Qt.AlignVCenter,
            index.data().toString())
    
    def setEditorData(self, editor, index):
    
        editor.setText(index.data().toString())
    
    def setModelData(self, editor, model, index):
    
        model.setData(index, QVariant(editor.text()))
    
    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)


class ActionEditorDialog(QDialog):

    # Redefine the tr() function for this class.
    def tr(self, text):
    
        return qApp.translate("ActionEditorDialog", text)

    def __init__(self, actions, parent):
    
        QDialog.__init__(self, parent)
        self.actions = filter(lambda action: action.parent() == parent, actions)
        
        self.actionTable = QTableWidget(self)
        self.actionTable.setColumnCount(2)
        self.actionTable.setHorizontalHeaderLabels(
            [self.tr("Description"), self.tr("Shortcut")]
            )
        self.actionTable.horizontalHeader().setStretchLastSection(True)
        self.actionTable.verticalHeader().hide()
        self.actionTable.setItemDelegate(ActionEditorDelegate(self))
        
        self.connect(self.actionTable, SIGNAL("cellChanged(int, int)"),
                     self.validateAction)
        
        row = 0
        for action in self.actions:
        
            if action.text().isEmpty():
                continue
            
            self.actionTable.insertRow(self.actionTable.rowCount())
            
            item = QTableWidgetItem()
            item.setText(action.text())
            item.setFlags(Qt.ItemIsEnabled)
            self.actionTable.setItem(row, 0, item)
            
            item = QTableWidgetItem()
            item.setText(action.shortcut().toString())
            item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable)
            item.oldShortcutText = item.text()
            self.actionTable.setItem(row, 1, item)
            
            row += 1
        
        self.actionTable.resizeColumnsToContents()
        
        ok_button = QPushButton(self.tr("&OK"))
        cancel_button = QPushButton(self.tr("&Cancel"))
        
        self.connect(ok_button, SIGNAL("clicked()"), self.accept)
        self.connect(cancel_button, SIGNAL("clicked()"), self.reject)
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)
        button_layout.addStretch(1)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        
        mainLayout = QVBoxLayout()
        mainLayout.setMargin(8)
        mainLayout.setSpacing(8)
        mainLayout.addWidget(self.actionTable)
        mainLayout.addLayout(button_layout)
        self.setLayout(mainLayout)
        
        self.setWindowTitle(self.tr("Edit Shortcuts"))
    
    def accept(self):
    
        row = 0
        for action in self.actions:
        
            if not action.text().isEmpty():
                action.setText(self.actionTable.item(row, 0).text())
                action.setShortcut(QKeySequence(self.actionTable.item(row, 1).text()))
                row += 1
        
        QDialog.accept(self)
    
    def loadSettings(self, settings, actions):
    
        settings.beginGroup("Actions")
    
        for action in actions:
        
            shortcutText = settings.value(action.text()).toString()
            if not shortcutText.isEmpty():
                action.setShortcut(QKeySequence(shortcutText))
        
        settings.endGroup()
    
    loadSettings = classmethod(loadSettings)
    
    def saveSettings(self, settings, actions):
    
        settings.beginGroup("Actions")
    
        for action in actions:
        
            shortcutText = action.shortcut().toString()
            settings.setValue(action.text(), QVariant(shortcutText))
        
        settings.endGroup()
    
    saveSettings = classmethod(saveSettings)
    
    def validateAction(self, row, column):
    
        if column != 1:
            return
        
        item = self.actionTable.item(row, column)
        shortcutText = QKeySequence(item.text()).toString()
        thisRow = self.actionTable.row(item)
        
        if not shortcutText.isEmpty():
            for row in range(self.actionTable.rowCount()):
                if row == thisRow:
                    continue
                
                other = self.actionTable.item(row, 1)
                
                if other.text() == shortcutText:
                    other.setText(item.oldShortcutText)
                    break
            
            item.setText(shortcutText)
            item.oldShortcutText = shortcutText
        
        self.actionTable.resizeColumnToContents(1)
