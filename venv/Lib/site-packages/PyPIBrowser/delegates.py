#!/usr/bin/env python

"""
delegates.py

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

from PyQt4.QtCore import Qt
from PyQt4.QtGui import QItemDelegate, QPalette, QPen, QStyle

try:
    from PyQt4.QtGui import QStylePainter, QStyleOptionProgressBarV2
    with_style = True
except ImportError:
    with_style = False


class ProgressDelegate(QItemDelegate):

    """ProgressDelegate(QItemDelegate)
    
    A custom delegate that displays data obtained from a model in the form
    of a progress bar.
    
    The text used on the bar is obtained for a given model index using the
    standard DisplayRole, and the percentage value indicating the progress
    of some operation is obtained using the UserRole.
    """
    
    def __init__(self, parent = None):
    
        QItemDelegate.__init__(self, parent)
    
    def paint(self, painter, option, index):
    
        """paint(self, painter, option, index)
        
        Paints the contents of the delegate using the given painter to
        perform painting operations. The specified option contains
        information about the paint device.
        
        The contents of the delegate are obtained from a model using the
        specified model index.
        """
        
        value = index.data(Qt.UserRole)
        if not value.isValid():
            return QItemDelegate.paint(self, painter, option, index)
        else:
            percentage = value.toDouble()[0]
        
        painter.setPen(Qt.NoPen)

        if option.state & QStyle.State_Selected:
            backgroundBrush = option.palette.alternateBase()
            labelBrush = option.palette.highlight()
            color = labelBrush.color()
            color.setAlpha(127)
            labelBrush.setColor(color)
        else:
            backgroundBrush = option.palette.base()
            labelBrush = option.palette.highlight()
            color = labelBrush.color()
            color.setAlpha(127)
            labelBrush.setColor(color)
        
        painter.setBrush(backgroundBrush)
        painter.drawRect(option.rect)
        
        if with_style:
        
            stylePainter = QStylePainter()
            stylePainter.begin(painter.device(), self.parent())
            progressOption = QStyleOptionProgressBarV2()
            progressOption.initFrom(self.parent())
            progressOption.rect = option.rect
            progressOption.minimum = 0
            progressOption.maximum = 100
            progressOption.progress = int(percentage)
            progressOption.text = index.data(Qt.DisplayRole).toString()
            progressOption.textAlignment = Qt.AlignCenter
            progressOption.textVisible = True
            
            stylePainter.drawControl(QStyle.CE_ProgressBar, progressOption)
            stylePainter.end()
        
        else:
            painter.setBrush(labelBrush)
            w = option.rect.width() * percentage/100.0
            painter.drawRect(option.rect.x(), option.rect.y(),
                             w, option.rect.height())
            
            if option.state & QStyle.State_Selected:
                painter.setPen(QPen(option.palette.color(QPalette.HighlightedText)))
            else:
                painter.setPen(QPen(option.palette.color(QPalette.Text)))
        
        painter.setFont(option.font)
        painter.drawText(option.rect, Qt.AlignCenter,
                         index.data(Qt.DisplayRole).toString())
