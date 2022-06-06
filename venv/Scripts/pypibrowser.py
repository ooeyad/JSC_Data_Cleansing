#!c:\users\oo_ey\pycharmprojects\golden_record_project\venv\scripts\python.exe

"""
pypibrowser.py

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

import sys
from PyQt4.QtCore import QString, QLocale, QTranslator
from PyQt4.QtGui import QApplication

from PyPIBrowser.window import Window
from PyPIBrowser import pypi_resources

if __name__ == "__main__":

    app = QApplication(sys.argv)
    
    translator = QTranslator()
    locale = QLocale.system().name().toLower()
    translator.load(QString(":/translations/pypibrowser_%1.qm").arg(locale))
    app.installTranslator(translator)
    
    window = Window()
    window.show()
    sys.exit(app.exec_())
