# -*- coding: utf-8 -*-
"""
/***************************************************************************
 EUSDialog
                                 A QGIS plugin
 EUS
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2019-12-09
        git sha              : $Format:%H$
        copyright            : (C) 2019 by Peter Wernerus / Fraunhofer IOSB
        email                : peter.wernerus@iosb.fraunhofer.de
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os

from PyQt5 import uic
from PyQt5 import QtWidgets

# This loads your .ui file so that PyQt can populate your plugin with the elements from Qt Designer
FORM_CLASS, _ = uic.loadUiType(os.path.join(
    os.path.dirname(__file__), 'GinLab_Gui.ui'))


class GinLab_Gui(QtWidgets.QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        """Constructor."""
        super(GinLab_Gui, self).__init__(parent)
        # Set up the user interface from Designer through FORM_CLASS.
        # After self.setupUi() you can access any designer object by doing
        # self.<objectname>, and you can use autoconnect slots - see
        # http://qt-project.org/doc/qt-4.8/designer-using-a-ui-file.html
        # #widgets-and-dialogs-with-auto-connect
        self.setupUi(self)
        self.keyPressEventFigure = None

    def keyPressEvent(self, event):
        super(GinLab_Gui, self).keyPressEvent(event)
        if self.keyPressEventFigure is not None:
            self.keyPressEventFigure(event)
