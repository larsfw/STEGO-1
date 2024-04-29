from .GinLab_Gui import GinLab_Gui
from .GinLab_Gui_d import Ui_MainWindow as GinLab_Gui_d
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QAction, QToolBar, QCheckBox, QHBoxLayout,QRadioButton, QLineEdit
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from numpy import matlib as ml
import math


class MyNavigationToolbar(NavigationToolbar):
    def __init__(self, figure ,gui):
        NavigationToolbar.__init__(self, figure ,gui)


class ImageCanvas(FigureCanvas):
    def __init__(self, parent=None, img=None):
        if img is None:
            img = []
        FigureCanvas.__init__(self, Figure())
        self.marker = []
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        self.img = img

        self.graph_1 = [0, 0]
        self.graph_2 = [0, 0]
        self.graph_counter = 0

        self.marker_x = 0
        self.marker_y = 0
        nn = len(self.img)

        sq = math.ceil(math.sqrt(nn))
        if nn > 0:
            self.figure.subplots(sq, math.ceil(nn / sq), sharey=True, sharex=True)

        for i in range(0, nn):
            img[i] =  np.atleast_3d(img[i])

            self.figure.axes[i].imshow(self.img[i], cmap='gray')
            # fig.axes[i].set_axis_off()
            self.figure.axes[i].set_yticks([])
            self.figure.axes[i].set_xticks([])
            self.figure.axes[i].set_title(str(i + 1))
            self.figure.axes[i].set_xlabel(str(i + 1))

        if sq*sq > nn:
            for i in range(nn, sq * math.ceil(nn / sq)):
                self.figure.axes[i].remove()

        self.mpl_connect('button_press_event', self.on_click)
        parent.keyPressEventFigure = self.key_press
        self.figure.tight_layout()
        self.navigation_toolbar = MyNavigationToolbar(self, parent)
        self.on_click_toolbar = QToolBar("My main toolbar", parent)

        button_group = QtWidgets.QWidget()
        layout = QHBoxLayout()
        button_group.setLayout(layout)

        self.b = [None]*2
        c = QRadioButton("Point klick")

        self.b[0] = QRadioButton("Point klick")
        self.b[0].setChecked(True)
        layout.addWidget(self.b[0])

        self.b[1] = QRadioButton("Line tool")
        layout.addWidget(self.b[1])

        self.line_tool_channel = QLineEdit()
        self.line_tool_channel.setSizePolicy(QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed))
        self.line_tool_channel.setText('al')
        self.line_tool_channel.setMaximumWidth(30)
        layout.addWidget(self.line_tool_channel)
        self.on_click_toolbar.addWidget(button_group)
        # self.gui.NTB.press_zoom(self.ntb_pressed)


    def key_press(self, event):
        # print('you pressed', event.key, event.xdata, event.ydata)

        if event.key() == 16777235:
            self.delete_marker()
            self.update_marker(self.marker_x, self.marker_y - 1)

        if event.key() == 16777237:
            self.delete_marker()
            self.update_marker(self.marker_x, self.marker_y + 1)

        if event.key() == 16777234:
            self.delete_marker()
            self.update_marker(self.marker_x - 1, self.marker_y)

        if event.key() == 16777236:
            self.delete_marker()
            self.update_marker(self.marker_x + 1, self.marker_y)
        pass

    def manage_marker(self, event):

        print('manage_marker')

        if self.navigation_toolbar.mode != '':
            print('tool selected: do nothing')
            return

        if event.inaxes is None:
            return

        if event.button == QtCore.Qt.LeftButton:
            print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                  ('double' if event.dblclick else 'single', event.button,
                   event.x, event.y, event.xdata, event.ydata))
            self.delete_marker()
            if event.dblclick:
                return
            self.update_marker(np.round(event.xdata), round(event.ydata))

    def delete_marker(self):

        for i in self.marker:
            i.remove()
        self.marker = []
        self.draw()

    def update_marker(self, x=0, y=0):
        self.marker_x = x
        self.marker_y = y
        self.marker = [None] * len(self.figure.axes)

        for i in range(0, len(self.figure.axes)):
            dims = np.shape(self.img[i])
            self.marker[i] = self.figure.axes[i].plot(x % dims[1] , y % dims[0], 'r+')[0]

        for i in range(0, len(self.figure.axes)):
            dims = np.shape(self.img[i])
            xxx = self.img[i][int(y) % dims[0], int(x) % dims[1], :]
            self.figure.axes[i].set_xlabel('f(' + str(int(x)% dims[1] ) + ',' + str(int(y) % dims[0]) + ') = ' + str(np.round(xxx * 100) / 100))
        self.draw()
        # print('update_marker')

    def on_click(self, event):
        print(self.b[0].isChecked())
        print('on_click')
        if self.b[0].isChecked():
            self.manage_marker(event)
        if self.b[1].isChecked():
            self.line_tool(event)

    def line_tool(self, event):

        if self.navigation_toolbar.mode != '':
            print('tool selected: do nothing')
            return

        if event.inaxes is None:
            return

        if self.graph_counter <= 0:
            self.graph_1 = [np.round(event.xdata), round(event.ydata)]
            self.graph_counter = 1

        elif self.graph_counter == 1:

            self.graph_counter = 0

            im_idx = self.figure.axes.index(event.inaxes)
            options = self.line_tool_channel.text()

            self.graph_2 = [np.round(event.xdata), round(event.ydata)]

            x = np.array(self.graph_1)
            y = np.array(self.graph_2)
            ll = np.max(np.abs(x-y))

            ls = np.tile(np.linspace(0, 1, int(ll)), (2,1))
            x1 = np.tile(x, (int(ll),1)).transpose()
            y1 = np.tile(y, (int(ll), 1)).transpose()
            val_idx = np.round(x1 + ls*(y1-x1)).astype(int)

            m,n,s = self.img[0].shape

            fig = Figure()
            figc = FigureCanvas(fig)

            toolbar = NavigationToolbar(figc, figc)
            fig.subplots(1,1)
            ax = fig.axes[0]
            ax_idx = event.inaxes.numCols

            for option in options:

                if option == 'a':

                    for cur_channel in range(0,s):

                        cur_image = self.img[im_idx][:, :, cur_channel]
                        image_val = cur_image[val_idx[1], val_idx[0]]
                        ax.plot(np.linspace(0, 1, int(ll)), image_val, label='Channel'+str(cur_channel))

                if option == 'm':
                    ss = len(self.img)
                    for ii in range(0, ss):
                        cur_image = np.sum(self.img[ii][:, :, :], axis=2)

                        image_val = cur_image[val_idx[1], val_idx[0]]/s
                        ax.plot(np.linspace(0, 1, int(ll)), image_val, label='Image ' + str(ii))

                if option == 'l':
                    ss = len(self.img)
                    p = self.figure.axes[im_idx].plot([x[0], y[0]], [x[1], y[1]], 'b-')
                    self.draw()
                if option.isnumeric():

                    cur_channel = int(option)
                    if cur_channel < s:
                        cur_image = self.img[im_idx][:, :, cur_channel]
                        image_val = cur_image[val_idx[1], val_idx[0]]
                        ax.plot(np.linspace(0, 1, int(ll)), image_val, label='Channel' + str(cur_channel))

                ax.legend()
            figc.set_window_title('asd')
            figc.mpl_connect('close_event', self.test)
            #figc.close_event = self.test
            figc.show()


    def test(self,event):
        print('Test')

class PlotFigure(FigureCanvas):
    def __init__(self):
        FigureCanvas.__init__(self)

class GinLab:

    def __init__(self, img=[]):
        self.gui = GinLab_Gui()
        self.gui_d = GinLab_Gui_d()
        # self.gui_d.actionAbout.(self.reset_view)
        self.gui.action_about.triggered.connect(self.action_about_trigger)
        self.gui.action_quit.triggered.connect(self.action_quit_trigger)
        self.gui.action_save_image.triggered.connect(self.action_save_image_trigger)
        self.gui.action_view.triggered.connect(self.action_view_trigger)
        self.gui.action_tight_layout.triggered.connect(self.action_tight_layout_trigger)
        self.gui.show()

        self.main_image_canvas = ImageCanvas(self.gui, img)
        self.gui.addToolBar(QtCore.Qt.TopToolBarArea, self.main_image_canvas.navigation_toolbar)
        self.gui.addToolBar(QtCore.Qt.TopToolBarArea, self.main_image_canvas.on_click_toolbar)
        self.gui.grid_layout_figure.addWidget(self.main_image_canvas)

        self.main_image_canvas.draw()

    def action_tight_layout_trigger(self):
        self.main_image_canvas.figure.tight_layout()
        self.main_image_canvas.draw()
        print('action_tight_layout_trigger')

    def action_view_trigger(self):
        print('action_view_trigger')

    def action_quit_trigger(self):
        self.quit
        print('action_quit_trigger')

    def action_save_image_trigger(self):
        print('action_save_image_trigger')

    def action_about_trigger(self):
        print('action_about_trigger')

    def disp_images(self, img):
        self.gui.removeToolBar(self.main_image_canvas.navigation_toolbar)
        self.gui.grid_layout_figure.removeWidget(self.main_image_canvas)

        self.main_image_canvas = ImageCanvas(self.gui, img)
        self.gui.addToolBar(QtCore.Qt.TopToolBarArea, self.main_image_canvas.navigation_toolbar)
        self.gui.addToolBar(QtCore.Qt.TopToolBarArea, self.main_image_canvas.on_click_toolbar)
        self.gui.grid_layout_figure.addWidget(self.main_image_canvas, 1)
        self.main_image_canvas.draw()

    @staticmethod
    def run(img):
        from PyQt5 import QtWidgets
        import sys

        app = QtWidgets.QApplication(sys.argv)
        gl = GinLab(img)
        # gl.disp_images(img)
        #sys.exit(app.exec_())
        return app.exec_()

