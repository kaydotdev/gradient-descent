import sys
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotTab:
    def __init__(self):
        self.widget = QtWidgets.QWidget()
        self.figure = Figure()

        layout = QtWidgets.QVBoxLayout(self.widget)
        canvas = FigureCanvas(self.figure)
        layout.addWidget(canvas)


class Window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        layout = QtWidgets.QVBoxLayout(self)

        algorithm_selection_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(algorithm_selection_layout)

        algorithm_dropdown = QtWidgets.QComboBox()
        algorithm_dropdown.addItems(['SGD', 'Momentum', 'NAG', 'Adagrad', 'RMSProp', 'ADAM'])
        algorithm_selection_layout.addWidget(QtWidgets.QLabel("Algorithm:"))
        algorithm_selection_layout.addWidget(algorithm_dropdown, stretch=1)

        smoothing_degree = QtWidgets.QSpinBox()
        algorithm_selection_layout.addWidget(QtWidgets.QLabel("Smoothing degree:"))
        algorithm_selection_layout.addWidget(smoothing_degree, stretch=1)

        x_input = QtWidgets.QLineEdit()
        algorithm_selection_layout.addWidget(QtWidgets.QLabel("{x} = "))
        algorithm_selection_layout.addWidget(x_input, stretch=1)

        params_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(params_layout)

        function_input = QtWidgets.QLineEdit()
        params_layout.addWidget(QtWidgets.QLabel("F(x) = "))
        params_layout.addWidget(function_input, stretch=1)

        action_layout = QtWidgets.QHBoxLayout()
        layout.addLayout(action_layout)

        generate_button = QtWidgets.QPushButton("Generate")
        generate_button.clicked.connect(self.generate_plot)
        action_layout.addWidget(generate_button, stretch=1)

        clear_button = QtWidgets.QPushButton("Clear fields")
        clear_button.clicked.connect(self.clear_fields)
        action_layout.addWidget(clear_button, stretch=1)

        plot_widget = QtWidgets.QTabWidget()
        tab_projection, tab_trajectory, tab_loss = PlotTab(), PlotTab(), PlotTab()

        plot_widget.addTab(tab_projection.widget, "3D projection")
        plot_widget.addTab(tab_trajectory.widget, "Trajectory")
        plot_widget.addTab(tab_loss.widget, "Loss/Iteration")

        layout.addWidget(plot_widget)

        self.setLayout(layout)
        self.setWindowTitle("Gradient performance tester")
        self.setGeometry(100, 100, 800, 600)

    def generate_plot(self):
        pass

    def clear_fields(self):
        pass


# Run the application
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
