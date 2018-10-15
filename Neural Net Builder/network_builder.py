import inspect

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization

from ipywidgets import Layout, IntText, FloatText, Checkbox, HTML

from utils import *
from train_plots import *
from diagnostic_plots import *


overflow_args = dict(overflow_x='hidden', overflow_y='hidden')
textbox_layout1 = Layout(width='100px', **overflow_args)
textbox_layout2 = Layout(width='180px', **overflow_args)
textbox_layout3 = Layout(width='250px', **overflow_args)
btn_layout = Layout(width='34px')

widget_map = {float: FloatText, int: IntText, bool: Checkbox}

activation_choices = get_activations()
loss_choices = get_losses()
optimizer_choices = get_optimizers()


class Layer(Box):
    def __init__(self, *args, **kwargs):
        self.name = kwargs['name']
        self.build_widgets()
        kwargs['children'] = [self.widget_layout]
        super(Layer, self).__init__(*args, **kwargs)

    def build_widgets(self, *args, **kwargs):
        self.label = HTML(value=self.name)
        self.nodes_box = IntText(layout=textbox_layout1, value=1)
        self.widget_layout = VBox([self.label, self.nodes_box])

    def get_data(self):
        return dict(name=self.name, nodes=self.nodes_box.value)


class InputLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(InputLayer, self).__init__(*args, **kwargs)

    def build_widgets(self, *args, **kwargs):
        Layer.build_widgets(self, *args, **kwargs)
        self.nodes_box.disabled = True


class OutputLayer(Layer):
    def __init__(self, *args, **kwargs):
        super(OutputLayer, self).__init__(*args, **kwargs)

    def build_widgets(self, *args, **kwargs):
        Layer.build_widgets(self, *args, **kwargs)
        self.activation_dropdown = Dropdown(options=activation_choices,
                                            layout=textbox_layout1,
                                            value=activation_choices['linear'])
        self.widget_layout = VBox([self.label,
                                   self.nodes_box,
                                   self.activation_dropdown])

    def get_data(self):
        return dict(name=self.name,
                    nodes=self.nodes_box.value,
                    activation=self.activation_dropdown.value)


class HiddenLayer(OutputLayer):
    def __init__(self, *args, **kwargs):
        super(HiddenLayer, self).__init__(*args, **kwargs)
        self.label.layout.width = '100px'

    def build_widgets(self, *args, **kwargs):
        OutputLayer.build_widgets(self, *args, **kwargs)
        self.nodes_box.disabled = False

        self.activation_dropdown.value = activation_choices['relu']
        self.dropout_prob_box = FloatText(value=0,
                                          layout=textbox_layout1)
        self.bn_check = Checkbox(layout=textbox_layout1)
        self.widget_layout = VBox([self.label,
                                   self.nodes_box,
                                   self.activation_dropdown,
                                   self.bn_check,
                                   self.dropout_prob_box])

    def get_data(self):
        return dict(name=self.name,
                    nodes=self.nodes_box.value,
                    activation=self.activation_dropdown.value,
                    batch_norm=self.bn_check.value,
                    dropout_p=self.dropout_prob_box.value)


class NeuralNetworkBuilder(Box):
    def __init__(self, *args, **kwargs):
        self.X_train, self.y_train = kwargs['train_data']
        self.X_val, self.y_val = kwargs.get('val_data', (None, None))

        self.metrics = kwargs.get('metrics', 'acc')

        self.width = kwargs.get('width', 1300)
        self.height = kwargs.get('height', 600)

        self.tab_layout = Layout(width=str(self.width) + 'px',
                                 height=str(self.height) + 'px')
        default_train_dashboard = TrainingPlotsDashboard(width=self.width,
                                                         height=self.height)
        self.train_plots = kwargs.get('train_plots', default_train_dashboard)

        default_train_callback = TrainingCallback(dashboard=self.train_plots,
                                                  X_train=self.X_train,
                                                  y_train=self.y_train)
        self.train_callback = kwargs.get('train_callback',
                                         default_train_callback)

        default_diag_plots = DiagnosticPlots()
        self.diagnostics_plots = kwargs.get('diagnostic_plots',
                                            default_diag_plots)
        self.diagnostics_plots.set_layout(self.tab_layout)

        self.hidden_layers = []
        self.hidden_layer_layout = HBox()
        self.build_widgets()

        kwargs['children'] = [self.widget_layout]
        super(NeuralNetworkBuilder, self).__init__(*args, **kwargs)

    def build_widgets(self, *args, **kwargs):
        # widgets for network params tab
        self.epoch_box = IntText(description='Epochs',
                                 value=100,
                                 layout=textbox_layout2)

        self.batch_box = IntText(description='Batch Size',
                                 value=64,
                                 layout=textbox_layout2)

        self.loss_box = Dropdown(description='Loss',
                                 options=loss_choices,
                                 value=loss_choices['mse'],
                                 layout=textbox_layout3)

        self.optimizer_box = Dropdown(description='Optimizer',
                                      options=optimizer_choices,
                                      value=optimizer_choices['adam'],
                                      layout=textbox_layout3)
        self.optim_params_label = HTML('<div style="width: 250px; ' +
                                       'text-align: center; ' +
                                       'font-size: 16px">optimizer params' +
                                       '</div>')
        self.optim_params_layout = VBox()
        self.optimizer_box.observe(self.populate_optim_params, 'value')
        self.populate_optim_params(None)
        self.network_params_layout = VBox([self.epoch_box,
                                           self.batch_box,
                                           self.loss_box,
                                           HBox([self.optimizer_box,
                                                 VBox([
                                                      self.optim_params_label,
                                                      self.optim_params_layout
                                                      ])
                                                 ])],
                                          layout=self.tab_layout)

        # widgets for layers tab
        label_tmpl = '<div style="font-size: 18px; height: 28px">{}</div>'
        self.labels_layout = VBox([HTML(label_tmpl.format('')),
                                   HTML(label_tmpl.format('Nodes')),
                                   HTML(label_tmpl.format('Activation')),
                                   HTML(label_tmpl.format('Batch Norm')),
                                   HTML(label_tmpl.format('Dropout prob'))])

        self.hidden_layers_label = \
            HTML('<div style="font-size: 18px">Hidden Layers</div>')

        self.add_layer_btn = Button(icon='fa-plus', button_style='success',
                                    layout=btn_layout)
        self.remove_layer_btn = Button(icon='fa-minus', button_style='danger',
                                       layout=btn_layout)
        self.add_layer_btn.on_click(lambda btn: self.add_hidden_layer())
        self.remove_layer_btn.on_click(lambda btn: self.remove_hidden_layer())
        self.hidden_layer_btns = HBox([self.hidden_layers_label,
                                       self.add_layer_btn,
                                       self.remove_layer_btn],
                                      layout=Layout(margin='6px 0px 20px 0px'))

        self.input_layer = InputLayer(name='Inputs')
        self.input_layer.nodes_box.value = self.X_train.shape[1]

        self.output_layer = OutputLayer(name='Outputs')

        if len(self.y_train.shape) > 1:
            self.output_layer.nodes_box.value = self.y_train.shape[1]
        else:
            self.output_layer.nodes_box.value = 1

        self.layers_layout = VBox([self.hidden_layer_btns,
                                   HBox([self.labels_layout,
                                         self.input_layer,
                                         self.hidden_layer_layout,
                                         self.output_layer])],
                                  layout=self.tab_layout)

        # widgets for diagnostic plots
        self.status_label = HTML()

        self.train_btn = Button(description='Train Model',
                                button_style='success')
        self.train_btn.on_click(lambda btn: self.train_model())

        self.training_layout = VBox([self.train_btn,
                                     self.train_plots], layout=self.tab_layout)

        self.tab = Tab([self.network_params_layout, self.layers_layout,
                        self.training_layout, self.diagnostics_plots],
                       _titles={0: 'Network Parameters', 1: 'Architecture',
                                2: 'Training', 3: 'Diagnostics'})
        self.widget_layout = VBox([self.tab, self.status_label])

    def populate_optim_params(self, *args):
        """
        inspect the selected optimizer from the optimizer box and create
        input widgets for the params and populate them with defaults
        """
        argspec = inspect.getfullargspec(self.optimizer_box.value)
        param_wids = [widget_map.get(type(val), Text)(description=arg,
                                                      layout=textbox_layout2,
                                                      value=val)
                      for arg, val in zip(argspec.args[1:], argspec.defaults)]
        self.optim_params_layout.children = param_wids

    def add_hidden_layer(self):
        # adds GUI for a new hidden layer
        hidden_layer = HiddenLayer(name='Hidden Layer ' +
                                   str(len(self.hidden_layers) + 1))
        self.hidden_layers.append(hidden_layer)
        self.hidden_layer_layout.children = self.hidden_layers

    def remove_hidden_layer(self):
        # removes GUI for the last hidden layer
        if len(self.hidden_layers) > 0:
            self.hidden_layers.pop()
            self.hidden_layer_layout.children = self.hidden_layers

    def train_model(self):
        # set the training progress bar's max value to correct value
        self.train_plots.epochs = self.epoch_box.value
        self.train_plots.progress_bar.max = self.epoch_box.value - 1

        self.model = Sequential()

        # add hidden layers
        for i, h in enumerate(self.hidden_layer_layout.children):
            layer_data = h.get_data()
            if layer_data:
                if i == 0:  # first layer
                    self.model.add(Dense(layer_data['nodes'],
                                         input_dim=self.X_train.shape[1]))
                else:
                    self.model.add(Dense(layer_data['nodes']))

                # add batch norm layer if requested
                # make sure it's added *before* activations
                if layer_data['batch_norm']:
                    self.model.add(BatchNormalization())

                self.model.add(Activation(layer_data['activation']))

                # add drop out layer if p > 0
                dropout_p = layer_data['dropout_p']
                if dropout_p > 0:
                    self.model.add(Dropout(dropout_p))

        # add output layer
        output_layer_data = self.output_layer.get_data()

        if len(self.hidden_layer_layout.children) == 0:  # no hidden layers
            self.model.add(Dense(output_layer_data['nodes'],
                                 input_dim=self.X_train.shape[1],
                                 activation=output_layer_data['activation']))
        else:
            self.model.add(Dense(output_layer_data['nodes'],
                                 activation=output_layer_data['activation']))

        optim_args = {txt_box.description: txt_box.value for txt_box
                      in self.optim_params_layout.children}
        optimizer = self.optimizer_box.value(**optim_args)
        self.model.compile(loss=self.loss_box.value,
                           optimizer=optimizer,
                           metrics=[self.metrics])

        self.diagnostics_plots.reset()

        if self.X_val is not None and self.y_val is not None:
            self.fit_hist = self.model.fit(self.X_train, self.y_train,
                                           validation_data=(self.X_val,
                                                            self.y_val),
                                           epochs=self.epoch_box.value,
                                           batch_size=self.batch_box.value,
                                           verbose=0,
                                           callbacks=[self.train_callback])
        else:
            self.fit_hist = self.model.fit(self.X_train, self.y_train,
                                           epochs=self.epoch_box.value,
                                           batch_size=self.batch_box.value,
                                           verbose=0,
                                           callbacks=[self.train_callback])

        # set the trained model for the diagnostic plots
        self.diagnostics_plots.update(y_true=self.y_val,
                                      X=self.X_val,
                                      model=self.model)
