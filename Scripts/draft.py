"""
Draft file containing code that has not been used for final results
"""
from distutils.command.sdist import sdist
import inspect
from keras.layers import Dense, Dropout, Conv1D, Conv2D, Layer, BatchNormalization, LayerNormalization
from keras.layers import Activation, SpatialDropout1D, SpatialDropout2D, Lambda, Flatten, LeakyReLU
from keras.layers import ZeroPadding1D, ZeroPadding2D
from keras import backend as K, Model, Input, optimizers, layers
import numpy as np

################ From init.py ################

class RunModels:
    """
    This class is used by optuna to permute config parameters in the models in an optimal way
    """
    def __init__(self, train_data, test_data, config, config_range=None, scalers=False):
        self.config = config
        self.config_range = config_range
        self.model_name_gen = give_modelname()
        self.train_data = train_data
        self.test_data = test_data
        self.seis_testimage_fp = "../OneDrive - NGI/Documents/NTNU/MSC_DATA/2DUHRS_06_MIG_DEPTH/TNW_B02_5110_MIG_DPT.sgy"
        self.ai_testimage_fp = "../OneDrive - NGI/Documents/NTNU/MSC_DATA/00_AI/TNW_B02_5110_MIG.Abs_Zp.sgy"
        
        # Set up directory to log tensorboard
        self.tbdir = Path('./_tb')
        if self.tbdir.exists(): rmtree(str(self.tbdir), ignore_errors=True); self.tbdir.mkdir()
        
        # Set up image folder
        if not os.path.isdir('./TEMP'): os.mkdir('./TEMP')

        # Load scalers
        if scalers:
            self.X_scaler, self.y_scaler = scalers

        # Create colormaps for plotting
        if len(self.train_data) == 2: # If the task is not just reconstruction
            traces, train_y = self.train_data

            # Making the code resilient to different dimensionality of predictions; may be only target, only reconstruct or both
            # if len(train_y) == 2: 
            #     flat_target = train_y[0].flatten()
            # elif len(train_y) == 1:
            #     flat_target = train_y.flatten()
            # else: raise ValueError('Unexpected dimansionality of target')     %%%%%%%%%% Code may not be needed because inputs are scaled

            self.target_cmap = lambda x : plt.cm.plasma(x) # Target parameter colormap
        elif len(self.train_data) == 1:
            traces = self.train_data
            raise ValueError('This part should not run')

        # Getting colormap for plotting
        self.seis_st_dev = stats.tstd(traces, axis=None)
        seis_norm = mpl.colors.Normalize(-self.seis_st_dev, self.seis_st_dev) # Seismic plot is scaled to std dev in the data
        self.seis_cmap = lambda x : plt.cm.seismic(seis_norm(x))

    def update_config_with_suggestions(self, trial):
        if not isinstance(self.sfunc, dict):
            self.sfunc = dict()
            self.sfunc['float'], self.sfunc['int'], self.sfunc['categorical'] = [trial.suggest_float, trial.suggest_int, trial.suggest_categorical]

        for key, items in self.config_range.items():
            suggest_func = self.sfunc[items[0]]
            self.config[key] = suggest_func(key, *items[1])

    def objective(self, trial):
        # Update name of the model and configs for model run
        groupname, modelname = next(self.model_name_gen)
        self.update_config_with_suggestions(trial)

        # New callback for this model instance
        self.tb_callback = tf.keras.callbacks.TensorBoard(log_dir=str(self.tbdir), histogram_freq=1)

        # Call the model to fit on training data
        model, History = compiled_TCN(self.train_data, self.config, callbacks=[self.tb_callback])

        # Saving the model
        model_loc = './Models/{}/{}'.format(groupname, modelname)
        if not os.path.isdir(model_loc): os.mkdir(model_loc)
        #model.save(model_loc)

        # Plotting the model for later analysis
        #plot_model(model.generator, to_file=model_loc+'/model.png', show_shapes=True, show_layer_names=True, expand_nested=True, show_layer_activations=True)

        # Evaluating the model
        X, Y = test_data
        error = model.evaluate(X, Y, batch_size = 20, verbose=2, steps=40)
        
        # First metric should always be loss
        if isinstance(error, list):
            tot_error = error[0]
        
        # Image colormaps
        seis_cmap = self.seis_cmap
        ai_cmap = self.target_cmap
        
        # Have to get the traces here, because groupings may change
        seis_testimage, ai_testimage, _ = get_matching_traces(self.seis_testimage_fp,
                                                              self.ai_testimage_fp, 
                                                              group_traces=self.config['group_traces'], 
                                                              trunc=80)
        
        # Scale the test images with dataset scalers
        old_X_shp = seis_testimage.shape; old_y_shp = ai_testimage.shape
        seis_testimage = self.X_scaler.transform(seis_testimage.reshape(-1, 1)); ai_testimage = self.y_scaler.transform(ai_testimage.reshape(-1, 1))
        seis_testimage = seis_testimage.reshape(old_X_shp); ai_testimage = ai_testimage.reshape(old_y_shp)

        # With the testimage create a prediction images for visual inspection
        target_pred, recon_pred, target_pred_diff = create_pred_image(model,  [seis_testimage, ai_testimage])
        #create_ai_error_image((target_pred_diff)**2, seis_testimage, filename=model_loc+'/error_image.png')

        image_folder = './TEMP/{}'.format(groupname)
        if not os.path.isdir(image_folder): os.makedirs(image_folder, exist_ok=True)
        
        # Image with comparisons
        p_name = image_folder + '/{}_combined_target_pred.jpg'.format(modelname)
        rec_p_name = image_folder + '/{}_combined_recon_pred.jpg'.format(modelname)

        # Applying colormaps to the images
        im_p = ai_cmap(target_pred); im_rec_p = seis_cmap(recon_pred)

        # Saving the images
        Image.fromarray((im_p[:, :, :3]*255).astype(np.uint8)).save(p_name)
        Image.fromarray((im_rec_p[:, :, :3]*255).astype(np.uint8)).save(rec_p_name)

        # Update scoreboard for repo image
        # if update_scores('{}/{}'.format(groupname, modelname), rec_error):
        #     replace_md_image(p_name, rec_error)

        # Save data about model params and training history
        save_config(model_loc, config)
        save_training_progression(History.history, model_loc)

        del model # Clear up the memory location for next model
        return tot_error
    


################# from Architecures.py ####################

    class multi_task_GAN(Model):

    def __init__(self, discriminators, generators, alpha=1, beta=1):
        """
        """
        super(multi_task_GAN, self).__init__()
        self.seismic_discriminator  = discriminators[1]
        self.ai_discriminator       = discriminators[0]
        self.seismic_generator      = generators[1]
        self.ai_generator           = generators[0]
        self.alpha                  = alpha
        self.beta                   = beta

    def compile(self, g_optimizer, d_optimizer, g_loss, d_loss, **kwargs):
        super(multi_task_GAN, self).compile(**kwargs)
        self.g_optimizer    = g_optimizer
        self.d_optimizer  = d_optimizer
        self.g_loss         = g_loss
        self.d_loss         = d_loss
        self.gen_X_metric  = keras.metrics.Mean(name='gen_X_loss')
        self.gen_y_metric  = keras.metrics.Mean(name='gen_y_loss')
        self.disc_X_accuracy = keras.metrics.Accuracy()
        self.disc_y_accuracy = keras.metrics.Accuracy()

    @property
    def metrics(self):
        return [self.gen_X_metric, self.gen_y_metric, self.disc_X_accuracy, self.disc_y_accuracy]
    
    def train_step(self, batch_data):
        real_X, real_y = batch_data
        batch_size = tf.shape(real_X)[0]
        real_y, _ = real_y

        real_y_1 = real_y*(tf.ones_like(real_y) + .0001*tf.random.uniform(tf.shape(real_y)))
        real_y_2 = real_y*(tf.ones_like(real_y) + .0001*tf.random.uniform(tf.shape(real_y)))

        with tf.GradientTape(persistent=True) as tape:
            fake_X = self.seismic_generator(real_X, training=True)
            fake_y = self.ai_generator(real_X, training=True)
            disc_real_X = self.seismic_discriminator(real_X, training=True)
            disc_fake_X = self.seismic_discriminator(fake_X, training=True)
            disc_real_y = self.ai_discriminator(real_y_1, training=True)
            disc_fake_y = self.ai_discriminator(fake_y, training=True)

            X_predictions = tf.concat([disc_fake_X, disc_real_X], axis=0)
            X_truth       = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)
            y_predictions = tf.concat([disc_fake_y, disc_real_y], axis=0)
            y_truth       = tf.concat([tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0)

            X_truth += 0.5e-6 * tf.random.uniform(tf.shape(X_truth), minval=0)
            y_truth += 0.5e-6 * tf.random.uniform(tf.shape(y_truth), minval=0)

            # Discriminator loss
            self.disc_X_accuracy.update_state(X_truth, X_predictions)
            self.disc_y_accuracy.update_state(y_truth, y_predictions)
            disc_X_loss = self.d_loss(X_truth, X_predictions)
            disc_y_loss = self.d_loss(y_truth, y_predictions)
        
        # Get discriminator gradients
        disc_X_grads = tape.gradient(disc_X_loss, self.seismic_discriminator.trainable_variables)
        disc_y_grads = tape.gradient(disc_y_loss, self.ai_discriminator.trainable_variables)

        # Apply those gradients
        self.d_optimizer.apply_gradients(
            zip(disc_X_grads, self.seismic_discriminator.trainable_variables)
        )
        self.d_optimizer.apply_gradients(
            zip(disc_y_grads, self.ai_discriminator.trainable_variables)
        )


        with tf.GradientTape(persistent=True) as tape:
            fake_X = self.seismic_generator(real_X)
            fake_y = self.ai_generator(real_X)
            X_predictions = self.seismic_discriminator(fake_X)
            y_predictions = self.ai_discriminator(fake_y)

            misleading_X_truth   = tf.zeros((batch_size, 1))
            misleading_y_truth   = tf.zeros((batch_size, 1))

            # Generator loss
            gX_loss = self.g_loss(real_X, fake_X)
            gy_loss = self.g_loss(real_y_2, fake_y)
            dX_loss = self.d_loss(misleading_X_truth, X_predictions)
            dy_loss = self.d_loss(misleading_y_truth, y_predictions)
            gen_X_loss = self.alpha*(dX_loss) + self.beta*(gX_loss)
            gen_y_loss = self.alpha*(dy_loss) + self.beta*(gy_loss)

        # Get the gradients
        gen_X_grads = tape.gradient(gen_X_loss, self.seismic_generator.trainable_variables)
        gen_y_grads = tape.gradient(gen_y_loss, self.ai_generator.trainable_variables)

        # Update the weights
        self.g_optimizer.apply_gradients(
            zip(gen_X_grads, self.seismic_generator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(gen_y_grads, self.ai_generator.trainable_variables)
        )

        return {
                'generator_X_loss'   : gen_X_loss,
                'generator_y_loss'   : gen_y_loss,
                'discriminator_X_loss': disc_X_loss,
                'discriminator_y_loss': disc_y_loss
                }
    
    def call(self, input):
        return [self.ai_generator(input), self.seismic_generator(input)]
    


class ResidualBlock(Layer):
    """
    If one would wish to write this as a class. Inspired by keras-tcn
    """
    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size,
                 padding: str,
                 activation: str = 'relu',
                 convolution_func: str = Conv2D,
                 dropout_type: str ='spatial',
                 dropout_rate: float = 0.,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs): # Any initializers for the Layer class
        """
        Creates a residual block for use in a TCN
        """
        # Checking whether dilations are a power of two
        assert (dilation_rate != 0) & ((dilation_rate & (dilation_rate - 1)) == 0), \
               'Dilations must be powers of 2'

        if convolution_func == Conv2D:
            self.dim = 2

            # Dilations only occur in depth; See Mustafa et al. 2021
            self.dilation_rate = (1, dilation_rate) # Height, width
        else:
            self.dim = 1
            self.dilation_rate = dilation_rate

        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer

        self.padding = padding
        self.activation = activation
        self.convolution_func = convolution_func # function for use in convolution layers
        self.dropout_type = dropout_type # Can be 'normal' or 'spatial'; decides what type of dropout layer is applied
        self.dropout_rate = dropout_rate

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm

        # Variables to be filled
        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        """
        Inspired by a function of the same name in another TCN implementation
        Not sure why input_shape is not used, but required as an input.
        """
        return [self.res_output_shape, self.res_output_shape]

    def _build_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)

        # This looks suspicious
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape) # Not sure if compute output shape does anything here

    def build(self, input_shape):

        with K.name_scope(self.name): # Gets the name from **kwargs
            
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = f'{self.convolution_func.__name__}_{k}'
                with K.name_scope(name):

                    # Check out inputs here
                    conv = self.convolution_func(
                                                 filters=self.nb_filters,
                                                 kernel_size=self.kernel_size,
                                                 dilation_rate=self.dilation_rate,
                                                 padding=self.padding,
                                                 name=name,
                                                 kernel_initializer=self.kernel_initializer
                    )
                    if self.use_weight_norm:
                        from tensorflow_addons.layers import WeightNormalization

                        # WeightNormalization API is different than other Normalizations; requires wrapping
                        with K.name_scope('norm_{}'.format(k)):
                            conv = WeightNormalization(conv)
                    self._build_layer(conv)

                # Other Normalization types
                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._build_layer(BatchNormalization())
                    elif self.use_layer_norm:
                        self._build_layer(LayerNormalization())
                    elif self.use_weight_norm:
                        pass # Already done above
                
                with K.name_scope('act_and_dropout_{}'.format(k)):
                    if self.dropout_type == 'normal':
                        d_func = Dropout
                        dname = 'Dropout'
                    if self.dropout_type == 'spatial':
                        dname = 'SDropout'
                        if self.dim == 1:
                            d_func = SpatialDropout1D
                        elif self.dim == 2:
                            d_func = SpatialDropout2D

                    self._build_layer(Activation(self.activation, name='Act_{}_{}'.format(self.convolution_func.__name__, k)))
                    self._build_layer(d_func(rate=self.dropout_rate, name='{}{}D_{}'.format(dname, self.dim, k)))
    
            if self.nb_filters != input_shape[-1]:
                # 1x1 convolution mathes the shapes (channel dimension).
                name = 'matching_conv'
                with K.name_scope(name):

                    self.shape_match_conv = self.convolution_func(
                        filters=self.nb_filters,
                        kernel_size=1,
                        padding='same',
                        name=name,
                        kernel_initializer=self.kernel_initializer # Why initialize this kernel with the same initializer?
                    )
            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)
            
            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)
            
            # Names of these layers should be investigated
            self._build_layer(Activation(self.activation, name='Act_Conv_Blocks'))
            self.final_activation = Activation(self.activation, name='Act_Res_Block')
            self.final_activation.build(self.res_output_shape) # According to philipperemy this probably is not be necessary

            # Forcing keras to add layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation) # I think this fixes the name issue

            super(ResidualBlock, self).build(input_shape) # This to make sure self.built is set to True

    def call(self, inputs, training=None, **kwargs):
        x1 = inputs
        for layer in self.layers:
            training_flag = 'traning' in dict(inspect.signature(layer.call).parameters)
            x1 = layer(x1, training=training) if training_flag else layer(x1)
        x2 = self.shape_match_conv(inputs)
        x1_x2 = self.final_activation(layers.add([x2, x1], name='Add_Res'))
        return [x1_x2, x1]


class TCN(Layer):
    """
    Creates a TCN layer.
    """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=(3, 9),
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_type = 'spatial',
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 convolution_func = Conv2D,
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 reduce_rows=False,
                 **kwargs):
        
        self.return_sequences = return_sequences
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.kernel_initializer = kernel_initializer
        self.nb_filters = nb_filters
        self.activation_name = activation
        self.convolution_func = convolution_func
        self.padding = padding

        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm

        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None

        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')
        
        if isinstance(self.nb_filters, list):
            assert len(self.nb_filters) == len(self.dilations) # Change of filter amount coincide with padding
            if len(set(self.nb_filters)) > 1 and self.use_skip_connections:
                raise ValueError('Skip connections are not compatible'
                                 'with a list of filters, unless they are all equal.')
        if padding != 'causal' and padding != 'same':
            raise ValueError('Only \'causal\' or \'same\' padding are compatible for this layer.')
        
        super(TCN, self).__init__(**kwargs)
    
    @property
    def receptive_field(self):
        return 1 + 2*(self.kernel_size-1)*self.nb_stacks*sum(self.dilations) # May need to pick the kernel dimension

    def build(self, input_shape):

        # Makes sure the i/o dims of each block are the same
        self.build_output_shape = input_shape

        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1 # A cheap way to do a false case for below
    
        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                if isinstance(self.nb_filters, list):
                    res_block_filters = self.nb_filters[i] 
                else:
                    res_block_filters = self.nb_filters
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=res_block_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          activation=self.activation_name,
                                                          convolution_func=self.convolution_func,
                                                          dropout_type=self.dropout_type,
                                                          dropout_rate=self.dropout_rate,
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          use_weight_norm=self.use_weight_norm,
                                                          kernel_initializer=self.kernel_initializer,
                                                          name='residual_block_{}'.format(len(self.residual_blocks))))

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        self.output_slice_index = None
        if self.padding == 'same':
            time = self.build_output_shape.as_list()[1]
            if time is not None:
                self.output_slice_index = int(self.build_output_shape.as_list()[1] / 2)
            else:
                # It will be known at call time
                self.padding_same_and_time_dim_unknown = True
        else:
            self.output_slice_index = -1 # causal case.
        self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, :], name='Slice_Output')
        self.slicer_layer.build(self.build_output_shape.as_list())

    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            batch_size = self.build_output_shape[0]
            batch_size = batch_size.value if hasattr(batch_size, 'value') else batch_size
            nb_filters = self.build_output_shape[-1]
            return [batch_size, nb_filters]
        else:
            # Compatibility tensorflow 1.x
            return [v.value if hasattr(v, 'value') else v for v in self.build_output_shape]

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        self.layers_outputs = [x]
        self.skip_connections = []
        for res_block in self.residual_blocks:
            try:
                x, skip_out = res_block(x, training=training)
            except TypeError: # backwards compatibility
                x, skip_out = res_block(K.cast(x, 'float32'), training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)
        
        if self.use_skip_connections:
            x = layers.add(self.skip_connections, name='Add_Skip_Connections')
            self.layers_outputs.append(x)
        
        if not self.return_sequences:
            # Case: time dimension is unknown. e.g. (bs, None, input_dim).
            if self.padding_same_and_time_dim_unknown:
                self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)
        return x
    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCN, self).get_config() # Non-recursive, uses Layer.get_config(); key names must be standardized
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_type'] = self.dropout_type
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation_name
        config['convolution_func'] = self.convolution_func
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config


class CNN(Layer):

    def __init__(self,
                nb_filters=64,
                kernel_size=3,
                nb_stacks=3,
                padding='collapse',
                activation='relu',
                convolution_func = Conv2D,
                kernel_initializer='he_normal',
                dropout_rate = 0.001,
                use_dropout = False,
                use_batch_norm=False,
                use_layer_norm=False,
                use_weight_norm=False,
                **kwargs):
        
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size

        # If collapse padding is used, then the kernel size must be odd, and stacks must be sufficient to collapse the input
        self.nb_stacks = nb_stacks



        self.padding = padding
        self.activation = activation
        self.convolution_func = convolution_func
        self.pad_func = ZeroPadding2D
        self.dim = 2
        self.collapse = (padding=='collapse')
        if self.collapse:
            self.padding='valid'
            if kernel_size%2==0:
                raise ValueError('Kernel size must be odd for collapse padding.')
            if nb_stacks < int((kernel_size-1)/2):
                raise ValueError('Number of stacks must be sufficient to collapse the input.')


        # Compute the shape the data must have before being fed to the first layer
        self.data_shape = None

        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout


        # Not sure if needed..
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm

        self.conv_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None


        if convolution_func.__name__ == 'Conv1D':
            self.dim = 1
            self.pad_func = ZeroPadding1D
            if padding=='collapse':
                raise ValueError('Collapse padding is not supported for 1D convolutions.')


        super(CNN, self).__init__(**kwargs)

        
        

    def build(self, input_shape):

        self.build_output_shape = input_shape
        self.conv_blocks = []

        for k in range(self.nb_stacks):

            if self.collapse:
                #  the first dimension
                self.conv_blocks.append(self.pad_func(padding=((0, 0), ((self.kernel_size-1)/2,(self.kernel_size-1)/2)), name='pad_{}'.format(len(self.conv_blocks))))
                self.build_output_shape = self.conv_blocks[-1].compute_output_shape(self.build_output_shape)
                self.__setattr__(self.conv_blocks[-1].name, self.conv_blocks[-1])
            for i, f in enumerate([self.nb_filters]):
                conv_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                self.conv_blocks.append(self.convolution_func(filters=conv_filters, 
                                                              kernel_size=self.kernel_size,
                                                              padding = self.padding,
                                                              activation=self.activation,
                                                              kernel_initializer=self.kernel_initializer,
                                                              name='convolution_layer_{}'.format(len(self.conv_blocks))))
        
        for layer in self.conv_blocks:
            self.__setattr__(layer.name, layer)


    def call(self, inputs, training=None, **kwargs):
        x = inputs
        self.layers_outputs = [x]
        for conv_block in self.conv_blocks:
            try:
                x = conv_block(x, training=training)
            except TypeError: # also backwards compatibiltiy
                x = conv_block(K.cast(x, 'float32'), training=training)
                self.layers_outputs.append(x)
        return x

    def get_config(self):
        config = super(CNN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['padding'] = self.padding
        config['activation'] = self.activation
        config['convolution_func'] = self.convolution_func
        config['kernel_initializer'] = self.kernel_initializer
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        return config


def compiled_TCN(training_data, config, **kwargs):
    """
    This function is to be called for initiating a model
    with provided configurations
    """
    nb_filters              = config['nb_filters']
    kernel_size             = config['kernel_size']
    nb_tcn_stacks           = config['nb_tcn_stacks']
    nb_reg_stacks           = config['nb_reg_stacks']
    nb_rec_stacks           = config['nb_rec_stacks']
    dilations               = config['dilations']
    padding                 = config['padding']
    use_skip_connections    = config['use_skip_connections']
    dropout_type            = config['dropout_type']
    dropout_rate            = config['dropout_rate']
    return_sequences        = config['return_sequences']
    activation              = config['activation']
    convolution_func        = config['convolution_func']
    learning_rate           = config['learn_rate']
    kernel_initializer      = config['kernel_initializer']
    use_batch_norm          = config['use_batch_norm']
    use_layer_norm          = config['use_layer_norm']
    use_weight_norm         = config['use_weight_norm']
    use_adversaries         = config['use_adversaries']

    batch_size              = config['batch_size']
    epochs                  = config['epochs']


    # Data
    X, y = training_data

    input_shape = tuple([*X.shape[1:], 1])
    input_layer = Input(shape=input_shape)

    # Feature Extraction module
    x = TCN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_tcn_stacks,
            dilations=dilations,
            padding=padding,
            use_skip_connections=use_skip_connections,
            dropout_type = dropout_type,
            dropout_rate=dropout_rate,
            return_sequences=return_sequences,
            activation=activation,
            convolution_func=convolution_func,
            kernel_initializer=kernel_initializer,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_weight_norm=use_weight_norm,
            name='Feature_exctraction_module'
    )(input_layer)
    
    #print('receptive field is: {}'.format(x.receptive_field()))

    # Regression module
    # reg_ksize = y[0].shape[-1]/(nb_reg_stacks) + 1  # for 1d preserving the shape of the data
    # reg_ksize = int(reg_ksize)
    reg = CNN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_reg_stacks,
            padding='same',
            activation=activation,
            convolution_func=convolution_func,
            kernel_initializer=kernel_initializer,
            name = 'Regression_module'
            )(x)   

    reg = convolution_func(1, kernel_size, padding=padding, activation='linear', name='regression_output')(reg)
    
    # Reconstruciton module
    rec = CNN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_rec_stacks,
            padding=padding,
            activation=activation,
            convolution_func=convolution_func,
            kernel_initializer=kernel_initializer,
            name = 'Reconstruction_module'
            )(x)


    rec = convolution_func(1, kernel_size, padding=padding, activation='linear', name='reconstruction_output')(rec)

    output_layer = [reg, rec] # Regression, reconstruction

    if use_adversaries:
        seis_gen_model = Model(inputs=input_layer, outputs=rec)
        ai_gen_model   = Model(inputs=input_layer, outputs=reg)
        seis_disc_model = discriminator(output_layer[1].shape[1:], 3, name='seismic_discriminator')
        ai_disc_model = discriminator(output_layer[0].shape[1:], 3, name='ai_discriminator')


        model = multi_task_GAN([ai_disc_model, seis_disc_model],
                               [ai_gen_model, seis_gen_model], 
                               alpha=config['alpha'],
                               beta=config['beta'])

        generator_loss = keras.losses.MeanSquaredError()
        discriminator_loss = keras.losses.BinaryCrossentropy()

        generator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.)
        discriminator_optimizer = keras.optimizers.Adam(learning_rate=learning_rate*0.1, clipnorm=1.) # Discriminators learn more slowly

        model.compile(g_optimizer=generator_optimizer, 
                      d_optimizer=discriminator_optimizer, 
                      g_loss=generator_loss, 
                      d_loss=discriminator_loss)
        # model.summary()
    else:
        model = Model(inputs = input_layer, 
                  outputs = output_layer)
        model.compile(keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.), loss={'regression_output' : 'mean_squared_error',
                                                                           'reconstruction_output' : 'mean_squared_error'})
        model.summary()

    History = model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, **kwargs)
    
    # Tree model here
    #model.ai_generator.



    return model, History

    
    
def model_loss(y_true, y_pred):
    loss = tf.reduce_mean(tf.norm((y_pred-y_true), ord=2))
    return loss

def weight_share_loss(y_true, y_pred):
    """
    As described in Mustafa et al. (2021)
    """
    yt_reg, yt_recon = y_true[0], y_true[1]
    yp_reg, yp_recon = y_pred[0], y_pred[1]
    recon_loss = model_loss(yt_recon, yp_recon)
    reg_loss = model_loss(yt_reg, yp_reg)
    total_loss = reg_loss + recon_loss
    return total_loss

def discriminator(Input_shape, 
                  depth = 4, 
                  convolution_func=Conv1D, 
                  dropout = 0.1, 
                  name='discriminator'):
    """
    Descriminator model for use in adversarial learning
    """
    input_layer = Input(Input_shape)
    x = input_layer
    for _ in range(depth):
        x = convolution_func(1, kernel_size=4, padding='valid')(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = Dropout(rate = dropout)(x)
        x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    output_score = Dense(1, activation='sigmoid')(x)
    return Model(input_layer, output_score, name=name)


####################### From Feat_aug.py ############################

def get_traces(fp, mmap=True, zrange: tuple = (None, 100)):
    """
    This function should conserve some information about the domain (time or depth) of
    the data.
    """
    with segyio.open(fp, ignore_geometry=True) as seis_data:
        z = seis_data.samples
        if mmap:
            seis_data.mmap()  # Only to be used if the file size is small compared to available memory
        traces = segyio.collect(seis_data.trace)
    
    traces, z = traces[:, z<zrange[1]], z[z<zrange[1]]
    return traces, z


def get_matching_traces(fp_X, fp_y, mmap = True, zrange: tuple = (25, 100), group_traces: int = 1, trunc = False):
    """
    Function to assist in maintaining cardinality of the dataset
    %%%%%%%%%%%%%%%% Can add overlap here
    """
    assert group_traces%2, 'Amount of traces must be odd to have a center trace'

    with segyio.open(fp_X, ignore_geometry=True) as X_data:
        with segyio.open(fp_y, ignore_geometry=True) as y_data:
            # retrieving depth values for target and input data
            z_X = X_data.samples
            z_y = y_data.samples

            # getting index of max depth for truncation
            X_max_idx = amax(where(z_X < zrange[1])) + 1
            y_max_idx = amax(where(z_y < zrange[1])) + 1

            # The acoustic impedance starts at depth 25m
            X_min_idx = amin(where(z_X >= z_y[0]))


            if mmap: X_data.mmap(); y_data.mmap() # initiate mmap mode for large datasets

            # get information about what traces are overlapping
            nums_X = segyio.collect(X_data.attributes(segyio.TraceField.CDP))
            nums_y = segyio.collect(y_data.attributes(segyio.TraceField.CDP))
            CDP, idx_X, idx_y = intersect1d(nums_X, nums_y, return_indices=True)
            assert len(idx_X) == len(idx_y)

            # collect the data
            X_traces = segyio.collect(X_data.trace)[idx_X, X_min_idx:X_max_idx]
            y_traces = segyio.collect(y_data.trace)[idx_y, :y_max_idx]

            X_func = interp1d(z_X[X_min_idx:X_max_idx], X_traces, kind='cubic', axis=1)
            X_traces = X_func(z_y[:y_max_idx])
 
            # y_refl, slope = ai_to_reflectivity(y_traces)
            # y_interp = interp1d(z_y[:y_max_idx], y_refl, kind='nearest', axis=1)
            # y_interp_refl = array(y_interp(z_X[X_min_idx:X_max_idx]))
            # y_traces = reflectivity_to_ai(y_interp_refl, slope)

    if not group_traces == 1:
        num_traces = X_traces.shape[0]
        len_traces = X_traces.shape[1]
        num_images = num_traces//group_traces
        indices_truncated = num_images*group_traces
        discarded_images = num_traces-indices_truncated
        l_indices = (discarded_images//2); r_indices = indices_truncated + l_indices + discarded_images%2
        X_traces = X_traces[l_indices:r_indices].reshape((num_images, group_traces, len_traces))
        y_traces = y_traces[l_indices:r_indices].reshape((num_images, group_traces, len_traces))
    
    if trunc:  # Done as a quick way to remove bad data, as it is most often at the ends
        X_traces = X_traces[trunc:-trunc]
        y_traces = y_traces[trunc:-trunc]
    
    return X_traces, y_traces, (z_X, z_y)


def sgy_to_keras_dataset(X_data_label_list,
                         y_data_label_list,
                         test_size=0.2, 
                         group_traces = 1,
                         zrange: tuple = (None, 100), 
                         reconstruction = True,
                         validation = False, 
                         X_normalize = None,
                         y_normalize = 'MinMaxScaler',
                         random_state=1,
                         shuffle=True,
                         min_y = 0.,
                         fraction_data=False,
                         truncate_data=False):
    """
    random_state may be passed for recreating results
    """
    data_dict = load_data_dict()

    # Something to evaluate that z is same for all in a feature
    X = array([]); y = array([])

    for i, key in enumerate(X_data_label_list):
        x_dir = Path(data_dict[key])
        y_dir = Path(data_dict[y_data_label_list[i]])
        matched = match_files(x_dir, y_dir)
        if fraction_data: matched = matched[:int(len(matched)*fraction_data)] # %%%%%%%%%%%%%%%%%%%%%quickfix
        m_len = len(matched)
        for i, (xm, ym) in enumerate(matched):
            # Giving feedback to how the collection is going
            sys.stdout.write('\rCollecting trace data into dataset {}/{}'.format(i+1, m_len))
            sys.stdout.flush()

            try: x_traces, y_traces, z = get_matching_traces(xm, ym, zrange=zrange, group_traces=group_traces, trunc=truncate_data)
            except: print('\nCould not load file {}\n'.format(xm)); continue

            if not len(X):
                X = array(x_traces)
                y = array(y_traces)
            else:
                X = row_stack((X, x_traces))
                y = row_stack((y, y_traces))
    sys.stdout.write('\n'); sys.stdout.flush()
    
    y[where(y<min_y)] = min_y

    X_scaler = None
    y_scaler = None
    # Normalization
    if X_normalize == 'MinMaxScaler':
        X_scaler = MinMaxScaler()
        X_new = X_scaler.fit_transform(X.reshape(-1, 1))
        X = X_new.reshape(X.shape)
    elif X_normalize == 'StandardScaler':
        X_scaler = StandardScaler()
        X_new = X_scaler.fit_transform(X.reshape(-1, 1))
        X = X_new.reshape(X.shape)
    if y_normalize == 'MinMaxScaler':
        y_scaler = MinMaxScaler()
        y_new = y_scaler.fit_transform(y.reshape(-1, 1))
        y = y_new.reshape(y.shape)


    train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state,
                                                        shuffle=shuffle)  # dataset must be np.array
    
    if validation:
        test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        if reconstruction:
            train_y = [train_y, train_X]
            test_y = [test_y, test_X]
            val_y = [val_y, val_X]
        return (train_X, train_y), (test_X, test_y), (val_X, val_y)
    
    if reconstruction:
        train_y = [train_y, train_X]
        test_y = [test_y, test_X]
    return (train_X, train_y), (test_X, test_y), (X_scaler, y_scaler)


def match_files(X_folder_loc, y_folder_loc, file_extension='.sgy'):
    """
    Matches the features from two seperate traces by filename
    Filenames corresponding to each other are returned as a list
    of tuples in the form [(X_file, y_file)]
    """
    X_dir = list(Path(X_folder_loc).glob('*'+file_extension)); y_dir = list(Path(y_folder_loc).glob('*'+file_extension))

    file_pairs = []
    for i, fname in enumerate(X_dir):
        X_name = fname.name[:fname.name.find('_MIG_DPT.sgy')]
        j_list = []
        for j, yfile in enumerate(y_dir):
            y_name = yfile.name[:yfile.name.find('_MIG.Abs_Zp.sgy')]
            if X_name == y_name:
                file_pairs.append((str(fname), str(yfile)))
                j_list.append(j)
        [y_dir.pop(j) for j in j_list]
    return file_pairs


def get_seismic_where_there_is_cpt(cpt, z_cpt, seis, z_seis):
    """
    For a given cpt trace, this function gives a pair of seismic image and cpt for a continuous range where all cpt parameters are defined.
    """
    # Get the indices of the cpt trace where all parameters are defined
    idx = np.where(~np.isnan(cpt).any(axis=1))[0]
    # Get the corresponding cpt depth
    z_cpt = z_cpt[idx]
    # Get the corresponding cpt trace
    cpt = cpt[idx]
    # Get the corresponding seismic trace
    seis = seis[idx]
    # Get the corresponding seismic depth
    z_seis = z_seis[idx]

    return cpt, z_cpt, seis, z_seis


def load_data_dict():
    data_json = './Data/data.json'
    with open(data_json, 'r') as readfile:
        data_dict = json.loads(readfile.read())
    return data_dict


def update_data_dict():
    """ Edit this funciton to change the filepaths to the relevant data
        Filepaths must be relative to the repository, which is in user folder.
        Double dot (..) steps outside of this folder to access the OneDrive
        folder
    """
    data_json = './Data/data.json'
    root = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/'
    data_dict = {
        '00_AI'                 : root + '00_AI',
        '2DUHRS_06_MIG_DEPTH'   : root + '2DUHRS_06_MIG_DEPTH'
    }
    with open(data_json, 'w') as writefile:
        writefile.write(json.dumps(data_dict, indent=2))


def find_nth(haystack, needle, n : int):
    n = abs(n); assert n > 0
    max_needle_amount = len(haystack.split(needle)); assert n < max_needle_amount
    if n-1:
        intermed = haystack.find(needle) + 1
        loc = intermed + find_nth(haystack[intermed:], needle, n-1)
    else:
        return haystack.find(needle)
    return loc


def find_duplicates(m_files):
    dupes = dict()
    names = []
    for i, (X_m, y_m) in enumerate(m_files):

        name = X_m[62:-12]
        name_list = name.split('_')
        for n in names:
            n_list = n.split('_')
            if (len(name_list)>3) and (len(n_list)>3):
                if n_list[:3] == name_list[:3]:
                    key = '_'.join(n_list[:3])
                    if not (key in dupes.keys()):
                        dupes[key] = [X_m[:62] + '_'.join(n_list) + X_m[-12:], X_m]
                    else:
                        dupes[key].append(X_m)
        names.append(name)
    return dupes


def box_plots_for_duplicates():
    import matplotlib.pyplot as plt
    import numpy as np

    d_dict = load_data_dict()
    m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    lines = []
    dupe_names = []
    dupe_labels = []
    dupe = find_duplicates(m_files=m_files)
    for key, val in dupe.items():
        trc = []
        lbls = []
        for file in val:
            trace, _ = get_traces(file, zrange=(25, 100))
            trc.append(trace.flatten())
            lbls.append(file[62:-12])
        lines.append(trc)
        dupe_names.append(key)
        dupe_labels.append(lbls)
    
    for name, labels, collection in zip(dupe_names, dupe_labels, lines):
        plt.clf()
        plt.boxplot(collection, labels=labels)
        plt.title(name)
        plt.xticks(rotation=10)
        plt.savefig('Data/dupelicates/{}.png'.format(name))



def img_plots_for_dupelicates():
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    d_dict = load_data_dict()
    m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    lines = []
    dupe_names = []
    dupe_labels = []
    dupe = find_duplicates(m_files=m_files)
    for key, val in dupe.items():
        trc = []
        lbls = []
        for file in val:
            trace, _ = get_traces(file, zrange=(25, 100))
            trc.append(trace.T)
            lbls.append(file[62:-12])
        lines.append(trc)
        dupe_names.append(key)
        dupe_labels.append(lbls)
    
    for name, labels, collection in zip(dupe_names, dupe_labels, lines):
        plt.clf()
        fig, ax = plt.subplots(len(labels))
        fig.tight_layout(h_pad=1)
        for i, im in enumerate(collection):
            norm = Normalize(-2, 2)
            ax[i].imshow(im, cmap = 'seismic', norm=norm)
            ax[i].set_title(label=labels[i], fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.savefig('Data/dupelicates/Image_{}.png'.format(name))


def negative_ai_values():
    import matplotlib.pyplot as plt
    import numpy as np
    from random import randint
    d_dict = load_data_dict()
    m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    low_val = 0
    # y_below_zero = []
    t_b_z = []
    for file, i in m_files:
        # _, z = get_traces(file, zrange=(25, 100))
        traces, z_ai = get_traces(i, zrange=(25, 100))
        t = traces[np.where(np.any((traces<low_val), axis=1)), :]
        t_b_z+=list(t[0])
        # y_below_zero.append(t_b_z)
    # plt.hist(y_below_zero)
    print(np.shape(t_b_z))
    r = randint(0, len(t_b_z))
    plt.plot(t_b_z[r], z_ai)
    print('Minimum on the plot is: {}'.format(np.min(t_b_z[r])))
    print('Total minimum is {}'.format(np.min(t_b_z)))
    plt.show()
