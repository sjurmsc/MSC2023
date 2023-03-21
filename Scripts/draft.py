"""
Draft file containing code that has not been used for final results
"""

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
    


    # ML models

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
    