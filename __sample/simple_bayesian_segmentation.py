import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class BayesianDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, prior_scale=1.0, name=None):
        super(BayesianDenseLayer, self).__init__(name=name)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.prior_scale = prior_scale
    
    def build(self, input_shape):
        self.w_mean = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='glorot_uniform',
                                      trainable=True,
                                      name='w_mean')
        
        self.w_stddev = self.add_weight(shape=(input_shape[-1], self.units),
                                        initializer=tf.keras.initializers.Constant(value=0.5),
                                        trainable=True,
                                        name='w_stddev')
        
        self.b_mean = self.add_weight(shape=(self.units,),
                                      initializer='zeros',
                                      trainable=True,
                                      name='b_mean')
        
        self.b_stddev = self.add_weight(shape=(self.units,),
                                        initializer=tf.keras.initializers.Constant(value=0.5),
                                        trainable=True,
                                        name='b_stddev')
        
        self.prior = tfd.Independent(tfd.Normal(loc=0., scale=self.prior_scale),
                                     reinterpreted_batch_ndims=1)
    
    def call(self, inputs):
        w = tfd.Normal(loc=self.w_mean, scale=tf.math.softplus(self.w_stddev))
        b = tfd.Normal(loc=self.b_mean, scale=tf.math.softplus(self.b_stddev))
        self.add_loss(tf.reduce_sum(self.prior.log_prob(w)) + tf.reduce_sum(self.prior.log_prob(b)))
        outputs = tf.matmul(inputs, w.sample()) + b.sample()
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs
    
    
class BayesianSegmentationModel(tf.keras.Model):
    def __init__(self, input_shape, num_classes, prior_scale=1.0):
        super(BayesianSegmentationModel, self).__init__()
        self.prior_scale = prior_scale
        
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape)
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = BayesianDenseLayer(256, activation='relu', prior_scale=prior_scale)
        self.dense2 = BayesianDenseLayer(num_classes, activation='softmax', prior_scale=prior_scale)
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        outputs = self.dense2(x)
        return outputs
    
    
model = BayesianSegmentationModel(input_shape=(256, 256, 3), num_classes=2, prior_scale=1.0)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy()])


model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))
