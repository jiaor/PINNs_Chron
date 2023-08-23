import tensorflow as tf
import numpy as np
import datetime
from TF_Mad_He import solve_age
from TF_Ketch07 import solve_ft_age

# Set data type
DTYPE='float32'
tf.keras.backend.set_floatx(DTYPE)
tf.get_logger().setLevel('ERROR')

##############################
def trapz(y, x):
    dx = (x[-1] - x[0]) / (int(x.shape[0]) - 1)
    return ((y[0] + y[-1])/2 + tf.reduce_sum(y[1:-1])) * dx
    
class PINN1D(tf.keras.Model):
    """ Set basic architecture of the PINN model."""
    def __init__(self, lb, ub, 
            output_dim=1,
            num_hidden_layers=3,
            num_neurons_per_layer=20,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)
        
        #default values for bottom and surface temperatures and model thickness
        
        self.Tbot = 500.
        self.Tsurf = 0.
        self.h = 20.

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lb = lb
        self.ub = ub
        
        # Define NN architecture
        # X is defined as (time, z)
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        
        self.transform = tf.keras.layers.Lambda(
            lambda x: x[:, 0:1] * (self.Tbot - self.Tgrad() * x[:, 1:2])   
        )
    
        self.out = tf.keras.layers.Dense(output_dim)
        
    def Tgrad(self):
        return (self.Tbot - self.Tsurf) / self.h
        
    def call(self, X):
        """Forward-pass through NN."""
        Y = self.scale(X)
        for i in range(self.num_hidden_layers):
            Y = self.hidden[i](Y)
            
        Y = self.transform(Y)
        return self.out(Y)

##############################
# Define model architecture
class PINN3D(PINN1D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # X is defined as (time, x, y, z)
        self.transform = tf.keras.layers.Lambda(
            lambda x: x[:, 0:1] * (self.Tbot - self.Tgrad(x[:, 0:1], x[:, 1:2], x[:, 2:3]) * x[:, 3:4])   
        )
        
    def Tgrad(self, x, y, z):
        raise NotImplementedError('geothermal gradient function needs to be defined')

##############################
class PINN1DSolver():
    def __init__(self, model, X_r):
        self.model = model
        self.start_time = datetime.datetime.now()
        self.runname = self.start_time.strftime("%y%m%d_%H%M")
        self.savepath = 'saved_model'
        
        #default values for kappa and uplift rate
        self.kappa = 25.
        self.uplift = .1 
        
        # Store collocation points
        self.t = X_r[:,0:1]
        self.z = X_r[:,1:2]
        
        # Initialize history of losses and global iteration counter
        self.loss_hist = []
        self.heat_hist = []
        self.bound_hist = []
        self.iter = 0
        
        # Define matrix for computing ages
        n=50
        self.D = tf.Variable(np.ones(n), shape=(n,), dtype=DTYPE)
        self.L = tf.Variable(np.zeros(n), shape=(n,), dtype=DTYPE)
        self.U = tf.Variable(np.zeros(n), shape=(n,), dtype=DTYPE)
        self.xf = tf.Variable(np.zeros(n), shape=(n,), dtype=DTYPE)    
    
    def get_r(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables representing t and z during this GradientTape
            tape.watch(self.t)
            tape.watch(self.z)
            
            # Compute current values T(t,x)
            T = self.model(tf.stack([self.t[:,0], self.z[:,0]], axis=1))
            T_z = tape.gradient(T, self.z)
            
        T_t = tape.gradient(T, self.t)
        T_zz = tape.gradient(T_z, self.z)
        
        del tape
        return self.fun_r(self.t, self.z, T, T_t, T_z, T_zz)
    
    def loss_fn(self, X, T):
        phi_heat = self.heat_loss()
        phi_bound = self.bound_loss(X, T)
        loss = tf.reduce_sum([phi_heat, phi_bound])
        return loss, phi_heat, phi_bound
        
    def heat_loss(self):
        # Compute phi_r
        r = self.get_r()
        return tf.reduce_mean(tf.square(r))
    
    def bound_loss(self, X, T):
        T_pred = self.model(X)
        return tf.reduce_mean(tf.square(T - T_pred))
        
    def get_grad(self, X, T):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.model.trainable_variables)
            loss, phi_heat, phi_bound = self.loss_fn(X, T)
            
        g = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return loss, g, phi_heat, phi_bound

    def fun_u(self, t):
        return self.uplift
    
    def fun_r(self, t, z, T, T_t, T_z, T_zz):
        # Residual of the PDE, i.e., heat function
        u = self.fun_u(t)
        return T_t + u * T_z - self.kappa * T_zz
    
    def get_tT(self):
        sample_t = tf.linspace(self.model.lb[0], self.model.ub[0], 100)
        u = self.fun_u(sample_t)
        depth = []
        for i in range(len(sample_t)):
            d = trapz(u[i:], sample_t[i:])
            depth.append(d)
    
        depth[-1] = 0.
        z = tf.convert_to_tensor(self.model.ub[1] - depth)
        sample_T = tf.reshape(self.model(tf.stack([sample_t, z], axis=1)), sample_t.shape)
        return sample_t, sample_T
    
    def pred_age(self, sample_t, sample_T, method='AHe', grain_radius=80):
        if method=='AFT':
            age = solve_ft_age(sample_t, sample_T, dpar=2.)
        else:
            age, _ = solve_age(self.model.ub[0]-sample_t, sample_T, self.D, self.L, self.U, self.xf,
                               method=method, beta=2, grain_radius=grain_radius)
        return age

    def solve_with_Adam(self, optimizer, X, T, N=1000, echofreq=100, savefreq=1000):
        """This method performs a gradient descent type optimization."""
        @tf.function
        def train_step():
            loss, g, phi_heat, phi_bound = self.get_grad(X, T)
            # Perform gradient descent step
            optimizer.apply_gradients(zip(g, self.model.trainable_variables))
            return loss, phi_heat, phi_bound
        
        for i in range(N):
            loss, phi_heat, phi_bound = train_step()
            self.current_loss = loss.numpy()
            self.phi_heat = phi_heat.numpy()
            self.phi_bound = phi_bound.numpy()
            self.callback(echofreq=echofreq, savefreq=savefreq)
                 
    def callback(self, xr=None, echofreq=100, savefreq=1000):
        self.iter += 1
        self.loss_hist.append(self.current_loss)
        self.heat_hist.append(self.phi_heat)
        self.bound_hist.append(self.phi_bound)
        
        if self.iter % echofreq == 0:
            print('{} It {:5d}: loss = {:10.8e}: run time = {:g} seconds'.format(
                        self.runname, self.iter, self.current_loss,
                (datetime.datetime.now()-self.start_time).seconds))
            if self.iter % savefreq == 0:
                path = self.savepath + '/'+ self.runname + '/' + str(self.iter) + '/'    
                tf.saved_model.save(self.model, path)
                np.save(self.savepath + '/'+ self.runname + '/loss_hist', self.loss_hist)
                np.save(self.savepath + '/'+ self.runname + '/heat_hist', self.heat_hist)
                np.save(self.savepath + '/'+ self.runname + '/bound_hist', self.bound_hist)
        
##############################
class PINN3DSolver(PINN1DSolver):    
    def __init__(self, model, X_r):
        super().__init__(model, X_r,)
        self.savepath = 'saved_model3d'
        # Store collocation points
        self.t = X_r[:, 0:1]
        self.x = X_r[:, 1:2]
        self.y = X_r[:, 2:3]
        self.z = X_r[:, 3:4]
    
    def get_r(self):
        with tf.GradientTape(persistent=True) as tape:
            # Watch variables during this GradientTape
            tape.watch(self.t)
            tape.watch(self.x)
            tape.watch(self.y)
            tape.watch(self.z)
            
            # Compute current values T(t, x, y, z)
            T = self.model(tf.stack([self.t[:,0], self.x[:,0], self.y[:,0], self.z[:,0]], axis=1))
            T_x = tape.gradient(T, self.x)
            T_y = tape.gradient(T, self.y)
            T_z = tape.gradient(T, self.z)
            
        T_t = tape.gradient(T, self.t)
        T_xx = tape.gradient(T_x, self.x)
        T_yy = tape.gradient(T_y, self.y)
        T_zz = tape.gradient(T_z, self.z)
        
        del tape
        return self.fun_r(self.t, self.x, self.y, self.z,
                          T, T_t, T_x, T_y, T_z, T_xx, T_yy, T_zz)
    
    def fun_r(self, t, x, y, z, T, T_t, T_x, T_y, T_z, T_xx, T_yy, T_zz):
        # Residual of the PDE, i.e., heat function
        u = self.fun_u(t)
        return T_t + u * T_z - self.kappa * (T_xx + T_yy + T_zz)
    
    def fun_h(self, t, x0, y0):
        return tf_h_fn(t, [[x0]], [[y0]])[0]
    
    def get_tT(self, x0, y0):        
        sample_t = tf.linspace(self.model.lb[0], self.model.ub[0], 100)
        u = self.fun_u(sample_t)
        x_ar = tf.ones_like(u) * x0
        y_ar = tf.ones_like(u) * y0
        depth = []
        for i in range(len(sample_t)):
            d = trapz(u[i:], sample_t[i:])
            depth.append(d)
    
        depth[-1] = 0.
        z = self.fun_h(sample_t[-1], x0, y0) - depth
        sample_T = tf.reshape(self.model(tf.stack([sample_t, x_ar, y_ar, z], axis=1)), sample_t.shape)
        return sample_t, sample_T