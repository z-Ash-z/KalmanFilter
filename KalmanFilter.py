import numpy as np
from matplotlib import pyplot as plt


class KalmanFilter():
    """
    Kalman Filter using a constant acceleration model.
    """

    def __init__(self, dt : float, u : float, sigma_a : float, sigma_z : float) -> None:
        self.dt = dt
        self.u = u
        self.sigma_a = sigma_a
        self.sigma_z = sigma_z
        
        self.A = np.matrix([[1, self.dt], 
                            [0, 1]])
        
        self.B = np.matrix([[(1/2) * self.dt**2],
                            [self.dt]])
        
        self.C = np.matrix([1, 0])

        self.R = np.matrix([[(self.dt**4) / 4, (self.dt**3) / 2],
                            [(self.dt**3) / 2, (self.dt**2)]]) * self.sigma_a**2
        
        self.Q = self.sigma_z**2

        self.I = np.eye(self.C.shape[1])

        self.x = np.matrix([[1000],
                            [50]])
        
        self.P = np.eye(self.A.shape[1])


    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.R

    def update(self, z : float) -> np.ndarray:
        S = np.dot(self.C, np.dot(self.P, self.C.T)) + self.Q
        K = np.dot(self.P, np.dot(self.C.T, np.linalg.pinv(S)))
        self.x = self.x + np.dot(K, (z - np.dot(self.C, self.x)))
        self.P = np.dot((self.I - np.dot(K, self.C)), self.P)
        return self.x

def main():
    
    # The time step of the sensor.
    dt = 0.1 

    # The time steps.
    t = np.arange(0, 100, dt) 

    # The real value or the ground truth.
    real_x = dt * ((t**2) - t) 

    u = 1 # Control Data.
    sigma_a = 0.25 # Process noise.
    sigma_z = 2 # Sensor noise.
    kf = KalmanFilter(dt, u, sigma_a, sigma_z)

    predictions = list()
    measurements = list()

    for x in real_x:
        # For every dt, we first predict the value using the Kalman filter 
        # and then update the filter using the measurement.
        
        # Simulating the sensor reading with noise.
        z = kf.C.item(0) * x + np.random.normal(0, 50)
        measurements.append(z)
        
        kf.predict()
        x_kf = kf.update(z).item(0)
        
        predictions.append(x_kf)

    plt.figure()
    plt.plot(t, measurements, label='Measurements', color='b', linewidth=0.5)
    plt.plot(t, np.array(real_x), label='Real Track', color='y', linewidth=1.5)
    plt.plot(t, predictions, label='Kalman Filter Prediction', color='r', linewidth=1.5)
    plt.show()

if __name__ == "__main__":
    main()
