import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 1. Definición de parámetros

fs = 1000
t = np.linspace(0, 1, fs)

fm = 5
fc = 50
Am = 1
Ac = 1


# 2. Señal de mensaje

mensaje = Am * np.sin(2 * np.pi * fm * t)


# 3. Señal portadora

portadora = Ac * np.cos(2 * np.pi * fc * t)


# 4. Modulación AM

senal_modulada = (1 + mensaje) * portadora


# 5. Ruido (ruido blanco gaussiano)

ruido = 0.3 * np.random.randn(len(t))
senal_ruidosa = senal_modulada + ruido


# 6. Atenuación

senal_atenuada = 0.5 * senal_modulada


# 7. FFT (Frecuencia)

def calcular_fft(signal):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)
    return xf, np.abs(yf)


# 8. Gráficas

plt.figure(figsize=(12,10))

# Señal mensaje
plt.subplot(4,1,1)
plt.plot(t, mensaje)
plt.title("Señal de Mensaje")
plt.xlabel("Tiempo")
plt.ylabel("Amplitud")

# Señal modulada
plt.subplot(4,1,2)
plt.plot(t, senal_modulada)
plt.title("Señal AM")

# Señal con ruido
plt.subplot(4,1,3)
plt.plot(t, senal_ruidosa)
plt.title("Señal con Ruido")

# Señal atenuada
plt.subplot(4,1,4)
plt.plot(t, senal_atenuada)
plt.title("Señal Atenuada")

plt.tight_layout()
plt.show()


# 9. FFT de señal modulada

xf, yf = calcular_fft(senal_modulada)

plt.figure()
plt.plot(xf, yf)
plt.title("Dominio de Frecuencia - Señal AM")
plt.xlabel("Frecuencia")
plt.ylabel("Magnitud")
plt.xlim(0,100)
plt.show()
