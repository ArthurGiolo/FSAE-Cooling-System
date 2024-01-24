import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numba import njit

#### Valores iniciais:
c_H20 = 4186 # J/kg.K
c_ar  = 1019 # J/kg.K
T     = 40 + 273 # Temperatura Inicial 
T_ar  = 40 + 273 # Temperatura Ambiente
U     = [15,20,25,30,35,40,45,50,55,60,65,70] # Coeficiente de transferência global de calor 
A     = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,1,1.5,2,2.5,3,4,5,6] # Area frontal
UA    = np.zeros((len(U),len(A))) # Matriz de temperaturas máximas em provas

#### Retirar os valores do OptimumLap (Velocidade, Rotação e Raio de curva):
optimum_vals = pd.read_csv("C:/Users/arthu/Desktop/1 - Radiador/1 - Codigo Rads/Dados Optimum/Enduro_2019.csv")

## Criar vetor de velocidade:
vel    = np.array(optimum_vals['km/h'],dtype=np.float64)

## Criar vetor de rotação do motor:
rpm    = np.array(optimum_vals['rpm'],dtype=np.float64)

## Criar vetor de raio de curva:
radius = np.array(optimum_vals['m'],dtype=np.float64)

#### Funções para converter valores:
@njit
def m_H20(RPM):
    return 0.0000563076*RPM - 0.108393 # kg/s

@njit
def m_ar(vel, A):
    mps = vel/3.6
    rho = 1.225 # kg/m3
    return rho*mps*A

@njit
def Q_ponto(RPM):
    return (0.000990973*RPM**2 + 4.46993*RPM - 2533.92)/20 # J/s

@njit
def entalpia(T):
    return 0.345946*(T-273) + 4152.73*(T-273) + 1001.14 # J/kg

@njit
def temperatura(H):
    return (0.00023783*H + 0.0509005) + 273 # Kelvin


def otimiza(T,vel,rpm,U,A):
    for j in range(len(U)):
        for k in range(len(A)):
            T_max = 0        # Temperatura máxima do motor
            for i in range(len(vel)):
                ########### 1 - Calor entrando no motor ###########
                T = Q_ponto(rpm[i])/(m_H20(rpm[i])*c_H20) + T
                if T > T_max:
                    T_max = T

                ########### 1 - Radiador ###########

                Cf  = m_ar(vel[i], A[k])*c_ar
                Cq  = m_H20(rpm[i])*c_H20

                # Determimando troca de calor máxima (método da efetividade)
                if Cf < Cq:
                    C     = Cf/Cq
                    NTU  = (U[j]*A[k])/Cf
                    Q_max = Cf*(T - T_ar)
                else:
                    C     = Cq/Cf
                    NTU   = (U[j]*A[k])/Cq
                    Q_max = Cq*(T - T_ar)

                # Taxa de calor cedido pelo Radiador 
                e     = 1 - np.exp((NTU**0.22/C)*(np.exp(-C*NTU**0.78) - 1))
                Q     = e*Q_max

                # Temperatura na saída do Radiador 
                T = T - Q/(m_H20(rpm[i])*c_H20)

            if T_max >= 500:
                UA[j,k] = 500 - 273
            else:
                UA[j,k] = T_max - 273

    return UA


eixos = otimiza(T,vel,rpm,U,A)
fig,ax = plt.subplots()
CS = ax.contourf(A,U,eixos)
ax.clabel(CS, inline = True, fontsize = 10, colors = 'black')
plt.title('Temperatura máxima com um radiador')
plt.ylabel('Coeficiente de transferência global (W/m2.K)')
plt.xlabel('Área (m2)')
plt.show()
