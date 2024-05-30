from __future__ import division
import numpy as np
import time
import random
import math


np.random.seed(1234)


#RIS setting
#RIS_position = []
RIS_phase_numbers = 8
n_elements_total = 12
n_elements = 4

position_RIS = [290, 375]
BS_position = [180, 270]

class V2Vchannels:
    # Simulator of the V2V Channels

    def __init__(self):
        self.t = 0  # 开始时间
        self.h_bs = 1.5  # 车辆的高度
        self.h_ms = 1.5  # 车辆的高度
        self.fc = 2  # 中心频率
        self.decorrelation_distance = 10
        self.shadow_std = 3

    def get_path_loss(self, position_A, position_B):
        d1 = abs(position_A[0] - position_B[0])
        d2 = abs(position_A[1] - position_B[1])
        d = math.hypot(d1, d2) + 0.001   #求两个位置之间的距离
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)

        #路径损失模型：PL = Alog10(d[m])+B+Clog10(fc[GHz]/5)+X
        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(self.h_ms) + 2.7 * np.log10(self.fc / 5)

        def PL_NLos(d_a, d_b):
            n_j = max(2.8 - 0.0024 * d_b, 1.84)
            return PL_Los(d_a) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_b) + 3 * np.log10(self.fc / 5)

        if min(d1, d2) < 7:
            PL = PL_Los(d)
        else:
            PL = min(PL_NLos(d1, d2), PL_NLos(d2, d1))
        return PL  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        return np.exp(-1 * (delta_distance / self.decorrelation_distance)) * shadowing \
               + math.sqrt(1 - np.exp(-2 * (delta_distance / self.decorrelation_distance))) * np.random.normal(0, 3)  # standard dev is 3 db

class V2V_RIS_channels:
    """Simulator of the V2V_RIS channels"""
    def __init__(self):
        self.t = 0  # 开始时间
        self.h_bs = 25  # RIS的高度
        self.h_ms = 1.5  # 车辆的高度
        self.fc = 2  # 中心频率
        self.Decorrelation_distance = 10  #距离相关系数
        self.shadow_std = 8

    def get_path_loss(self, position_A, position_B, theta, n_elements_total):
        """Calculate RIS pathloss between V2V pairs"""
        theta_all = np.zeros(n_elements_total, dtype=complex)
        a_aoa_all = np.zeros(n_elements_total, dtype=complex)
        a_aod_all = np.zeros(n_elements_total, dtype=complex)
        n_elements_per_row = len(theta)
        number_of_row = np.floor(n_elements_total / n_elements_per_row)

        ds = 0.02
        dA1 = abs(position_A[0] - position_RIS[0])
        dA2 = abs(position_A[1] - position_RIS[1])
        dB1 = abs(position_B[0] - position_RIS[0])
        dB2 = abs(position_B[1] - position_RIS[1])
        dA = math.hypot(dA1, dA2) + 0.001
        dB = math.hypot(dB1, dB2) + 0.001
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)
        a_aoa = np.zeros(n_elements_per_row, dtype=complex)
        a_aod = np.zeros(n_elements_per_row, dtype=complex)
        theta_aoa = np.arctan((position_A[1] - position_RIS[1]) / (position_A[0] - position_RIS[0]))
        theta_aod = np.arctan((position_B[1] - position_RIS[1]) / (position_B[0] - position_RIS[0]))

        def PL_Los(d):
            if d <= 3:
                return 22.7 * np.log10(3) + 41 + 20 * np.log10(self.fc / 5)
            else:
                if d < d_bp:
                    return 22.7 * np.log10(d) + 41 + 20 * np.log10(self.fc / 5)
                else:
                    return 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(self.h_bs) - 17.3 * np.log10(
                        self.h_ms) + 2.7 * np.log10(self.fc / 5)

        PLA = 128.1 + 37.6 * np.log10(math.sqrt(dA ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) #24 + 20 * np.log10(dA) + 20 * np.log10(self.fc / 5)
        PLB = 128.1 + 37.6 * np.log10(math.sqrt(dB ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)

        for n in range(n_elements_per_row):
            a_aod[n] = np.exp(-1.0j * (2 * np.pi) * ds * (self.fc / 0.3) * n * np.sin(theta_aod))
            a_aoa[n] = np.exp(-1.0j * (2 * np.pi) * ds * (self.fc / 0.3) * n * np.sin(theta_aoa))

        for i in range(n_elements_total):
            index = i % n_elements_per_row
            theta_all[i] = theta[index]
            a_aoa_all[i] = a_aoa[index]
            a_aod_all[i] = a_aod[index]
        theta_diag = np.diag(theta_all)
        #求RIS的相移矩阵

        ChannelA = 1 / np.power(10, PLA / 10) * np.exp(-1.0j * (2 * np.pi) * dA * (self.fc / 0.3)) * a_aoa_all
        ChannelB = 1 / np.power(10, PLB / 10) * np.exp(-1.0j * (2 * np.pi) * dB * (self.fc / 0.3)) * a_aod_all.conj().T

        PL_RIS_sig = np.dot(np.dot(ChannelA, theta_diag), ChannelB) #信道增益
        if (np.real(PL_RIS_sig)+np.imag(PL_RIS_sig))!=0:
            PL_RIS = np.log10((1 / (PL_RIS_sig))) * 10
        else:
            PL_RIS = 0j

        return PL_RIS  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        #self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8)


class V2Ichannels:

    # Simulator of the V2I channels

    def __init__(self):
        self.h_bs = 25   #基站的高度
        self.h_ms = 1.5  #车辆的高度
        self.fc = 2
        self.Decorrelation_distance = 10  #距离相关系数
        self.shadow_std = 8

    def get_path_loss(self, position_A):
        d1 = abs(position_A[0] - BS_position[0])
        d2 = abs(position_A[1] - BS_position[1])
        distance = math.hypot(d1, d2)
        return 128.1 + 37.6 * np.log10(math.sqrt(distance ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000) # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)

class V2I_RIS_channels:
    def __init__(self):
        self.h_bs = 25  # 基站和RIS的高度
        self.h_ms = 1.5  # 车辆的高度
        self.fc = 2
        self.Decorrelation_distance = 10  # 距离相关系数
        self.shadow_std = 8
        self.shadow_std2 = 4

    def get_path_loss(self, position_A, theta, n_elements_total):
        """Calculate RIS pathloss between cellular users and BS"""
        theta_all = np.zeros(n_elements_total, dtype=complex)
        a_aoa_all = np.zeros(n_elements_total, dtype=complex)
        a_aod_all = np.zeros(n_elements_total, dtype=complex)
        n_elements_per_row = len(theta)
        number_of_row = np.floor(n_elements_total / n_elements_per_row)
        ds = 0.02 # The spacing between elements
        dA1 = abs(position_A[0] - position_RIS[0])
        dA2 = abs(position_A[1] - position_RIS[1])
        dB1 = abs(BS_position[0] - position_RIS[0])
        dB2 = abs(BS_position[1] - position_RIS[1])
        dA = math.hypot(dA1, dA2)
        dB = math.hypot(dB1, dB2)
        d_bp = 4 * (self.h_bs - 1) * (self.h_ms - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)
        #d_bp2 = 4 * (self.h_bs - 1) * (self.h_bs - 1) * self.fc * (10 ** 9) / (3 * 10 ** 8)
        a_aoa = np.zeros(n_elements_per_row, dtype=complex)
        a_aod = np.zeros(n_elements_per_row, dtype=complex)
        theta_aoa = np.arctan((position_A[1] - position_RIS[1]) / (position_A[0] - position_RIS[0]))

        theta_aod = np.arctan((BS_position[1] - position_RIS[1])/(BS_position[0] - position_RIS[0]))

        PLA = 128.1 + 37.6 * np.log10(math.sqrt(dA ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)  #车辆和RIS之间的 24 + 20 * np.log10(dA) + 20 * np.log10(self.fc / 5)
        PLB = 23.5* np.log10(math.sqrt(dB ** 2 + (self.h_bs - self.h_bs) ** 2) / 1000) + 57.5 + 23 * np.log10(2 / 5) #RIS和BS之间的路径损失，两者的高度都定义为25   128.1 + 37.6 * np.log10(math.sqrt(dB ** 2 + (self.h_bs - self.h_ms) ** 2) / 1000)

        for n in range(n_elements_per_row):
            a_aod[n] = np.exp(-1.0j * (2 * np.pi) * ds * (self.fc / 0.3) * n * np.sin(theta_aod))
            a_aoa[n] = np.exp(-1.0j * (2 * np.pi) * ds * (self.fc / 0.3) * n * np.sin(theta_aoa))

        for i in range(n_elements_total):
            index = i % n_elements_per_row
            theta_all[i] = theta[index]
            a_aoa_all[i] = a_aoa[index]
            a_aod_all[i] = a_aod[index]
        theta_diag = np.diag(theta_all)

        ChannelA = 1/np.power(10, PLA/10) * np.exp(-1.0j*(2*np.pi)*dA*(self.fc/0.3))*a_aoa_all
        ChannelB = 1/np.power(10, PLB/10) * np.exp(-1.0j*(2*np.pi)*dB*(self.fc/0.3))*a_aod_all.conj().T

        PL_RIS_sig = np.dot(np.dot(ChannelA,  theta_diag), ChannelB)
        if (np.real(PL_RIS_sig) + np.imag(PL_RIS_sig)) != 0:
            PL_RIS = np.log10((1 / (PL_RIS_sig))) * 10
        else:
            PL_RIS = 0j
        return PL_RIS  # + self.shadow_std * np.random.normal()

    def get_shadowing(self, delta_distance, shadowing):
        nVeh = len(shadowing)
        self.R = np.sqrt(0.5 * np.ones([nVeh, nVeh]) + 0.5 * np.identity(nVeh))
        return np.multiply(np.exp(-1 * (delta_distance / self.Decorrelation_distance)), shadowing) \
               + np.sqrt(1 - np.exp(-2 * (delta_distance / self.Decorrelation_distance))) * np.random.normal(0, 8, nVeh)



class Vehicle:
    """Vehicle simulator: include all the information for a Vehicle"""

    def __init__(self, start_position, start_direction, velocity):
        self.position = start_position
        self.direction = start_direction
        self.velocity = velocity
        self.neighbors = []
        self.destinations = []

class Environ:
    def __init__(self, down_lane, up_lane, left_lane, right_lane, width, height, n_veh, n_neighbor, V2I_min):

        self.down_lanes = down_lane
        self.up_lanes = up_lane
        self.left_lanes = left_lane
        self.right_lanes = right_lane
        self.width = width
        self.height = height

        self.h_bs = 25
        self.h_ms = 1.5
        self.fc = 2
        self.sig2_dB = -114  # 噪声功率
        self.sig2 = 10 ** (self.sig2_dB / 10)
        self.bsAntGain = 8  # 基站天线增益
        self.bsNoiseFigure = 5  # 基站接收器噪声系数
        self.vehAntGain = 3  # 车辆天线增益
        self.vehNoiseFigure = 11  # 车辆接收器噪声增益

        self.n_RB = n_veh
        self.n_Veh = n_veh
        self.n_neighbor = n_neighbor

        self.V2Vchannels = V2Vchannels()
        self.V2Ichannels = V2Ichannels()
        self.V2V_RIS_channels = V2V_RIS_channels()
        self.V2I_RIS_channels = V2I_RIS_channels()
        self.vehicles = []

        self.theta = np.zeros(n_elements, dtype=complex)
        self.next_theta_number = np.zeros((n_elements))

        self.V2I_power_dB = 23  # dBm

        self.demand = []
        self.V2V_Shadowing = []
        self.V2I_Shadowing = []
        self.V2V_RIS_Shadowing = []
        self.V2V_RIS_Shadowing1 = []
        self.V2V_RIS_Shadowing2 = []
        self.V2I_RIS_Shadowing = []
        self.delta_distance = []
        self.V2V_pathloss = []
        self.V2I_pathloss = []
        self.V2V_RIS_pathloss = []
        self.V2I_RIS_pathloss = []
        self.V2V_channels_abs = []
        self.V2I_channels_abs = []

        self.V2I_min = V2I_min

        self.bandwidth = int(1e6)  # bandwidth per RB, 1 MHz
        self.demand_size = int((4 * 190 + 300) * 8)  # V2V payload: 1060 Bytes every 100 ms
        self.time_fast = 0.001
        self.time_slow = 0.1  # update slow fading/vehicle position every 100 ms

        self.V2V_Interference_all = np.zeros((self.n_Veh, self.n_neighbor, self.n_RB)) + self.sig2

    def add_new_vehicles(self, start_position, start_direction, start_velocity):
        self.vehicles.append(Vehicle(start_position, start_direction, start_velocity))

    def add_new_vehicles_by_number(self, n):
        string = 'dulr'
        for i in range(n):
            ind = np.random.randint(0, len(self.down_lanes))

            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'd'  # velocity: 10 ~ 15 m/s, random
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [self.up_lanes[ind], np.random.randint(0, self.height)]
            start_direction = 'u'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(0, self.width), self.left_lanes[ind]]
            start_direction = 'l'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

            start_position = [np.random.randint(0, self.width), self.right_lanes[ind]]
            start_direction = 'r'
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

        for j in range(int(self.n_Veh % 4)): #当车辆数不是4的倍数时，按照这个添加车辆
            ind = np.random.randint(0, len(self.down_lanes))
            str = random.choice(string)
            start_position = [self.down_lanes[ind], np.random.randint(0, self.height)]
            start_direction = str  # velocity: 10 ~ 15 m/s, random
            self.add_new_vehicles(start_position, start_direction, np.random.randint(10, 15))

        # initialize channels
        self.V2V_Shadowing = np.random.normal(0, 3, [len(self.vehicles), len(self.vehicles)])
        self.V2V_RIS_Shadowing1 = self.V2V_RIS_Shadowing2 = np.random.normal(0, 8, [len(self.vehicles), len(self.vehicles)])
        self.V2I_Shadowing = np.random.normal(0, 8, len(self.vehicles))
        self.V2I_RIS_Shadowing = np.random.normal(0, 8, len(self.vehicles))

        self.delta_distance = np.asarray([c.velocity*self.time_slow for c in self.vehicles])

    def renew_positions(self):
        # ===============
        # This function updates the position of each vehicle
        # ===============

        i = 0
        while (i < len(self.vehicles)):
            delta_distance = self.vehicles[i].velocity * self.time_slow
            change_direction = False
            if self.vehicles[i].direction == 'u':
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):

                    if (self.vehicles[i].position[1] <= self.left_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.left_lanes[j] - self.vehicles[i].position[1])), self.left_lanes[j]]
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] <= self.right_lanes[j]) and ((self.vehicles[i].position[1] + delta_distance) >= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.right_lanes[j] - self.vehicles[i].position[1])), self.right_lanes[j]]
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] += delta_distance
            if (self.vehicles[i].direction == 'd') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.left_lanes)):
                    if (self.vehicles[i].position[1] >= self.left_lanes[j]) and ((self.vehicles[i].position[1] - delta_distance) <= self.left_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.vehicles[i].position[0] - (delta_distance - (self.vehicles[i].position[1] - self.left_lanes[j])), self.left_lanes[j]]
                            # print ('down with left', self.vehicles[i].position)
                            self.vehicles[i].direction = 'l'
                            change_direction = True
                            break
                if change_direction == False:
                    for j in range(len(self.right_lanes)):
                        if (self.vehicles[i].position[1] >= self.right_lanes[j]) and (self.vehicles[i].position[1] - delta_distance <= self.right_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.vehicles[i].position[0] + (delta_distance + (self.vehicles[i].position[1] - self.right_lanes[j])), self.right_lanes[j]]
                                # print ('down with right', self.vehicles[i].position)
                                self.vehicles[i].direction = 'r'
                                change_direction = True
                                break
                if change_direction == False:
                    self.vehicles[i].position[1] -= delta_distance
            if (self.vehicles[i].direction == 'r') and (change_direction == False):
                # print ('len of position', len(self.position), i)
                for j in range(len(self.up_lanes)):
                    if (self.vehicles[i].position[0] <= self.up_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.up_lanes[j] - self.vehicles[i].position[0]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] <= self.down_lanes[j]) and ((self.vehicles[i].position[0] + delta_distance) >= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.down_lanes[j] - self.vehicles[i].position[0]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                if change_direction == False:
                    self.vehicles[i].position[0] += delta_distance
            if (self.vehicles[i].direction == 'l') and (change_direction == False):
                for j in range(len(self.up_lanes)):

                    if (self.vehicles[i].position[0] >= self.up_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.up_lanes[j]):  # came to an cross
                        if (np.random.uniform(0, 1) < 0.4):
                            self.vehicles[i].position = [self.up_lanes[j], self.vehicles[i].position[1] + (delta_distance - (self.vehicles[i].position[0] - self.up_lanes[j]))]
                            change_direction = True
                            self.vehicles[i].direction = 'u'
                            break
                if change_direction == False:
                    for j in range(len(self.down_lanes)):
                        if (self.vehicles[i].position[0] >= self.down_lanes[j]) and ((self.vehicles[i].position[0] - delta_distance) <= self.down_lanes[j]):
                            if (np.random.uniform(0, 1) < 0.4):
                                self.vehicles[i].position = [self.down_lanes[j], self.vehicles[i].position[1] - (delta_distance - (self.vehicles[i].position[0] - self.down_lanes[j]))]
                                change_direction = True
                                self.vehicles[i].direction = 'd'
                                break
                    if change_direction == False:
                        self.vehicles[i].position[0] -= delta_distance

            # if it comes to an exit
            if (self.vehicles[i].position[0] < 0) or (self.vehicles[i].position[1] < 0) or (self.vehicles[i].position[0] > self.width) or (self.vehicles[i].position[1] > self.height):
                # delete
                #    print ('delete ', self.position[i])
                if (self.vehicles[i].direction == 'u'):
                    self.vehicles[i].direction = 'r'
                    self.vehicles[i].position = [self.vehicles[i].position[0], self.right_lanes[-1]]
                else:
                    if (self.vehicles[i].direction == 'd'):
                        self.vehicles[i].direction = 'l'
                        self.vehicles[i].position = [self.vehicles[i].position[0], self.left_lanes[0]]
                    else:
                        if (self.vehicles[i].direction == 'l'):
                            self.vehicles[i].direction = 'u'
                            self.vehicles[i].position = [self.up_lanes[0], self.vehicles[i].position[1]]
                        else:
                            if (self.vehicles[i].direction == 'r'):
                                self.vehicles[i].direction = 'd'
                                self.vehicles[i].position = [self.down_lanes[-1], self.vehicles[i].position[1]]

            i += 1



    def overall_channel(self, theta, n_elements_total):
        """The combined channel"""

        self.V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles))) + 50 * np.identity(len(self.vehicles))  # D2D之间路径损失
        self.V2I_pathloss = np.zeros((len(self.vehicles)))
        self.V2V_RIS_pathloss = np.zeros((len(self.vehicles), len(self.vehicles)), dtype=complex)  # D to RIS to D
        self.V2I_RIS_pathloss = np.zeros((len(self.vehicles)), dtype=complex)

        self.V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2I_channels_abs = np.zeros((len(self.vehicles)))

        self.V2V_Shadowing = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2V_RIS_Shadowing = np.zeros((len(self.vehicles), len(self.vehicles)))

        self.theta = theta

        self.V2V_to_BS_pathloss = np.zeros((len(self.vehicles)))  # D2D to BS interference channel
        self.V2I_to_V2V_pathloss = np.zeros((len(self.vehicles), len(self.vehicles)))  # Cellular user to D2D user interference channel
        self.V2V_to_BS_RIS_pathloss = np.zeros((len(self.vehicles)), dtype=complex)
        self.V2I_to_V2V_RIS_pathloss = np.zeros((len(self.vehicles), len(self.vehicles)), dtype=complex)

        self.V2I_to_V2V_channels_abs = np.zeros((len(self.vehicles), len(self.vehicles)))
        self.V2V_to_BS_channels_abs = np.zeros((len(self.vehicles)))
        #V2V信道计算
        for i in range(len(self.vehicles)):
            for j in range(i+1,len(self.vehicles)):
                self.V2V_Shadowing[j][i] = self.V2V_Shadowing[i][j] = self.V2Vchannels.get_shadowing(self.delta_distance[i] + self.delta_distance[j],
                                                                                                     self.V2V_Shadowing[i][j])
                self.V2V_pathloss[j,i] = self.V2V_pathloss[i][j] = self.V2Vchannels.get_path_loss(self.vehicles[i].position,
                                                                                                  self.vehicles[j].position)
                self.V2V_RIS_pathloss[j,i] = self.V2V_RIS_pathloss[i][j] = self.V2V_RIS_channels.get_path_loss(self.vehicles[i].position,
                                                                                                               self.vehicles[j].position,
                                                                                                               self.theta, n_elements_total)
                self.V2V_RIS_Shadowing1[i][j] = self.V2V_RIS_channels.get_shadowing(self.delta_distance[i], self.V2V_RIS_Shadowing1[i][j])
                self.V2V_RIS_Shadowing2[i][j] = self.V2V_RIS_channels.get_shadowing(self.delta_distance[j], self.V2V_RIS_Shadowing2[i][j])

        self.V2V_overall = 1 / np.abs(1 / np.power(10, self.V2V_pathloss / 10) + 1 / np.power(10, self.V2V_RIS_pathloss / 10))
        self.V2V_channels_abs = 10 * np.log10(self.V2V_overall) + self.V2V_RIS_Shadowing1 + self.V2V_RIS_Shadowing2 + self.V2V_Shadowing
        #V2I信道计算
        self.V2I_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_Shadowing)
        self.V2I_RIS_Shadowing = self.V2Ichannels.get_shadowing(self.delta_distance, self.V2I_RIS_Shadowing)
        for i in range(len(self.vehicles)):
            self.V2I_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].position)
            self.V2I_RIS_pathloss[i] = self.V2I_RIS_channels.get_path_loss(self.vehicles[i].position, self.theta, n_elements_total)

        self.V2I_overall = 1 / np.abs(1 / np.power(10, self.V2I_pathloss / 10) + 1 / np.power(10, self.V2I_RIS_pathloss / 10))
        self.V2I_channels_abs = 10 * np.log10(self.V2I_overall) + self.V2I_Shadowing + self.V2I_RIS_Shadowing
        #V2V对基站的干扰信道和V2I对V2V的干扰信道
        for i in range(len(self.vehicles)):
            self.V2V_to_BS_pathloss[i] = self.V2Ichannels.get_path_loss(self.vehicles[i].position)
            self.V2V_to_BS_RIS_pathloss[i] = self.V2I_RIS_channels.get_path_loss(self.vehicles[i].position, self.theta,n_elements_total)
            for j in range(len(self.vehicles)):
                self.V2I_to_V2V_pathloss[i,j] = self.V2Vchannels.get_path_loss(self.vehicles[i].position, self.vehicles[j].position) # i-th cellular user to j-th D2D user
                self.V2I_to_V2V_RIS_pathloss[i,j] = self.V2V_RIS_channels.get_path_loss(self.vehicles[i].position, self.vehicles[j].position, self.theta, n_elements_total)

        self.V2I_to_V2V_overall = 1 / np.abs(1 / np.power(10, self.V2I_to_V2V_pathloss / 10) + 1 / np.power(10, self.V2I_to_V2V_RIS_pathloss))
        self.V2I_to_V2V_channels_abs = 10 * np.log10(self.V2I_to_V2V_overall)

        self.V2V_to_BS_overall = 1 / np.abs(1 / np.power(10, self.V2V_to_BS_pathloss / 10) + 1 / np.power(10, self.V2V_to_BS_RIS_pathloss))
        self.V2V_to_BS_channels_abs = 10 * np.log10(self.V2V_to_BS_overall)


    def renew_neighbor(self):
        """ Determine the neighbors of each vehicles """

        for i in range(len(self.vehicles)):
            self.vehicles[i].neighbors = []
            self.vehicles[i].actions = []
        z = np.array([[complex(c.position[0], c.position[1]) for c in self.vehicles]])
        Distance = abs(z.T - z)

        for i in range(len(self.vehicles)):
            sort_idx = np.argsort(Distance[:, i])
            for j in range(self.n_neighbor):
                self.vehicles[i].neighbors.append(sort_idx[j + 1])
            destination = self.vehicles[i].neighbors

            self.vehicles[i].destinations = destination


    def renew_channel_fastfading(self):
        """Renew fast fading channel"""

        V2V_pathloss_with_fastfading = np.repeat(self.V2V_channels_abs[:, :, np.newaxis], self.n_RB, axis=2)
        self.V2V_pathloss_with_fastfading = V2V_pathloss_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2V_pathloss_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2V_pathloss_with_fastfading.shape)) / math.sqrt(2))

        #V2V_RIS_pathloss = np.repeat(self.V2V_RIS_pathloss[:, :, np.newaxis], self.n_RB, axis=2)


        #V2V_overall_with_fastfading = 1 / np.abs(
         #   1 / np.power(10, self.V2V_pathloss_with_fastfading / 10) + 1 / np.power(10, V2V_RIS_pathloss / 10))
        #self.V2V_channels_with_fastfading = 10 * np.log10(V2V_overall_with_fastfading)

        V2I_pathloss_with_fastfading = np.repeat(self.V2I_channels_abs[:, np.newaxis], self.n_RB, axis=1)
        self.V2I_pathloss_with_fastfading = V2I_pathloss_with_fastfading - 20 * np.log10(
            np.abs(np.random.normal(0, 1, V2I_pathloss_with_fastfading.shape) + 1j * np.random.normal(0, 1, V2I_pathloss_with_fastfading.shape)) / math.sqrt(2))

        #V2I_RIS_pathloss = np.repeat(self.V2I_RIS_pathloss[:, np.newaxis], self.n_RB, axis=1)

        #V2I_overall_with_fastfading = 1 / np.abs(
         #   1 / np.power(10, self.V2I_pathloss_with_fastfading / 10) + 1 / np.power(10, V2I_RIS_pathloss / 10))
        #self.V2I_channels_with_fastfading = 10 * np.log10(V2I_overall_with_fastfading)

    def get_state(self, idx=(0,0)):
        '''Get channel information from the environment'''
        theta_number = self.next_theta_number

        V2I_fast = (self.V2I_pathloss_with_fastfading[idx[0], :] - self.V2I_channels_abs[idx[0]] + 10)/35

        V2V_fast = (self.V2V_pathloss_with_fastfading[:, self.vehicles[idx[0]].destinations[idx[1]], :] -
                    self.V2V_channels_abs[:, self.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

        V2I_abs = (self.V2I_channels_abs[idx[0]] - 80)/60.0

        V2V_abs = (self.V2V_channels_abs[:, self.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0

        V2V_interference = (-self.V2V_Interference_all[idx[0], idx[1], :] - 60)/60

        AoI_levels = self.AoI[idx[0]] / (int(self.time_slow / self.time_fast))
        load_remaining = np.asarray([self.demand[idx[0], idx[1]] / self.demand_size])

        return np.concatenate((np.reshape(theta_number, -1), np.reshape(V2I_fast, -1), np.reshape(V2V_fast, -1), np.reshape(V2V_interference, -1), np.reshape(V2I_abs, -1),
                              np.reshape(V2V_abs, -1), np.reshape(AoI_levels, -1), np.reshape(load_remaining, -1)))

    def get_RIS_next_state(self, observation, actions_temp):
        M = 3
        self.observation = observation
        self.n_phases = np.power(3, n_elements)
        theta_number= np.real(self.observation[0 : 0+n_elements]) * RIS_phase_numbers
        element_phase_action = np.zeros(n_elements)
        phase_action = actions_temp % self.n_phases
        for n in range(n_elements):
            element_phase_action[n] = int(np.floor(phase_action % np.power(M, n + 1) / np.power(M, n)))
            if element_phase_action[n] == 0:
                theta_number[n] = theta_number[n]
            elif element_phase_action[n] == 1:
                theta_number[n] = (theta_number[n] + 1) % RIS_phase_numbers
            elif element_phase_action[n] == 2:
                theta_number[n] = (theta_number[n] - 1) % RIS_phase_numbers
            else:
                print("Something goes wrong!")
        self.next_theta_number = theta_number / RIS_phase_numbers
        theta = np.zeros(n_elements, dtype=complex)
        for n in range(n_elements):
            theta[n] = np.exp(1j * theta_number[n] * (2 * np.pi / RIS_phase_numbers))
        return theta, self.next_theta_number

    def Compute_Performance_Reward_Train(self, actions_channel):
        #这里输入的动作是一个数组，以优化power和channel

        sub_selection = actions_channel[:, :, 0].astype('int').reshape(len(self.vehicles), 1) # the channel selection part
        power_selection = actions_channel[:, :, 1].reshape(len(self.vehicles), 1) #power selection

        #------------------Compute V2I rate------------------
        V2I_Rate = np.zeros(self.n_RB)
        V2I_Interference = np.zeros(self.n_RB)  # V2I interference
        for i in range(len(self.vehicles)):
            for j in range(self.n_neighbor):
                if not self.active_links[i, j]:
                    continue
                V2I_Interference[sub_selection[i][j]] += 10 ** ((power_selection[i, j] - self.V2I_pathloss_with_fastfading[i, sub_selection[i, j]]
                                                                 + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        self.V2I_Interference = V2I_Interference + self.sig2
        V2I_Signals = 10 ** ((self.V2I_power_dB - self.V2I_pathloss_with_fastfading.diagonal() + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure) / 10)
        V2I_Rate = np.log2(1 + np.divide(V2I_Signals, self.V2I_Interference))

        #---------------Compute V2V rate-------------------
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor))
        V2V_Signal = np.zeros((len(self.vehicles), self.n_neighbor))
        sub_selection[(np.logical_not((self.active_links)))] = -1
        for i in range(self.n_RB):
            indexes = np.argwhere(sub_selection == i)
            for j in range(len(indexes)):
                receiver_j = self.vehicles[indexes[j, 0]].destinations[indexes[j, 1]]
                V2V_Signal[indexes[j, 0], [indexes[j, 1]]] = 10 ** ((power_selection[indexes[j, 0], indexes[j, 1]]
                                                                     - self.V2V_pathloss_with_fastfading[indexes[j][0], receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                #V2I links interference to V2V links
                V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((self.V2I_power_dB - self.V2V_pathloss_with_fastfading[i, receiver_j, i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

                #V2V interference
                for k in range(j + 1, len(indexes)):  # spectrum-sharing V2Vs
                    receiver_k = self.vehicles[indexes[k][0]].destinations[indexes[k][1]]
                    V2V_Interference[indexes[j, 0], indexes[j, 1]] += 10 ** ((power_selection[indexes[k, 0], indexes[k, 1]]
                                                                              - self.V2V_pathloss_with_fastfading[indexes[k][0]][receiver_j][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
                    V2V_Interference[indexes[k, 0], indexes[k, 1]] += 10 ** ((power_selection[indexes[j, 0], indexes[j, 1]]
                                                                              - self.V2V_pathloss_with_fastfading[indexes[j][0]][receiver_k][i] + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        self.V2V_Interference = V2V_Interference + self.sig2
        V2V_Rate = np.log2(1 + np.divide(V2V_Signal, self.V2V_Interference))

        V2I_AoI = self.Age_of_Information(V2I_Rate)

        # 约束条件
        self.demand -= V2V_Rate * self.time_fast * self.bandwidth
        self.demand[self.demand < 0] = 0  # eliminate negative demands

        self.individual_time_limit -= self.time_fast

        reward_elements = V2V_Rate
        reward_elements[self.demand <= 0] = 1

        self.active_links[np.multiply(self.active_links, self.demand <= 0)] = 0  # transmission finished, turned to "inactive"

        return V2I_Rate, V2V_Rate, V2I_AoI, self.demand, reward_elements


    def Age_of_Information(self, V2I_rate):
        # computing the vehicle age of information

        for i in range(int(self.n_RB)):
            if V2I_rate[i] >= self.V2I_min:
                self.AoI[i] = 1
            else:
                self.AoI[i] += 1
                if self.AoI[i] >= 100:
                    self.AoI[i] = 100
        return self.AoI

    #为了更好的训练，设置的多个奖励函数
    def act_for_training(self, observation, actions_temp, actions_channel):
        action_channel = actions_channel.copy()
        theta, next_theta_number = self.get_RIS_next_state(observation, actions_temp)
        self.overall_channel(theta)
        self.renew_channel_fastfading()
        V2V_demand = np.zeros(self.n_Veh)
        V2I_Rate, V2V_Rate, V2I_AoI, demand,reward_elements = self.Compute_Performance_Reward_Train(action_channel)
        V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)  # V2V success rates
        for i in range(len(V2I_Rate)):
            V2V_demand[i] = demand[i] / self.demand_size
        lambdda = 0.9
        reward = lambdda * (np.sum(V2I_Rate) / (self.n_Veh * 10) - np.sum(V2I_AoI) / (self.n_Veh * 10)) - (1 - lambdda) * (np.sum(V2V_demand) / (self.n_Veh * self.n_neighbor))

        return reward, V2I_Rate, V2V_Rate, V2V_success, V2I_AoI


    #当前训练的奖励函数
    def act_for_training1(self, action_channel, action_phase):
        self.next_theta_number = action_phase.reshape(n_elements, 1)
        theta = np.zeros(n_elements, dtype=complex)
        for n in range(n_elements):
            theta[n] = np.exp(1j * self.next_theta_number[n] * (2 * np.pi / RIS_phase_numbers))
        self.overall_channel(theta, n_elements_total)
        self.renew_channel_fastfading()
        #因为要训练最优相移，但相移更新了，路径损失也会更新，所以在计算奖励前要再进行一次信道更新
        V2I_Rate, V2V_Rate, V2I_AoI, demand, reward_elements = self.Compute_Performance_Reward_Train(action_channel)
        V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)  # V2V success rates
        per_user_reward = np.zeros(self.n_Veh)
        for i in range(self.n_Veh):
            per_user_reward[i] = (-10) * (demand[i] / self.demand_size) - V2I_AoI[i]
        # reward = (-np.sum(V2I_AoI) / (self.n_Veh)) + ( np.sum(reward_elements) / (self.n_Veh * self.n_neighbor) )  + np.sum(V2I_Rate)/(self.n_Veh * 20)
        reward = np.mean(per_user_reward)
        return reward, V2I_Rate, V2V_Rate, V2V_success, V2I_AoI
        '''self.next_theta_number = action_phase.reshape(self.n_RB, 1)
        theta = np.zeros(RIS_numbers, dtype=complex)
        for n in range(RIS_numbers):
            theta[n] = np.exp(1j * self.next_theta_number[n] * (2 * np.pi / RIS_phase_numbers))
        self.overall_channel(theta, n_elements)
        self.renew_channel_fastfading()
        V2I_Rate, V2V_Rate, V2I_AoI, demand, reward_elements = self.Compute_Performance_Reward_Train(action_channel)
        V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)  # V2V success rates
        V2V_demand = np.zeros(4)
        for i in range(len(V2I_Rate)):
            V2V_demand[i] = demand[i] / self.demand_size
        reward = (np.sum(V2I_Rate) / (self.n_Veh * 50))  - np.sum(V2I_AoI) / (self.n_Veh) - (np.sum(V2V_demand) / (self.n_Veh * self.n_neighbor))
        return reward, V2I_Rate, V2V_Rate, V2V_success, V2I_AoI'''

    def act_for_testing(self, action_channel, action_phase):
        self.next_theta_number = action_phase.reshape(n_elements, 1)
        theta = np.zeros(n_elements, dtype=complex)
        for n in range(n_elements):
            theta[n] = np.exp(1j * self.next_theta_number[n] * (2 * np.pi / RIS_phase_numbers))
        self.overall_channel(theta, n_elements_total)
        self.renew_channel_fastfading()

        V2I_Rate, V2V_Rate, V2I_AoI, demand, reward_elements = self.Compute_Performance_Reward_Train(action_channel)
        V2V_success = 1 - np.sum(self.active_links) / (self.n_Veh * self.n_neighbor)  # V2V success rates
        return V2I_Rate, V2V_Rate, V2I_AoI, V2V_success


    def Compute_Interference(self, actions_all):
        #计算V2V的干扰：来自V2I链路的干扰和来自其他V2V队的干扰
        V2V_Interference = np.zeros((len(self.vehicles), self.n_neighbor, self.n_RB)) + self.sig2

        sub_selection = actions_all[:, :, 0].astype('int').reshape(len(self.vehicles), 1)  # the channel selection part
        power_selection = actions_all[:, :, 1].reshape(len(self.vehicles), 1)  # power selection
        sub_selection[np.logical_not(self.active_links)] = -1

        # interference from V2I links
        for i in range(self.n_RB):
            for k in range(len(self.vehicles)):
                for m in range(len(sub_selection[k, :])):
                    V2V_Interference[k, m, i] += 10 ** ((self.V2I_power_dB - self.V2V_pathloss_with_fastfading[i][self.vehicles[k].destinations[m]][i]
                                                         + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)

        # interference from peer V2V links
        for i in range(len(self.vehicles)):
            for j in range(len(sub_selection[i, :])):
                for k in range(len(self.vehicles)):
                    for m in range(len(sub_selection[k, :])):
                        # if i == k or channel_selection[i,j] >= 0:
                        if i == k and j == m or sub_selection[i, j] < 0:
                            continue
                        V2V_Interference[k, m, sub_selection[i, j]] += 10 ** ((power_selection[i, j]
                                                                                   - self.V2V_pathloss_with_fastfading[i][self.vehicles[k].destinations[m]][sub_selection[i,j]]
                                                                               + 2 * self.vehAntGain - self.vehNoiseFigure) / 10)
        self.V2V_Interference_all = 10 * np.log10(V2V_Interference)


    def new_random_game(self, n_Veh=0):

        # make a new game
        self.vehicles = []
        if n_Veh > 0:
            self.n_Veh = n_Veh
        self.add_new_vehicles_by_number(int(self.n_Veh / 4))
        self.renew_neighbor()
        self.overall_channel(self.theta, n_elements_total)
        self.renew_channel_fastfading()

        self.demand = self.demand_size * np.ones((self.n_Veh , self.n_neighbor))
        self.individual_time_limit = self.time_slow * np.ones((self.n_Veh , self.n_neighbor))
        self.active_links = np.ones((self.n_Veh, self.n_neighbor), dtype='bool')
        self.AoI = np.ones(self.n_Veh, dtype=np.float16)*100
