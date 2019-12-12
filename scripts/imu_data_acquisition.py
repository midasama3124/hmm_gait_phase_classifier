#!/usr/bin/python
import rospy
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool
import smbus
import time
import ctypes
import numpy as np

def enum(*args):
    enums = dict(zip(args, range(len(args))))
    return type('Enum', (), enums)

def twos_complement(input_value, num_bits):
    mask = 2**(num_bits - 1)
    return -(input_value & mask) + (input_value & ~mask)

class IMU_BNO055:

    """
    BNO055
    IMU 9DOF Sensor Fusion
    """

    def __init__(self, bus = 1, address = 0x28):
        self.bus = smbus.SMBus(bus)    # /dev/i2c-1
        self.address = address
        self.start_time = time.time()
        # self.gyro = {'x': 0, 'y': [], 'z': 0}
        # self.accel = {'x': 0, 'y': [], 'z': 0}
        self.gyro = {'x': 0, 'y': 0, 'z': 0}
        self.accel = {'x': 0, 'y': 0, 'z': 0}
        self.quat = {'x': 0, 'y': 0, 'z': 0, 'w': 0}
        """Butterworth low-pass filter (Cutoff frequency: 15 Hz)"""
        self.gyro_filter_states = np.empty(2)
        self.accel_filter_states = np.empty(2)
        '''Numerator'''
        self.gyro_a0_ = 0.1176
        self.gyro_a1_ = 0.2352
        self.gyro_a2_ = 0.1176
        self.accel_a0_ = 0.1398
        self.accel_a1_ = 0.2796
        self.accel_a2_ = 0.1398
        '''Denominator'''
        self.gyro_b0_ = 1
        self.gyro_b1_ = -0.8239
        self.gyro_b2_ = 0.2942
        self.accel_b0_ = 1
        self.accel_b1_ = -0.7006
        self.accel_b2_ = 0.2597
        """Calibration stage"""
        # self.cal_angle = 0                   # Initial calibration angle
        GPwrMode = self.GPwrMode.NormalG     #  Gyro power mode
        Gscale = self.Gscale.GFS_250DPS   #  Gyro full scale
        # Godr = GODR_250Hz     #  Gyro sample rate
        Gbw = self.Gbw.GBW_23Hz        #  Gyro bandwidth
        Ascale = self.Ascale.AFS_2G       #  Accel full scale
        # Aodr = AODR_250Hz     #  Accel sample rate
        APwrMode = self.APwrMode.NormalA     #  Accel power mode
        Abw = self.Abw.ABW_31_25Hz     #  Accel bandwidth, accel sample rate divided by ABW_divx
        # Mscale = MFS_4Gauss   #  Select magnetometer full-scale resolution
        MOpMode = self.MOpMode.Regular     #  Select magnetometer perfomance mode
        MPwrMode = self.MPwrMode.Normal     #  Select magnetometer power mode
        Modr = self.Modr.MODR_10Hz      #  Select magnetometer ODR when in BNO055 bypass mode
        PWRMode = self.PWRMode.Normalpwr     #  Select BNO055 power mode
        OPRMode = self.OPRMode.NDOF        #
        #  Select BNO055 config mode
        # while True:
        #     try:
        self.write_byte(self.BNO055_OPR_MODE, self.OPRMode.CONFIGMODE )
        time.sleep(0.025)
        #  Select page 1 to configure sensors
        self.write_byte(self.BNO055_PAGE_ID, 0x01)
        #  Configure ACC
        self.write_byte(self.BNO055_ACC_CONFIG, APwrMode << 5 | Abw << 3 | Ascale )
        #  Configure GYR
        self.write_byte(self.BNO055_GYRO_CONFIG_0, Gbw << 3 | Gscale )
        self.write_byte(self.BNO055_GYRO_CONFIG_1, GPwrMode)
        #  Configure MAG
        self.write_byte(self.BNO055_MAG_CONFIG, MPwrMode << 5 | MOpMode << 3 | Modr )
        #  Select page 0 to read sensors
        self.write_byte(self.BNO055_PAGE_ID, 0x00)
        #  Select BNO055 gyro temperature source
        self.write_byte(self.BNO055_TEMP_SOURCE, 0x01 )
        #  Select BNO055 sensor units (temperature in degrees C, rate in dps, accel in mg)
        self.write_byte(self.BNO055_UNIT_SEL, 0x01 )
        #  Select BNO055 system power mode
        self.write_byte(self.BNO055_PWR_MODE, PWRMode )
        #  Select BNO055 system operation mode
        self.write_byte(self.BNO055_OPR_MODE, OPRMode )
        time.sleep(0.025)
                # break

        # ROS Node Initialization
        self.node_name = 'imu_data_acquisition'
        rospy.init_node(self.node_name, anonymous = True)
        self.pub = rospy.Publisher("/imu_data", Imu, queue_size = 1, latch = False)
        rospy.Subscriber("kill_gait_assistance", Bool, self.updateFlagImuAcquisition)
        self.kill_flag = False

    def updateFlagImuAcquisition(self,flag_signal):
        self.kill_flag = flag_signal.data

    """2nd order Butterworth low-pass filter (cutoff frequency: 15 Hz)"""
    def low_pass_filter_15hz(self, in_signal):
        tmp = (in_signal - self.gyro_b1_ * self.gyro_filter_states[0]) - self.gyro_b2_ * self.gyro_filter_states[1];
        filt_signal = (self.gyro_a0_*tmp + self.gyro_a1_*self.gyro_filter_states[0]) + self.gyro_a2_*self.gyro_filter_states[1];

        self.gyro_filter_states[1] = self.gyro_filter_states[0];
        self.gyro_filter_states[0] = tmp;

        return filt_signal

    """2nd order Butterworth low-pass filter (cutoff frequency: 17 Hz)"""
    def low_pass_filter_17hz(self, in_signal):
        tmp = (in_signal - self.accel_b1_ * self.accel_filter_states[0]) - self.accel_b2_ * self.accel_filter_states[1];
        filt_signal = (self.accel_a0_*tmp + self.accel_a1_*self.accel_filter_states[0]) + self.accel_a2_*self.accel_filter_states[1];

        self.accel_filter_states[1] = self.accel_filter_states[0];
        self.accel_filter_states[0] = tmp;

        return filt_signal

    # def read_initial_angle(self):
    #     """Read initial calibration pitch angle.
    #     No return value, it modifies global variables though.
    #     """
    #     angles = []
    #     diff = 2
    #     print("Stand still 5 seconds to measure initial calibration angle...")
    #     time.sleep(5)
    #
    #     while diff <= 2 and len(angles) < 500:
    #         self.read_eul()
    #         # Corresponding angle is pitch euler angle from IMU
    #         angles.append(self.euler['y'])
    #         if len(angles) > 500: del angles[0]
    #         diff = np.std(angles)
    #
    #     self.cal_angle = np.mean(angles)
    #     print("Initial calibration angle: {} degrees".format(self.cal_angle))

    def read_data(self, subAddress):
        """Read count number of 16-bit signed values starting from the provided
        address. Returns a tuple of the values that were read.
        """
        rawData = self.read_bytes(subAddress, 6)
        intX = (rawData[1] << 8) | rawData[0]
        intY = (rawData[3] << 8) | rawData[2]
        intZ = (rawData[5] << 8) | rawData[4]
        x = twos_complement(intX, 16)
        y = twos_complement(intY, 16)
        z = twos_complement(intZ, 16)
        return (x, y, z)

    def read_eul(self, degrees = 0):
        """Return the current absolute orientation as a tuple of heading, roll,
        and pitch euler angles in degrees.
        """
        x, y, z = self.read_data(self.BNO055_EUL_HEADING_LSB)
        if degrees == 0:
            unit = 16.0
        else:   # radians
            unit = 900.0
        self.euler = {'x': x/unit, 'y': y/unit, 'z': z/unit}

    def read_quat(self):
        """Return the current orientation as a tuple of X, Y, Z, W quaternion
        values.
        """
        rawData = self.read_bytes(self.BNO055_QUA_DATA_W_LSB, 8)
        intQW = (rawData[1] << 8) | rawData[0]
        intQX = (rawData[3] << 8) | rawData[2]
        intQY = (rawData[5] << 8) | rawData[4]
        intQZ = (rawData[7] << 8) | rawData[6]
        qw = twos_complement(intQW, 16)/16384.   # 2E14
        qx = twos_complement(intQX, 16)/16384.
        qy = twos_complement(intQY, 16)/16384.
        qz = twos_complement(intQZ, 16)/16384.
        self.quat['x'] = qx
        self.quat['y'] = qy
        self.quat['z'] = qz
        self.quat['w'] = qw

    def read_accel(self):
        """Return the current accelerometer reading as a tuple of X, Y, Z values
        in meters/second^2.
        """
        x, y, z = self.read_data(self.BNO055_ACC_DATA_X_LSB)
        self.accel['x'] = x/100.0
        # self.accel['y'].append(y)
        self.accel['y'] = self.low_pass_filter_17hz(y/100.0)
        # self.accel['y'] = y/100.0
        self.accel['z'] = z/100.0
        # One must have at least 100 samples to start filtering data
        # if(len(self.accel['y']) >= 100):
        #     # Filtering function return np array that needs to be converted into list
        #     self.accel['y'] = self.butter_lowpass_filter(self.accel['y']).tolist()
        #     # Delete first item of each list for accel dicty
        #     del self.accel['y'][0]

    def read_gyro(self):
        """Return the current gyroscope (angular velocity) reading as a tuple of
        X, Y, Z values in degrees per second.
        """
        x, y, z = self.read_data(self.BNO055_GYR_DATA_X_LSB)
        self.gyro['x'] = x/16.0
        # self.gyro['y'].append(y)
        self.gyro['y'] = self.low_pass_filter_15hz(y/16.0)
        # self.gyro['y'] = y/16.0
        self.gyro['z'] = z/16.0
        # One must have at least 100 samples to start filtering data
        # if(len(self.gyro['y']) >= 100):
        #     # Filtering function return np array that needs to be converted into list
        #     self.gyro['y'] = self.butter_lowpass_filter(self.gyro['y']).tolist()
        #     # Delete first item of each list for gyro dicty
        #     del self.gyro['y'][0]
        #     # print('Type: {}'.format(type(self.gyro['y'])))

    def read_mag(self):
        """Return the current magnetometer reading as a tuple of X, Y, Z values
        in micro-Teslas.
        """
        x, y, z = self.read_data(self.BNO055_MAG_DATA_X_LSB)
        self.mag = {'x': x/16.0, 'y': y/16.0, 'z': z/16.0}

    def read_lin_accel(self, ms2 = 0):
        """Return the current linear acceleration (acceleration from movement,
        not from gravity) reading as a tuple of X, Y, Z values in meters/second^2.
        """
        x, y, z = self.read_data(self.BNO055_LIA_DATA_X_LSB)
        if ms2 == 0:
            unit = 100.0
        else:   # mg
            unit = 1.0
        self.linAccel = {'x': x/unit, 'y': y/unit, 'z': z/unit}

    def read_grav(self, ms2 = 0):
        """Return the current gravity acceleration reading as a tuple of X, Y, Z
        values in meters/second^2.
        """
        x, y, z = self.read_data(self.BNO055_GRV_DATA_X_LSB)
        if ms2 == 0:
            unit = 100.0
        else:   # mg
            unit = 1.0
        self.grav = {'x': x/unit, 'y': y/unit, 'z': z/unit}

    def read_temper(self):
        """Return the current temperature in Celsius."""
        c = self.read_bytes(self.BNO055_TEMP, 1)
        f = float(c[0])*1.8+32
        self.temp = {'C': c[0], 'F': f}

    def read_calib(self):
        """Read the calibration status of the sensors and return a 4 tuple with
        calibration status as follows:
          - System, 3=fully calibrated, 0=not calibrated
          - Gyroscope, 3=fully calibrated, 0=not calibrated
          - Accelerometer, 3=fully calibrated, 0=not calibrated
          - Magnetometer, 3=fully calibrated, 0=not calibrated
        """
        calib_stat = self.read_bytes(self.BNO055_CALIB_STAT, 1)
        sys  = ( calib_stat[0] & 0b11000000 ) >> 6    # Take first 2 bits and move them to the end
        gyro = ( calib_stat[0] & 0b00110000 ) >> 4
        acc  = ( calib_stat[0] & 0b00001100 ) >> 2
        mag  = ( calib_stat[0] & 0b00000011 ) >> 0
        # Calibration status, from 0 (not calibrated) to 3 (fully calibrated)
        self.calib_stat = {'sys': sys, 'gyro': gyro, 'acc': acc, 'mag':mag}

    def self_test(self):
        """Self test result register value with the following meaning:
        Bit value: 1 = test passed, 0 = test failed
        Bit 0 = Accelerometer self test
        Bit 1 = Magnetometer self test
        Bit 2 = Gyroscope self test
        Bit 3 = MCU self test
        Value of 0x0F = all good!
        """
        # Switch to configuration mode if running self test.
        self.write_byte(self.BNO055_OPR_MODE, self.OPRMode.CONFIGMODE )
        time.sleep(0.025)
        # Perform a self test.
        sys_trigger = self.read_bytes(self.BNO055_SYS_TRIGGER, 1)[0]
        self.write_byte(self.BNO055_SYS_TRIGGER, sys_trigger | 0x1)
        # Wait for self test to finish.
        time.sleep(1.0)
        # Read test result.
        self_test = self.read_bytes(self.BNO055_ST_RESULT, 1)
        mcu  = ( self_test[0] & 0b00001000 ) >> 3
        gyro = ( self_test[0] & 0b00000100 ) >> 2
        mag  = ( self_test[0] & 0b00000010 ) >> 1
        acc  = ( self_test[0] & 0b00000001 ) >> 0
        # Sensor is working if bit is set to 1
        self.result = {'mcu': mcu, 'gyro': gyro, 'acc': acc, 'mag':mag}
        # Go back to operation mode.
        self.write_byte(self.BNO055_OPR_MODE, self.OPRMode.NDOF )

    def get_revision(self):
        """Get revision numbers
        Returns a tuple with revision numbers for Software revision, Bootloader
        version, Accelerometer ID, Magnetometer ID, and Gyro ID."""
        # Read revision values.
        accel = self.read_bytes(self.BNO055_ACC_ID, 1)[0]
        mag = self.read_bytes(self.BNO055_MAG_ID, 1)[0]
        gyro = self.read_bytes(self.BNO055_GYRO_ID, 1)[0]
        bl = self.read_bytes(self.BNO055_BL_REV_ID, 1)[0]
        sw_lsb = self.read_bytes(self.BNO055_SW_REV_ID_LSB, 1)[0]
        sw_msb = self.read_bytes(self.BNO055_SW_REV_ID_MSB, 1)[0]
        sw = ((sw_msb << 8) | sw_lsb) & 0xFFFF
        # Return the results as a tuple of all 5 values.
        self.revision = {"sw": sw, "bl": bl, "acc": accel, "mag": mag, "gyro": gyro}

    def write_byte(self, subAddress, value):
        """Write an 8-bit value to the provided register address.  If ack is True
        then expect an acknowledgement in serial mode, otherwise ignore any
        acknowledgement (necessary when resetting the device).
        """
        self.bus.write_byte_data(self.address, subAddress, value)

    def read_bytes(self, subAddress, count):
        """Read a number of unsigned byte values starting from the provided address.
        """
        return self.bus.read_i2c_block_data(self.address, subAddress, count)

    #  BNO055 Register Map
    #  Datasheet: https:# www.bosch-sensortec.com/en/homepage/products_3/sensor_hubs/iot_solutions/bno055_1/bno055_4
    #  BNO055 Page 0
    BNO055_CHIP_ID          = 0x00    #  should be 0xA0
    BNO055_ACC_ID           = 0x01    #  should be 0xFB
    BNO055_MAG_ID           = 0x02    #  should be 0x32
    BNO055_GYRO_ID          = 0x03    #  should be 0x0F
    BNO055_SW_REV_ID_LSB    = 0x04
    BNO055_SW_REV_ID_MSB    = 0x05
    BNO055_BL_REV_ID        = 0x06
    BNO055_PAGE_ID          = 0x07
    BNO055_ACC_DATA_X_LSB   = 0x08
    BNO055_ACC_DATA_X_MSB   = 0x09
    BNO055_ACC_DATA_Y_LSB   = 0x0A
    BNO055_ACC_DATA_Y_MSB   = 0x0B
    BNO055_ACC_DATA_Z_LSB   = 0x0C
    BNO055_ACC_DATA_Z_MSB   = 0x0D
    BNO055_MAG_DATA_X_LSB   = 0x0E
    BNO055_MAG_DATA_X_MSB   = 0x0F
    BNO055_MAG_DATA_Y_LSB   = 0x10
    BNO055_MAG_DATA_Y_MSB   = 0x11
    BNO055_MAG_DATA_Z_LSB   = 0x12
    BNO055_MAG_DATA_Z_MSB   = 0x13
    BNO055_GYR_DATA_X_LSB   = 0x14
    BNO055_GYR_DATA_X_MSB   = 0x15
    BNO055_GYR_DATA_Y_LSB   = 0x16
    BNO055_GYR_DATA_Y_MSB   = 0x17
    BNO055_GYR_DATA_Z_LSB   = 0x18
    BNO055_GYR_DATA_Z_MSB   = 0x19
    BNO055_EUL_HEADING_LSB  = 0x1A
    BNO055_EUL_HEADING_MSB  = 0x1B
    BNO055_EUL_ROLL_LSB     = 0x1C
    BNO055_EUL_ROLL_MSB     = 0x1D
    BNO055_EUL_PITCH_LSB    = 0x1E
    BNO055_EUL_PITCH_MSB    = 0x1F
    BNO055_QUA_DATA_W_LSB   = 0x20
    BNO055_QUA_DATA_W_MSB   = 0x21
    BNO055_QUA_DATA_X_LSB   = 0x22
    BNO055_QUA_DATA_X_MSB   = 0x23
    BNO055_QUA_DATA_Y_LSB   = 0x24
    BNO055_QUA_DATA_Y_MSB   = 0x25
    BNO055_QUA_DATA_Z_LSB   = 0x26
    BNO055_QUA_DATA_Z_MSB   = 0x27
    BNO055_LIA_DATA_X_LSB   = 0x28
    BNO055_LIA_DATA_X_MSB   = 0x29
    BNO055_LIA_DATA_Y_LSB   = 0x2A
    BNO055_LIA_DATA_Y_MSB   = 0x2B
    BNO055_LIA_DATA_Z_LSB   = 0x2C
    BNO055_LIA_DATA_Z_MSB   = 0x2D
    BNO055_GRV_DATA_X_LSB   = 0x2E
    BNO055_GRV_DATA_X_MSB   = 0x2F
    BNO055_GRV_DATA_Y_LSB   = 0x30
    BNO055_GRV_DATA_Y_MSB   = 0x31
    BNO055_GRV_DATA_Z_LSB   = 0x32
    BNO055_GRV_DATA_Z_MSB   = 0x33
    BNO055_TEMP             = 0x34
    BNO055_CALIB_STAT       = 0x35
    BNO055_ST_RESULT        = 0x36
    BNO055_INT_STATUS       = 0x37
    BNO055_SYS_CLK_STATUS   = 0x38
    BNO055_SYS_STATUS       = 0x39
    BNO055_SYS_ERR          = 0x3A
    BNO055_UNIT_SEL         = 0x3B
    BNO055_OPR_MODE         = 0x3D
    BNO055_PWR_MODE         = 0x3E
    BNO055_SYS_TRIGGER      = 0x3F
    BNO055_TEMP_SOURCE      = 0x40
    BNO055_AXIS_MAP_CONFIG  = 0x41
    BNO055_AXIS_MAP_SIGN    = 0x42
    BNO055_ACC_OFFSET_X_LSB = 0x55
    BNO055_ACC_OFFSET_X_MSB = 0x56
    BNO055_ACC_OFFSET_Y_LSB = 0x57
    BNO055_ACC_OFFSET_Y_MSB = 0x58
    BNO055_ACC_OFFSET_Z_LSB = 0x59
    BNO055_ACC_OFFSET_Z_MSB = 0x5A
    BNO055_MAG_OFFSET_X_LSB = 0x5B
    BNO055_MAG_OFFSET_X_MSB = 0x5C
    BNO055_MAG_OFFSET_Y_LSB = 0x5D
    BNO055_MAG_OFFSET_Y_MSB = 0x5E
    BNO055_MAG_OFFSET_Z_LSB = 0x5F
    BNO055_MAG_OFFSET_Z_MSB = 0x60
    BNO055_GYR_OFFSET_X_LSB = 0x61
    BNO055_GYR_OFFSET_X_MSB = 0x62
    BNO055_GYR_OFFSET_Y_LSB = 0x63
    BNO055_GYR_OFFSET_Y_MSB = 0x64
    BNO055_GYR_OFFSET_Z_LSB = 0x65
    BNO055_GYR_OFFSET_Z_MSB = 0x66
    BNO055_ACC_RADIUS_LSB   = 0x67
    BNO055_ACC_RADIUS_MSB   = 0x68
    BNO055_MAG_RADIUS_LSB   = 0x69
    BNO055_MAG_RADIUS_MSB   = 0x6A
    #  BNO055 Page 1
    BNO055_PAGE_ID          = 0x07
    BNO055_ACC_CONFIG       = 0x08
    BNO055_MAG_CONFIG       = 0x09
    BNO055_GYRO_CONFIG_0    = 0x0A
    BNO055_GYRO_CONFIG_1    = 0x0B
    BNO055_ACC_SLEEP_CONFIG = 0x0C
    BNO055_GYR_SLEEP_CONFIG = 0x0D
    BNO055_INT_MSK          = 0x0F
    BNO055_INT_EN           = 0x10
    BNO055_ACC_AM_THRES     = 0x11
    BNO055_ACC_INT_SETTINGS = 0x12
    BNO055_ACC_HG_DURATION  = 0x13
    BNO055_ACC_HG_THRESH    = 0x14
    BNO055_ACC_NM_THRESH    = 0x15
    BNO055_ACC_NM_SET       = 0x16
    BNO055_GYR_INT_SETTINGS = 0x17
    BNO055_GYR_HR_X_SET     = 0x18
    BNO055_GYR_DUR_X        = 0x19
    BNO055_GYR_HR_Y_SET     = 0x1A
    BNO055_GYR_DUR_Y        = 0x1B
    BNO055_GYR_HR_Z_SET     = 0x1C
    BNO055_GYR_DUR_Z        = 0x1D
    BNO055_GYR_AM_THRESH    = 0x1E
    BNO055_GYR_AM_SET       = 0x1F

    #  Set initial input parameters
    #  ACC Full Scale
    Ascale = enum('AFS_2G', 'AFS_4G', 'AFS_8G', 'AFS_18G')
    #  ACC Bandwidth
    Abw = enum('ABW_7_81Hz', 'ABW_15_63Hz', 'ABW_31_25Hz', 'ABW_62_5Hz', 'ABW_125Hz', 'ABW_250Hz', 'ABW_500Hz', 'ABW_1000Hz')
    #  ACC Pwr Mode
    APwrMode = enum('NormalA', 'SuspendA', 'LowPower1A', 'StandbyA', 'LowPower2A', 'DeepSuspendA')
    #  Gyro full scale
    Gscale = enum('GFS_2000DPS', 'GFS_1000DPS', 'GFS_500DPS', 'GFS_250DPS', 'GFS_125DPS')
    #  GYR Pwr Mode
    GPwrMode = enum('NormalG', 'FastPowerUpG', 'DeepSuspendedG', 'SuspendG', 'AdvancedPowerSaveG')
    #  Gyro bandwidth
    Gbw = enum('GBW_523Hz', 'GBW_230Hz', 'GBW_116Hz', 'GBW_47Hz', 'GBW_23Hz', 'GBW_12Hz', 'GBW_64Hz', 'GBW_32Hz')
    #  BNO-55 operation modes
    OPRMode = enum('CONFIGMODE', 'ACCONLY', 'MAGONLY', 'GYROONLY', 'ACCMAG', 'ACCGYRO', 'MAGGYRO', 'AMG', 'IMU', 'COMPASS', 'M4G', 'NDOF_FMC_OFF', 'NDOF')
    # OPRMode ={'CONFIGMODE': 0x00, 'ACCONLY': 0x01, 'MAGONLY': 0x02, 'GYROONLY': 0x03, 'ACCMAG': 0x04, 'ACCGYRO': 0x05, 'MAGGYRO': 0x06, 'AMG': 0x07, 'IMU': 0x08, 'COMPASS': 0x09, 'M4G': 0x0A, 'NDOF_FMC_OFF': 0x0B, 'NDOF': 0x0C}
    #  Power mode
    PWRMode = enum('Normalpwr', 'Lowpower', 'Suspendpwr')
    #  Magnetometer output data rate
    Modr = enum('MODR_2Hz', 'MODR_6Hz', 'MODR_8Hz', 'MODR_10Hz', 'MODR_15Hz', 'MODR_20Hz', 'MODR_25Hz', 'MODR_30Hz')
    #  MAG Op Mode
    MOpMode = enum('LowPower', 'Regular', 'EnhancedRegular', 'HighAccuracy')
    #  MAG power mod
    MPwrMode = enum('Normal', 'Sleep', 'Suspend', 'ForceMode')
    Posr = enum('P_OSR_00', 'P_OSR_01', 'P_OSR_02', 'P_OSR_04', 'P_OSR_08', 'P_OSR_16')
    Tosr = enum('T_OSR_00', 'T_OSR_01', 'T_OSR_02', 'T_OSR_04', 'T_OSR_08', 'T_OSR_16')
    #  bandwidth at full to 0.021 x sample rate
    IIRFilter = enum('full', 'BW0_223ODR', 'BW0_092ODR', 'BW0_042ODR', 'BW0_021ODR')
    Mode = enum('BMP280Sleep', 'forced', 'forced2', 'normal')
    SBy = enum('t_00_5ms', 't_62_5ms', 't_125ms', 't_250ms', 't_500ms', 't_1000ms', 't_2000ms', 't_4000ms',)

def main():
    sensor = IMU_BNO055(bus=1, address=0x29)

    # Parameters of ROS message
    msg = Imu()
    rate = rospy.Rate(100)   # 100 Hz
    start_time = time.time()

    """Read IMU calibration status"""
    sensor.read_calib()
    if (sensor.calib_stat["sys"]==3 and sensor.calib_stat["gyro"]==3 and sensor.calib_stat["acc"]==3 and sensor.calib_stat["mag"]==3):
        print("IMU is fully calibrated.")
    else:
        print("IMU is NOT fully calibrated.")
        if sensor.calib_stat["sys"]==3:
            print("System is fully calibrated.")
        else:
            print("System is NOT fully calibrated.")
        if sensor.calib_stat["gyro"]==3:
            print("Gyroscope is fully calibrated.")
        else:
            print("Gyroscope is NOT fully calibrated.")
        if sensor.calib_stat["acc"]==3:
            print("Accelerometer is fully calibrated.")
        else:
            print("Accelerometer is NOT fully calibrated.")
        if sensor.calib_stat["mag"]==3:
            print("Magnetometer is fully calibrated.")
        else:
            print("Magnet Float64,ometer is NOT fully calibrated.")

    """Perform self test on IMU"""
    print("Performing self-test on IMU...")
    sensor.self_test()
    if (sensor.result["mcu"]==1 and sensor.result["gyro"]==1 and sensor.result["acc"]==1 and sensor.result["mag"]==1):
        print("Test passed for entire IMU")
    else:
        print("Test NOT passed for entire IMU")

    # Print BNO055 software revision and other diagnostic data.
    sensor.get_revision()
    print('Software version:   {0}'.format(sensor.revision["sw"]))
    print('Bootloader version: {0}'.format(sensor.revision["bl"]))
    print('Accelerometer ID:   0x{0:02X}'.format(sensor.revision["acc"]))
    print('Magnetometer ID:    0x{0:02X}'.format(sensor.revision["mag"]))
    print('Gyroscope ID:       0x{0:02X}\n'.format(sensor.revision["gyro"]))

    print('Reading BNO055 data, press Ctrl-C to quit...')

    # # Reading initial calibration angle
    # sensor.read_initial_angle()

    while not rospy.is_shutdown():
        if not sensor.kill_flag:
            try:
                # Read the Euler angles for heading, roll, pitch (all in degrees).
                # sensor.read_eul()
                # Read the calibration status, 0=uncalibrated and 3=fully calibrated.
                # sys, gyro, accel, mag = sensor.read_calib()
                # Other values you can optionally read:
                # Orientation as a quaternion:
                sensor.read_quat()
                # Sensor temperature in degrees Celsius:
                #temp_c = sensor.read_temper()
                # Magnetometer data (in micro-Teslas):
                #x,y,z = sensor.read_mag()
                # Gyroscope data (in degrees per second):
                sensor.read_gyro()
                # Accelerometer data (in meters per second squared):
                sensor.read_accel()
                # Linear acceleration data (i.e. acceleration from movement, not gravity--
                # returned in meters per second squared):
                #x,y,z = sensor.read_lin_accel()
                # Gravity acceleration data (i.e. acceleration just from gravity--returned
                # in meters per second squared):
                #x,y,z = sensor.read_grav()
                # Print everything out.
                # print('time_stamp={}\t gyro_X={:.2f} gyro_Y={:.2f} gyro_Z={:.2f}\n\t accel_X={:.2f} accel_Y={:.2f} accel_Z={:.2f}'.format(
                #     rospy.get_rostime().nsecs, sensor.gyro['x'], sensor.gyro['y'], sensor.gyro['z'], sensor.accel['x'], sensor.accel['y'], sensor.accel['z']))
                # time.sleep(1)

                # # Computing angular difference in respect of initial calibration angle (pitch frame)
                # knee_angle = abs(sensor.cal_angle - sensor.euler['y'])

                # Transmission of ROS message
                msg.header.frame_id = "/" + sensor.node_name
                # msg.time_stamp = int(round((time.time() - start_time)*1000.0))
                msg.angular_velocity.x = sensor.gyro['x']
                msg.angular_velocity.y = sensor.gyro['y']
                msg.angular_velocity.z = sensor.gyro['z']
                msg.linear_acceleration.x = sensor.accel['x']
                msg.linear_acceleration.y = sensor.accel['y']
                msg.linear_acceleration.z = sensor.accel['z']
                msg.orientation.x = sensor.quat['x']
                msg.orientation.y = sensor.quat['y']
                msg.orientation.z = sensor.quat['z']
                msg.orientation.w = sensor.quat['w']
                sensor.pub.publish(msg)
            except:
                rospy.logwarn("Lost connection!")
            rate.sleep()
        else:
            break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("Program finished\n")
        sys.stdout.close()
        os.system('clear')
        raise
