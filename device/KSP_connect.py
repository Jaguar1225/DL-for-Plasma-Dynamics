from ctypes import *
import enum

import numpy as np

path = 'C:/Users/user/Desktop/UI Test/SPdbEthm.dll'
c_module = windll.LoadLibrary(path)

class SPError(enum.IntEnum):
    SP_NO_ERROR = 1
    SP_ERROR_DEVICE_IO_CTRL = -101
    SP_ERROR_OPEN_DRIVER = -102
    SP_ERROR_OPEN_FILE = -103
    SP_ERROR_EEPROM_READ = -104
    SP_ERROR_MEMORY_ALLOC = -105
    SP_ERROR_AUTODARK = -106
    SP_ERROR_NOTSUPORT_DEV = -107
    SP_ERROR_INVALIDHANDLE = -108
    SP_ERROR_INPUT_PARAM = -109
    SP_ERROE_SHUTTER_VALUE = -110
    SP_ERROE_FW_POSITION = -111
    SP_EEPROM_READ_ERROR = -112
    SP_ERROR_EEPROM_EMPTY = -113
    SP_ERROR_DATA_LACK = -114
    SP_ERROR_NOTFINDDEVICE = -115
    SP_ERROR_ALREAYDOPEN = -116
    SP_ERROR_WAIT_TIMEOUT = -117
    SP_ERROR_SCANNUM_RANGE = -118
    SP_ERROR_SMOOTH_RANGE = -119
    SP_ERROR_INVALIDVALUE = -120
    SP_ERROR_INVALID_INPUTCHANNEL = -121
    SP_ERROR_CHECKINTRERFACE = -122
    SP_ERROR_COM_SETTING = -201
    SP_ERROR_COM_READ = -202
    SP_ERROR_COM_WRITE = -203
    SP_ERROR_COM_OPERATING = -204
    SP_ERROR_COM_NOTMATCHDEV = -205
    SP_ERROR_COM_NOTCONNECTION = -206
    SP_ERROR_ETH_IPSCAN = -301
    SP_ERROR_ETH_NOTMATCHDEV = -302
    SP_ERROR_ETH_SOCKETCREATE = -303
    SP_ERROR_ETH_SOCKETCONNECT = -304
    SP_ERROR_ETH_TIMEOUT = -305
    SP_ERROR_ETH_RECVBUFFER = -307
    SP_EEROR_ETH_DISCONNECT = -308
    SP_ERROR_ETH_SENDPACKET = -309
    SP_ERROR_ETH_PACKET_SIZE_SMALL = -310

class SPCCDType(enum.IntEnum):
    SP_CCD_SONY = 0
    SP_CCD_TOSHIBA = 1
    SP_CCD_PDA = 2
    SP_CCD_G9212 = 3
    SP_CCD_S10420 = 4
    SP_CCD_G92XX_256 = 5
    SP_CCD_S10141 = 6
    SP_CCD_TCD1304AP = 7

class SPCCDPixel(enum.IntEnum):
    SP_CCD_PIXEL_G92XX_256 = 256
    SP_CCD_PIXEL_G92XX_256_REAL = 256
    SP_CCD_PIXEL_G9212 = 512
    SP_CCD_PIXEL_G9212_REAL = 512
    SP_CCD_PIXEL_PDA = 1056
    SP_CCD_PIXEL_PDA_REAL = 1024
    SP_CCD_PIXEL_SONY = 2080
    SP_CCD_PIXEL_SONY_REAL = 2048
    SP_CCD_PIXEL_TOSHIBA = 3680
    SP_CCD_PIXEL_TOSHIBA_REAL = 3648
    SP_CCD_PIXEL_S10420 = 2080
    SP_CCD_PIXEL_S10420_REAL = 2048
    SP_CCD_PIXEL_S10141 = 2080
    SP_CCD_PIXEL_S10141_REAL = 2048
    SP_CCD_PIXEL_TCD1304AP = 3680
    SP_CCD_PIXEL_TCD1304AP_REAL = 3648

class SPInterface(enum.IntEnum):
    SP_INTERFACE_USB = 0
    SP_INTERFACE_ETHERNET = 1

class SPScreenMode(enum.IntEnum):
    SP_SCAN_ALLDEVICE = 0
    SP_SCAN_CONNECTABLE = 1

class SPTriggerMode(enum.IntEnum):
    SP_TRIGGER_FREERUN_NEXT = 1
    SP_TRIGGER_FREERUN_PREV = 2
    SP_TRIGGER_SOFTWARE = 3
    SP_TRIGGER_EXTERNAL = 4

class SPNetMode(enum.IntEnum):
    SP_NETMODE_DHCP = 0
    SP_NETMODE_STATIC = 1

class DeviceList(Structure):
    _fields_ = [
        ('sInterfaceType', c_short),
        ('cCOM', c_char * 100),
        ('cModel', c_char * 100),
        ('cSerial', c_char * 100),
        ('cIPAddr', c_char * 100)
    ]

class DevInformation(Structure):
    _fields_ = [
        ('strCOM', c_char * 100),
        ('strModel', c_char * 100),
        ('strSerial', c_char * 100),
        ('strIPAddr', c_char * 100),
        ('strStaticIPAddr', c_char * 100),
        ('strMACAddr', c_char * 100),
        ('dWLTable', c_double * 3648),
        ('iDummyPixelNum', c_int),
        ('iInttime', c_int),
        ('iTimeavg', c_int),
        ('iTotPixelNum', c_int),
        ('iRealPixelNum', c_int),
        ('sTrgMode', c_short),
        ('sInterfaceType', c_short),
        ('sNetMode', c_short),
        ('sChannel', c_short),
        ('sCCDType', c_short)
    ]
class MEMORYSTATE(Structure):
    _fields_ = [
        ('sMemstate', c_short),
        ('MemName', c_char*100)
    ]

class KSPDeviceControl():
    def __init__(self,scan_mode = SPScreenMode.SP_SCAN_CONNECTABLE):
        self.num_device = 0
        self.scan_mode = scan_mode
    def Connect(self):
        self.NDeviceScan(scan_mode=self.scan_mode)
        self.NGetDeviceList()
        self.NConnect()
        self.NGetDevParam()
        self.NDevInfo()
        self.NGetEEPROM()
        self.NGetCCDType()
        self.NGetWLTable()

    def NDeviceScan(self,scan_mode = SPScreenMode.SP_SCAN_CONNECTABLE):
        self.num_device = dll_Functions['spNScanDevice'](scan_mode)
        if self.num_device <= 0:
            print('Device disabled!')
            self.ErrorAlarm(self.num_device, self.NDeviceScan)
    
    def NGetDeviceList(self):
        self.device_list = DeviceList()
        Err = dll_Functions['spNGetDeviceList'](byref(self.device_list))
        self.ErrorAlarm(Err, self.NGetDeviceList)
    
    def NConnect(self):
        if self.device_list.sInterfaceType:
            Err = dll_Functions['spNConnect'](self.device_list.sInterfaceType, self.device_list.cIPAddr)
        else:
            Err = dll_Functions['spNConnect'](self.device_list.sInterfaceType, self.device_list.cCOM)
        self.ErrorAlarm(Err, self.NConnect)
        if Err >= 0:
            self.channel = Err
    
    def NGetDevParam(self):
        self.dev_info = DevInformation()
        Err = dll_Functions['spNGetDevParam'](byref(self.dev_info), self.channel)
        self.ErrorAlarm(Err, self.NGetDevParam)

    def NDevInfo(self):
        self.model_info = (c_char * 30)()
        self.serial_info = (c_char * 30)()
        self.interface_type = c_short()
        Err = dll_Functions['spNDevInfo'](
            self.model_info, 
            self.serial_info,
            byref(self.interface_type),
            self.channel
            )
        self.ErrorAlarm(Err, self.NDevInfo)
    
    def NGetEEPROM(self):
        self.eeprom_data = (c_char*1024)()
        Err = dll_Functions['spNGetEEPROM'](self.eeprom_data,self.channel)
        self.ErrorAlarm(Err, self.NGetEEPROM)

    def NGetCCDType(self):
        CCDType = c_short()
        Err = dll_Functions['spNGetCCDType'](byref(CCDType), self.channel)
        self.ErrorAlarm(Err, self.NGetCCDType)
        for ccd_type in SPCCDType:
            if ccd_type.value == CCDType.value:
                self.CCDType = ccd_type.name
                PIXELNAME = self.CCDType.split('_')
                PIXELNAME.insert(2,'PIXEL')
                self.wl_num = SPCCDPixel[('_').join(PIXELNAME)].value
                PIXELNAME.append('REAL')
                self.wl_real_num = SPCCDPixel[('_').join(PIXELNAME)].value
                break
            else:
                pass
        
    def NGetWLTable(self):
        self.wl_table = (c_double * self.wl_num)()
        Err = dll_Functions['spNGetWLTable'](self.wl_table,self.channel)
        self.ErrorAlarm(Err, self.NGetWLTable)
        self.wl_table_np = np.frombuffer(self.wl_table,dtype=float)[:self.wl_real_num]
        
    def NSetIntTime(self, int_time):
        Err = dll_Functions['spNSetIntTime'](int_time,self.channel)
        self.ErrorAlarm(Err, self.NSetIntTime)
    
    def NSetTimeAvg(self,time_avg):
        Err = dll_Functions['spNSetTimeAvg'](time_avg, self.channel)
        self.ErrorAlarm(Err, self.NSetTimeAvg)
    
    def NSetTrgMode(self, trg_mode):
        Err = dll_Functions['spNSetTrgMode'](trg_mode,self.channel)
        self.ErrorAlarm(Err, self.NSetTrgMode)
        
    def NSetDevice(self, int_time, avg_time, trg_mode):
        Err = dll_Functions['spNSetDevice'](int_time, avg_time, trg_mode, self.channel)
        self.ErrorAlarm(Err, self.NSetDevice)

    def NGetNetInfo(self):
        self.IPAddr = c_char()
        self.MACAddr = c_char()
        self.NetMode = c_short()
        Err = dll_Functions['spNGetNetInfo'](byref(self.IPAddr),byref(self.MACAddr),byref(self.NetMode),self.channel)
        self.ErrorAlarm(Err, self.NGetNetInfo)

    def NGetGainOffsetValue(self):
        self.gain_value = c_int()
        self.offset_value = c_int()
        Err = dll_Functions['spNGetGainOffsetValue'](self.gain_value, self.offset_value, self.channel)
        self.ErrorAlarm(Err, self.NGetGainOffsetValue)
    
    def NReadDataEx(self):
        data = (c_long*self.wl_num)()
        Err = dll_Functions['spNReadDataEx'](data, self.channel)
        self.ErrorAlarm(Err, self.NReadDataEx)
        return data

    def NCheckConnection(self):
        Err = dll_Functions['spNCheckConnection'](self.channel)
        print(Err)
        self.ErrorAlarm(Err,self.NCheckConnection)

    def NGetMemState(self, Mempage):
        self.MemArray = MEMORYSTATE()
        Err = dll_Functions['spNGetMemState'](Mempage,byref(self.MemArray),self.channel)
        self.ErrorAlarm(Err,self.NGetMemState)

    def NGetMemData(self, Mempage, Address):
        self.DataArray = c_long()
        Err = dll_Functions['spNGetMemData'](Mempage,Address,byref(self.DataArray),self.channel)
        self.ErrorAlarm(Err,self.NGetMemData)
        print(self.DataArray)
        return self.DataArray

    def NSaveDataToMem(self, Mempage, Address, DataArray=None):
        Name = c_char()
        if DataArray==None:
            Err = dll_Functions['spNSaveDataToMem'](Mempage,Address,byref(Name),byref(self.DataArray),self.channel)
        else:
            Err = dll_Functions['spNSaveDataToMem'](Mempage,Address,byref(Name),byref(DataArray),self.channel)
        self.ErrorAlarm(Err, self.NSaveDataToMem)
        print(Name)

    def NDeleteMemData(self, Mempage, Address):
        Err = dll_Functions['spNDeleteMemData'](Mempage,Address,self.channel)
        self.ErrorAlarm(Err,self.NDeleteMemData)

    def NStartBurstData(self, Count):
        Err = dll_Functions['spNStartBurstData'](Count, self.channel)
        self.ErrorAlarm(Err, self.NStartBurstData)

    def NGetBurstData(self, Count):
        DataList = (POINTER(c_long) * Count)()
        for i in range(Count):
            DataList[i] = (c_long * self.wl_num)()
        Err = dll_Functions['spNGetBurstData'](Count, DataList, 0, self.channel)
        self.ErrorAlarm(Err, self.NGetBurstData)
        return DataList

    def NAutoDark(self, auto_dark):
        Err = dll_Functions['spNAutoDark'](auto_dark, self.channel)
        self.ErrorAlarm(Err, self.NAutoDark)

    def NDevClose(self):
        Err = dll_Functions['spNDevClose'](self.channel)
        self.ErrorAlarm(Err, self.NDevClose)
        
    def ErrorAlarm(self,Err,f):
        if Err <0:
            for error_enum in SPError:
                if error_enum.value == Err:
                    error_message = error_enum.name
                    print(f'Error Occured {f.__name__} {error_message}')
                    break
        else:
            pass


_fNDeviceScan = c_module.spNScanDevice
_fNScanDevice_IP = c_module.spNScanDevice_IP
_fNGetDeviceList = c_module.spNGetDeviceList
_fNConnect = c_module.spNConnect
_fNCheckConnection = c_module.spNCheckConnection
_fNGetDevParam = c_module.spNGetDevParam
_fNDevInfo = c_module.spNDevInfo
_fNGetCCDType = c_module.spNGetCCDType
_fNGetEEPROM = c_module.spNGetEEPROM
_fNGetWLTable = c_module.spNGetWLTable
_fNSetIntTime = c_module.spNSetIntTime
_fNSetTimeAvg = c_module.spNSetTimeAvg
_fNSetTrgMode = c_module.spNSetTrgMode
_fNSetDevice = c_module.spNSetDevice
_fNGetNetInfo = c_module.spNGetNETInfo
_fNGetGainOffsetValue = c_module.spNGetGainOffsetValue
_fNReadDataEx = c_module.spNReadDataEx
_fNGetMemState = c_module.spNGetMemState
_fNGetMemData = c_module.spNGetMemData
_fNSaveDataToMem = c_module.spNSaveDataToMem
_fNStartBurstData = c_module.spNStartBurstData
_fNGetBurstSingleData = c_module.spNGetBurstSingleData
_fNGetBurstData = c_module.spNGetBurstData
_fNDeleteMemData = c_module.spNDeleteMemData
_fNAutoDark = c_module.spNAutoDark
_fNDevClose = c_module.spNDevClose

dll_Functions_input = {
    'spNScanDevice':(c_short,),
    'spNScanDevcie_IP':(c_char_p,c_short,c_short),
    'spNGetDeviceList':(POINTER(DeviceList),),
    'spNConnect': (c_short, c_char_p),
    'spNCheckConnection':(c_short,),
    'spNGetDevParam':(POINTER(DevInformation),c_short),
    'spNDevInfo':(c_char_p, c_char_p, POINTER(c_short), c_short),
    'spNGetCCDType':(POINTER(c_short),c_short),
    'spNGetEEPROM':(c_char_p, c_short),
    'spNGetWLTable':(POINTER(c_double),c_short),
    'spNSetIntTime':(c_int,c_short),
    'spNSetTimeAvg':(c_short, c_short),
    'spNSetTrgMode':(c_short, c_short),
    'spNSetDevice':(c_int, c_short, c_short, c_short),
    'spNGetNetInfo':(c_char_p,c_char_p,POINTER(c_short),c_short),
    'spNGetGainOffsetValue':(POINTER(c_int),POINTER(c_int),c_short),
    'spNReadDataEx':(POINTER(c_long),c_short),
    'spNGetMemState':(c_short, POINTER(MEMORYSTATE),c_short),
    'spNGetMemData':(c_short, c_byte, POINTER(c_long), c_short),
    'spNSaveDataToMem':(c_short, c_byte, c_char_p, POINTER(c_long), c_short),
    'spNDeleteMemData':(c_short,c_byte,c_short),
    'spNStartBurstData':(c_short,c_short),
    'spNGetBurstData':(c_short, POINTER(POINTER(c_long)), c_short, c_short),
    'spNAutoDark':(c_short, c_short),
    'spNDevClose':(c_short,)
    }

dll_Functions_output = {
    'spNScanDevice':c_short,
    'spNScanDevice_IP':c_short,
    'spNGetDeviceList':c_short,
    'spNConnect':c_short,
    'spNCheckConnection':c_short,
    'spNGetDevParam':c_short,
    'spNDevInfo':c_short,
    'spNGetCCDType':c_short,
    'spNGetEEPROM':c_short,
    'spNGetWLTable':c_short,
    'spNSetIntTime':c_short,
    'spNSetTimeAvg':c_short,
    'spNSetTrgMode':c_short,
    'spNSetDevice':c_short,
    'spNGetNetInfo':c_short,
    'spNGetGainOffsetValue':c_short,
    'spNReadDataEx':c_short,
    'spNGetMemState':c_short,
    'spNGetMemData':c_short,
    'spNSaveDataToMem':c_short,
    'spNDeleteMemData':c_short,
    'spNStartBurstData':c_short,
    'spNGetBurstData':c_short,
    'spNAutoDark':c_short,
    'spNDevClose':c_short
    }

dll_Functions = {
    'spNScanDevice':_fNDeviceScan,
    'spNGetDeviceList':_fNGetDeviceList,
    'spNConnect':_fNConnect,
    'spNCheckConnection':_fNCheckConnection,
    'spNGetDevParam':_fNGetDevParam,
    'spNDevInfo':_fNDevInfo,
    'spNGetCCDType':_fNGetCCDType,
    'spNGetEEPROM':_fNGetEEPROM,
    'spNGetWLTable':_fNGetWLTable,
    'spNSetIntTime':_fNSetIntTime,
    'spNSetTimeAvg':_fNSetTimeAvg,
    'spNSetTrgMode':_fNSetTrgMode,
    'spNSetDevice':_fNSetDevice,
    'spNGetNetInfo':_fNGetNetInfo,
    'spNGetGainOffsetValue':_fNGetGainOffsetValue,
    'spNReadDataEx':_fNReadDataEx,
    'spNGetMemState':_fNGetMemState,
    'spNGetMemData':_fNGetMemData,
    'spNSaveDataToMem':_fNSaveDataToMem,
    'spNDeleteMemData':_fNDeleteMemData,
    'spNStartBurstData': _fNStartBurstData,
    'spNGetBurstData': _fNGetBurstData,
    'spNAutoDark':_fNAutoDark,
    'spNDevClose':_fNDevClose
    }

for k,f in dll_Functions.items():
    try:
        f.argtypes = dll_Functions_input[k]
    except KeyError:
        print(f'{f.__name__} argtypes')
    try:
        f.restype = dll_Functions_output[k]
    except KeyError:
        print(f'{f.__name__} restype')
