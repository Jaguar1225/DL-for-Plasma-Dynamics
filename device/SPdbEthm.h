#pragma once

#define DLLIMPORT extern "C" __declspec(dllimport)
#define	CALLTYPE	__stdcall

enum {
	SP_CCD_SONY		 = 0,
	SP_CCD_TOSHIBA   = 1,
	SP_CCD_PDA		 = 2,		// Back thinned CCD
	SP_CCD_G9212	 = 3,		// InGaAs
	SP_CCD_S10420	 = 4,		// Non Cooled BT Back thinned CCD
	SP_CCD_G92XX_256 = 5,
	SP_CCD_S10141	 = 6,
	SP_CCD_TCD1304AP = 7
};

enum {
	SP_CCD_PIXEL_G92XX_256		= 256,	// InGaAs
	SP_CCD_PIXEL_G92XX_256_REAL = 256,
	SP_CCD_PIXEL_G9212		    = 512,	// InGaAs
	SP_CCD_PIXEL_G9212_REAL		= 512,
	SP_CCD_PIXEL_PDA			= 1056,	// Back thinned CCD, it has to be a multiple of "32"
	SP_CCD_PIXEL_PDA_REAL		= 1024,
	SP_CCD_PIXEL_SONY			= 2080,
	SP_CCD_PIXEL_SONY_REAL		= 2048,
	SP_CCD_PIXEL_TOSHIBA		= 3680,
	SP_CCD_PIXEL_TOSHIBA_REAL	= 3648,
	SP_CCD_PIXEL_S10420			= 2080,	// Non Cooled Back thined
	SP_CCD_PIXEL_S10420_REAL	= 2048,
	SP_CCD_PIXEL_S10141			= 2080,
	SP_CCD_PIXEL_S10141_REAL	= 2048,
	SP_CCD_PIXEL_TCD1304AP		= 3680,
	SP_CCD_PIXEL_TCD1304AP_REAL = 3648
};

enum {
	SP_INTERFACE_USB	    = 0,
	SP_INTERFACE_ETHERNET   = 1
};

enum {
	SP_SCAN_ALLDEVICE		= 0,
	SP_SCAN_CONNECTABLE		= 1,
};

enum {	
	SP_TRIGGER_FREERUN_NEXT	= 1,
	SP_TRIGGER_FREERUN_PREV = 2,
	SP_TRIGGER_SOFTWARE		= 3,
	SP_TRIGGER_EXTERNAL		= 4
};

enum {
	SP_NETMODE_DHCP = 0,
	SP_NETMODE_STATIC = 1
};

enum {
	SP_NO_ERROR						= 1,
	SP_ERROR_DEVICE_IO_CTRL			= -101,
	SP_ERROR_OPEN_DRIVER			= -102,
	SP_ERROR_OPEN_FILE				= -103,
	SP_ERROR_EEPROM_READ			= -104,
	SP_ERROR_MEMORY_ALLOC			= -105,
	SP_ERROR_AUTODARK				= -106,
	SP_ERROR_NOTSUPORT_DEV			= -107,
	SP_ERROR_INVALIDHANDLE			= -108,
	SP_ERROR_INPUT_PARAM			= -109,
	SP_ERROE_SHUTTER_VALUE			= -110,
	SP_ERROE_FW_POSITION			= -111,
	SP_EEPROM_READ_ERROR			= -112,
	SP_ERROR_EEPROM_EMPTY			= -113,
	SP_ERROR_DATA_LACK				= -114,
	SP_ERROR_NOTFINDDEVICE			= -115,
	SP_ERROR_ALREAYDOPEN			= -116,
	SP_ERROR_WAIT_TIMEOUT			= -117,
	SP_ERROR_SCANNUM_RANGE			= -118,
	SP_ERROR_SMOOTH_RANGE			= -119,
	SP_ERROR_INVALIDVALUE			= -120,
	SP_ERROR_INVALID_INPUTCHANNEL	= -121,
	SP_ERROR_CHECKINTRERFACE		= -122,

	SP_ERROR_COM_SETTING			= -201,
	SP_ERROR_COM_READ				= -202,
	SP_ERROR_COM_WRITE				= -203,
	SP_ERROR_COM_OPERATING			= -204,
	SP_ERROR_COM_NOTMATCHDEV		= -205,
	SP_ERROR_COM_NOTCONNECTION		= -206,

	SP_ERROR_ETH_IPSCAN				= -301,
	SP_ERROR_ETH_NOTMATCHDEV		= -302,
	SP_ERROR_ETH_SOCKETCREATE		= -303,
	SP_ERROR_ETH_SOCKETCONNECT		= -304,
	SP_ERROR_ETH_TIMEOUT			= -305,
	SP_ERROR_ETH_RECVBUFFER			= -307,
	SP_EEROR_ETH_DISCONNECT			= -308,
	SP_ERROR_ETH_SENDPACKET			= -309,
	SP_ERROR_ETH_PACKET_SIZE_SMALL	= -310,
};

#pragma pack(push,1)
typedef struct _DevInformation
{
	char strCOM[100];
	char strModel[100];
	char strSerial[100];
	char strIPAddr[100];
	char strStaticIPAddr[100];
	char strMACAddr[100];
	double dWLTable[3648];
	int iDummyPixelNum;
	int iInttime;
	int iTimeavg;
	int iTotPixelNum;
	int iRealPixelNum;
	short sTrgMode;
	short sInterfaceType;
	short sNetMode;
	short sChannel;
	short sCCDType;
} DevInformation, * PDevInformation;
#pragma pack(pop)

#pragma pack(push,1)
typedef struct _DevceList
{
	short sInterfaceType;
	char cCOM[100];
	char cModel[100];
	char cSerial[100];
	char cIPAddr[100];
} DeviceList, * PDeviceList;
#pragma pack(pop)

typedef struct _MEMORYSTATE
{
	short sMemState;
	char MemName[100];
}MEMORYSTATE, * PMEMORYSTATE;

DLLIMPORT short CALLTYPE spNConnect(short sInterFace, char* cConnectAddr);
DLLIMPORT short CALLTYPE spNDevInfo(char* Model, char* Serial, short* sInterfaceType, short Channel);
DLLIMPORT short CALLTYPE spNGetCCDType(short* sCCDType, short Channel);
DLLIMPORT short CALLTYPE spNGetEEPROM(char* pcEEPData, short Channel);
DLLIMPORT short CALLTYPE spNSetIntTime(int iIntTime, short Channel);
DLLIMPORT short CALLTYPE spNSetDBLIntTime(double dIntTime, short Channel);
DLLIMPORT short CALLTYPE spNSetTimeAvg(short sAvgTime, short Channel);
DLLIMPORT short CALLTYPE spNSetTrgMode(short sTrgMode, short Channel);
DLLIMPORT short CALLTYPE spNSetDevice(int iIntTime, short sAverage, short sTrgMode, short Channel);
DLLIMPORT short CALLTYPE spNGetWLTable(double* WLTable, short Channel);
DLLIMPORT short CALLTYPE spNReadDataEx(long* lpArray, short Channel);
DLLIMPORT short CALLTYPE spNGetNETInfo(char* pcIPAddr, char* pcMACAddr, short* sNetMode, short Channel);
DLLIMPORT bool CALLTYPE spNCheckConnection(short Channel);
DLLIMPORT short CALLTYPE spNScanDevice(short sScanMode = SP_SCAN_CONNECTABLE);
DLLIMPORT short CALLTYPE spNGetDeviceList(_DevceList* DevList);
DLLIMPORT short CALLTYPE spNGetDevParam(_DevInformation* pDevInfo, short Channel);
DLLIMPORT short CALLTYPE spNDevClose(short Channel);

DLLIMPORT short CALLTYPE spNGetMemState(short sMemPage, PMEMORYSTATE stMemArray, short Channel);
DLLIMPORT short CALLTYPE spNGetMemData(short sMemPage, char btAddress, long* dataArray, short Channel);
DLLIMPORT short CALLTYPE spNSaveDataToMem(short sMemPage, char btAddress, char* cpName, long* lDataArray, short Channel);
DLLIMPORT short CALLTYPE spNDeleteMemData(short sMemPage, char btAddress, short Channel);

DLLIMPORT short CALLTYPE spNGetGainOffsetValue(int* piGainValue, int* piOffsetValue, short Channel);
DLLIMPORT short CALLTYPE spNSetGainOffsetValue(int iGainValue, int iOffsetValue, short Channel);

DLLIMPORT short CALLTYPE spNStartBurstData(short sCount, short sChannel);
DLLIMPORT short CALLTYPE spNGetBurstData(short sCount, long** dpDataList,short sMode, short Channel);

DLLIMPORT short CALLTYPE spNAutoDark(short sAutoDark, short Channel);

DLLIMPORT short CALLTYPE spNReadDataExOutTrg(long* lpArray, short Channel);
DLLIMPORT short CALLTYPE spNSetOutTrgPin(short sOutPin,short sDelay, short sChannel = 0);