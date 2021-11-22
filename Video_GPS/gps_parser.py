from serial import Serial
from pynmeagps import NMEAReader, NMEAMessage, GET, POLL, NMEA_MSGIDS
import pynmeagps
from io import BufferedReader
import cv2, time
from threading import Thread

class WebcamVideoStream:
	def __init__(self):
		self.stream = cv2.VideoCapture(0)
		self.stream.set(3, 1920)  
		self.stream.set(4, 1080)
		(self.grabbed, self.frame) = self.stream.read()
		self.stopped = False

	def start(self):
		t = Thread(target=self.update, args=( ))
		t.daemon = True
		t.start()
		return self

	def update(self):
		while True:
			if self.stopped:
				return
			(self.grabbed, self.frame) = self.stream.read()
			if (self.stream.isOpened()== False):
				print("Error opening video stream or file")

	def read(self):
		return self.frame

	def stop(self):
		self.stopped = True
		self.stream.release()
		cv2.destroyAllWindows()




class GPS:
	def __init__(self):
		self.stream = Serial("COM4")
		self.nms = NMEAReader(self.stream, msgmode=GET)
		self.stopped = False

	def start(self):
		t = Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		while True:
			if self.stopped:
				return
			(self.raw_data, self.parsed_data) = self.nms.read()

	def read(self):
		return self.parsed_data

	def stop(self):
		self.stopped = True


starting = True
gps = GPS()
gps.start()
stream = WebcamVideoStream()
stream.start()
f = open("gps_DATA.txt","w+")
i = 1
while (True):
	frame = stream.read()
	if starting:
		start = time.perf_counter()
		print("started video num:", i)
		name = str(i) + "output.avi"	
		out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (1920, 1080))
		starting = False

	# ////////////# GPS read, write gps data////////////////////////////
	parsed_data = gps.read()
	# print(pynmeagps.NMEA_TALKERS)
	if parsed_data.msgID == "GGA":
		data = "Time:" + str(parsed_data.time) + "\n" + "\t\tLat:" + str(parsed_data.lat) + parsed_data.NS + "\n"
		data = data + "\t\tLon:" + str(parsed_data.lon) + parsed_data.EW + "\n" + "\t\tAlt:" + str(parsed_data.alt) + "\n" + "/////////////////////////////////////////////////////////////////////////" + "\n"
		f.write(data)
	# /////////////////////////////////////////////////////////////// 
	out.write(frame)   
	cv2.imshow('frame',frame)	
	time1 = 10
	if(time.perf_counter() - start > time1):
		starting = True
		i += 1
	if cv2.waitKey(1) & 0xFF == 27:
		stream.stop()
		break
