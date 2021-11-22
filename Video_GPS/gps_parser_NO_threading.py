from serial import Serial
from pynmeagps import NMEAReader, NMEAMessage, GET, POLL, NMEA_MSGIDS
import pynmeagps
from io import BufferedReader
import cv2, time

def main():
	cap = cv2.VideoCapture(0)
	cap.set(3, 1920)
	cap.set(4, 1080)
	frame_width = int(cap.get(3))
	frame_height = int(cap.get(4))

	# GPS open COM port, read it and create file to write gps data later
	stream = Serial("COM4")
	nms = NMEAReader(stream, msgmode=GET)
	f = open("gps_DATA.txt","w+")

	i = 1
	starting = True
	while (True):
		ret, frame = cap.read()
		if starting:
			start = time.perf_counter()
			print("started video num:", i)
			name = str(i) + "output.avi"	
			out = cv2.VideoWriter(name, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width, frame_height))
			starting = False

		if (cap.isOpened()== False):
			print("Error opening video stream or file")
		if ret == True: 
			# ///////////////////////////////////////////////////////////////
			# GPS read, write gps data
			(raw_data, parsed_data) = nms.read()
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
				break
		
	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()
